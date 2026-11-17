"""AIW: task interaction blended by adaptive interaction weighting, a per-image softmax over the two streams computed
from the dispersion of the exchanged features (no supervision of the weights)."""
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from btinet.model import AttentionGate, ResidualBlock, activation_dispersion, bn_layer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TaskInteractionModule(nn.Module):
    def __init__(self, seg_c, clf_c=256, tau=0.7):
        super().__init__()
        self.tau = tau
        self.s_conv = nn.Conv2d(seg_c, seg_c // 2, 1)
        self.s_fc, self.s_bn = nn.Linear(seg_c // 2, clf_c), bn_layer(clf_c, 1)
        self.clf_gate = nn.Linear(clf_c, clf_c)
        self.p_fc, self.g_fc = nn.Linear(clf_c, seg_c), nn.Linear(seg_c, seg_c)

    def forward(self, D, f):
        s = self.s_bn(F.relu(self.s_fc(F.relu(self.s_conv(D)).mean((2, 3)))))
        f_enh = f + torch.sigmoid(self.clf_gate(s)) * s
        p = torch.sigmoid(self.p_fc(f))[:, :, None, None]
        g = torch.sigmoid(self.g_fc(D.mean((2, 3))))[:, :, None, None]
        return D * (1.0 + self.tau * g * p), f_enh


class AdaptiveInteractionWeighting(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 2))
        self.register_buffer("pop_seg", torch.zeros(()))
        self.register_buffer("pop_clf", torch.zeros(()))
        self.register_buffer("calibrated", torch.zeros(()))
        self.collect = None

    def weights(self, D_enh, f_enh):
        vs, vc = activation_dispersion(D_enh), activation_dispersion(f_enh)
        if self.collect is not None:
            self.collect.append((vs.sum().item(), vc.sum().item(), D_enh.shape[0]))
        if not self.training and self.calibrated > 0:
            ms, mc = self.pop_seg, self.pop_clf
        else:
            ms, mc = vs.mean(), vc.mean()
        return torch.softmax(self.net(torch.cat([vs / (ms + 1e-8), vc / (mc + 1e-8)], 1)), 1)

    def forward(self, D, D_enh, f, f_enh):
        with torch.autocast("cuda", enabled=False):
            D, D_enh, f, f_enh = D.float(), D_enh.float(), f.float(), f_enh.float()
            w = self.weights(D_enh, f_enh)
            D_fin = D + w[:, 0, None, None, None] * (D_enh - D)
            f_fin = f + w[:, 1:2] * (f_enh - f)
        return D_fin, f_fin, w


class AdaptiveInteractionModel(nn.Module):
    TAPS = [("s1", (1, 0)), ("s2", (2, 0)), ("s3", (3, 0)), ("s4", (5, 0))]

    def __init__(self, num_classes=3, dropout=0.3, pretrained=True):
        super().__init__()
        self.enc = timm.create_model("tf_efficientnet_b4.aa_in1k", pretrained=pretrained, drop_path_rate=0.2)
        for m in self.enc.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = 0.01
        self.enc.classifier = nn.Identity()
        self._feats = {}
        for name, (i, j) in self.TAPS:
            self.enc.blocks[i][j].bn1.register_forward_hook(self._hook(name))
        self.enc.bn2.register_forward_hook(self._hook("br"))
        b2b = self.enc.blocks[1][1]
        self.frozen = [self.enc.conv_stem, self.enc.bn1, self.enc.blocks[0], self.enc.blocks[1][0],
                       b2b.conv_pw, b2b.bn1, b2b.conv_dw, b2b.bn2]
        for m in self.frozen:
            for p in m.parameters():
                p.requires_grad = False
        ch = {"s1": 144, "s2": 192, "s3": 336, "s4": 960, "br": 1792}
        self.res = nn.ModuleDict({k: ResidualBlock(c, c, attention=True) for k, c in ch.items()})
        dec = [384, 192, 96, 48]
        skips, ups_in, clf_src = ["s4", "s3", "s2", "s1"], [1792, 384, 192, 96], ["br", "s4", "s3", "s2"]
        self.ups = nn.ModuleList([nn.ConvTranspose2d(ci, c, 3, 2, padding=1, output_padding=1) for ci, c in zip(ups_in, dec)])
        self.gates = nn.ModuleList([AttentionGate(ch[s], c, c // 2) for s, c in zip(skips, dec)])
        self.dblocks = nn.ModuleList([ResidualBlock(c + ch[s], c, attention=False) for s, c in zip(skips, dec)])
        self.drop = nn.Dropout(dropout)
        self.clf_fc = nn.ModuleList([nn.Linear(ch[s], 256) for s in clf_src])
        self.tims = nn.ModuleList([TaskInteractionModule(c, 256) for c in dec])
        self.aiws = nn.ModuleList([AdaptiveInteractionWeighting() for _ in dec])
        self.skips, self.clf_src = skips, clf_src
        self.seg_head = nn.Sequential(nn.ConvTranspose2d(48, 24, 3, 2, padding=1, output_padding=1),
                                      nn.Conv2d(24, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 1, 1))
        self.clf_head = nn.Sequential(nn.Linear(192 + 336 + 960 + 1792 + 4 * 256, 256), nn.ReLU(),
                                      bn_layer(256, 1), nn.Dropout(0.3), nn.Linear(256, num_classes))
        for name, m in self.named_modules():
            if not name.startswith("enc") and isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        mean, std = self.enc.pretrained_cfg["mean"], self.enc.pretrained_cfg["std"]
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1) * 255)

    def _hook(self, name):
        def f(module, inp, out):
            self._feats[name] = out
        return f

    def train(self, mode=True):
        super().train(mode)
        for m in self.frozen:
            m.eval()
        return self

    def forward(self, x):
        self._feats = {}
        self.enc.forward_features((x - self.mean) / self.std)
        e = {k: self.res[k](v) for k, v in self._feats.items()}
        d, clf_vecs = e["br"], []
        for lvl in range(4):
            up = self.ups[lvl](d)
            d = self.dblocks[lvl](torch.cat([up, self.gates[lvl](e[self.skips[lvl]], up)], 1))
            f = F.relu(self.clf_fc[lvl](e[self.clf_src[lvl]].mean((2, 3))))
            D_enh, f_enh = self.tims[lvl](d, f)
            d, f, _ = self.aiws[lvl](d, D_enh, f, f_enh)
            d = self.drop(d)
            clf_vecs.append(f)
        with torch.autocast("cuda", enabled=False):
            seg = torch.sigmoid(self.seg_head(d.float()))
            z = torch.cat([e[k].float().mean((2, 3)) for k in ("s2", "s3", "s4", "br")] + [v.float() for v in clf_vecs], 1)
            clf = torch.softmax(self.clf_head(z), 1)
        return {"seg": seg, "clf": clf}


@torch.no_grad()
def calibrate_aiw(model, X, batch_size=24):
    """Replaces the batch means of each level's weighting by its training-set means."""
    model.eval()
    stats = {}
    for lvl, a in enumerate(model.aiws):
        a.collect = []
        for i in range(0, len(X), batch_size):
            model(torch.from_numpy(X[i:i + batch_size]).permute(0, 3, 1, 2).float().to(DEVICE))
        ss, sc, n = map(sum, zip(*a.collect))
        a.collect = None
        a.pop_seg.fill_(ss / n); a.pop_clf.fill_(sc / n); a.calibrated.fill_(1.0)
        stats["L%d" % (lvl + 1)] = (ss / n, sc / n)
    return stats
