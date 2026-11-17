"""BTI-Net: EfficientNet-B4 encoder with MSCF residual blocks, global context block, four decoder levels with region
context, task interaction module (TIM) and utility-predictive attention (UPA), multi-scale segmentation head and
presence-aware class posterior."""
import math

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

DEC = (384, 192, 96, 48)
ENC = {"s1": 144, "s2": 192, "s3": 336, "s4": 960, "br": 1792}
SKIPS, CLF_SRC = ("s4", "s3", "s2", "s1"), ("br", "s4", "s3", "s2")
TAPS = (("s1", (1, 0)), ("s2", (2, 0)), ("s3", (3, 0)), ("s4", (5, 0)))
CLF_C = 256
KAPPA = 0.02
REGION_LEVELS = (0, 1)
DS_SEG_W, DS_CLF_W, DS_CAM_W, REGION_W = 0.4, 0.1, 0.1, 0.4
BYPASS_P = 0.3
BN_MOMENTUM = 0.1
ABLATIONS = ("no_tim", "no_c2s", "no_s2c", "gate_one", "gate_const", "no_gsup", "gate_aiw", "no_gctx_pres")


def bn_layer(c, d=2):
    return (nn.BatchNorm2d if d == 2 else nn.BatchNorm1d)(c, eps=1e-3, momentum=0.01)


class SepConv(nn.Module):
    def __init__(self, cin, cout, dilation=1, relu=False):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, padding=dilation, dilation=dilation, groups=cin, bias=False)
        self.pw = nn.Conv2d(cin, cout, 1)
        self.relu = relu

    def forward(self, x):
        x = self.pw(self.dw(x))
        return F.relu(x) if self.relu else x


class MultiScaleContextFusion(nn.Module):
    """MSCF: dilated separable branches (1, 2, 4) mixed by a per-image softmax over scales."""

    def __init__(self, c, reduction=8):
        super().__init__()
        self.branches = nn.ModuleList([SepConv(c, c, d) for d in (1, 2, 4)])
        self.att = nn.Sequential(nn.Linear(c, c // reduction), nn.ReLU(), nn.Linear(c // reduction, 3))
        self.fusion = nn.Conv2d(c, c, 1)

    def forward(self, x):
        w = torch.softmax(self.att(x.mean((2, 3))), dim=1)
        return self.fusion(sum(b(x) * w[:, i, None, None, None] for i, b in enumerate(self.branches)))


class DualPathAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.channel = nn.Sequential(nn.Linear(c, max(8, c // 16)), nn.ReLU(), nn.Linear(max(8, c // 16), c), nn.Sigmoid())
        self.spatial = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x):
        cw = self.channel(x.mean((2, 3)))[:, :, None, None]
        sw = torch.sigmoid(self.spatial(torch.cat([x.amax(1, keepdim=True), x.mean(1, keepdim=True)], 1)))
        return x * cw * sw


class ResidualBlock(nn.Module):
    def __init__(self, cin, c, attention=True, mscf=True):
        super().__init__()
        self.conv1, self.bn1 = SepConv(cin, c, relu=True), bn_layer(c)
        self.conv2, self.bn2 = SepConv(c, c), bn_layer(c)
        self.mscf = MultiScaleContextFusion(c) if mscf else None
        self.att = DualPathAttention(c) if attention else None
        self.proj = nn.Conv2d(cin, c, 1) if cin != c else None

    def forward(self, x):
        y = self.bn2(self.conv2(self.bn1(self.conv1(x))))
        if self.mscf is not None:
            y = self.mscf(y)
        if self.att is not None:
            y = self.att(y)
        return F.relu(y + (self.proj(x) if self.proj is not None else x))


class AttentionGate(nn.Module):
    def __init__(self, x_ch, g_ch, inter):
        super().__init__()
        self.tx, self.tg = nn.Conv2d(x_ch, inter, 1), nn.Conv2d(g_ch, inter, 1)
        self.context = nn.Sequential(SepConv(inter, inter, 1), SepConv(inter, inter, 2), nn.Conv2d(inter, 1, 1), nn.Sigmoid())

    def forward(self, x, g):
        return x * self.context(F.relu(self.tx(x) + self.tg(g)))


class UpBlock(nn.Module):
    def __init__(self, cin, c):
        super().__init__()
        self.conv, self.bn = nn.Conv2d(cin, c, 1, bias=False), bn_layer(c)

    def forward(self, x):
        return F.interpolate(F.relu(self.bn(self.conv(x))), scale_factor=2, mode="bilinear", align_corners=False)


class DecoderBlock(nn.Module):
    def __init__(self, cin, c):
        super().__init__()
        self.conv1, self.bn1 = nn.Conv2d(cin, c, 3, padding=1, bias=False), bn_layer(c)
        self.conv2, self.bn2 = nn.Conv2d(c, c, 3, padding=1, bias=False), bn_layer(c)
        self.mscf = MultiScaleContextFusion(c)
        self.proj = nn.Conv2d(cin, c, 1, bias=False)

    def forward(self, x):
        y = self.mscf(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))))
        return F.relu(y + self.proj(x))


class DropPath(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        keep = (torch.rand(x.shape[0], 1, 1, device=x.device) >= self.p).to(x.dtype) / (1 - self.p)
        return x * keep


class GCBlock(nn.Module):
    def __init__(self, d, heads, mlp, drop, dp, ls):
        super().__init__()
        self.h = heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, d * mlp), nn.GELU(), nn.Dropout(drop), nn.Linear(d * mlp, d), nn.Dropout(drop))
        self.pdrop, self.dp = nn.Dropout(drop), DropPath(dp)
        self.g1, self.g2 = nn.Parameter(ls * torch.ones(d)), nn.Parameter(ls * torch.ones(d))

    def forward(self, t):
        B, N, D = t.shape
        q, k, v = self.qkv(self.n1(t)).view(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, D)
        t = t + self.dp(self.g1 * self.pdrop(self.proj(a)))
        return t + self.dp(self.g2 * self.mlp(self.n2(t)))


class GlobalContext(nn.Module):
    """Two transformer layers on the 7 x 7 bottleneck, added residually (zero-initialised output)."""

    def __init__(self, c=1792, d=384, heads=6, layers=2, mlp=2, drop=0.1, dp=0.1, ls=0.1, grid=7):
        super().__init__()
        self.inp = nn.Conv2d(c, d, 1)
        self.peg = nn.Conv2d(d, d, 3, padding=1, groups=d)
        self.pos = nn.Parameter(torch.zeros(1, d, grid, grid))
        self.blocks = nn.ModuleList([GCBlock(d, heads, mlp, drop, dp, ls) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.out = nn.Conv2d(d, c, 1)
        nn.init.trunc_normal_(self.pos, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02); nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.inp.weight); nn.init.zeros_(self.inp.bias)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        t = self.inp(x)
        t = t + self.peg(t) + self.pos.to(t.dtype)
        t = t.flatten(2).transpose(1, 2)
        for b in self.blocks:
            t = b(t)
        t = self.norm(t).transpose(1, 2).reshape(B, -1, H, W)
        return x + self.out(t).to(x.dtype)


class RegionContext(nn.Module):
    """Attention of every position to a lesion and a background prototype, pooled under a predicted region map."""

    def __init__(self, c):
        super().__init__()
        h = c // 2
        self.region = nn.Conv2d(c, 1, 1)
        self.phi, self.phi_bn = nn.Conv2d(c, h, 1, bias=False), bn_layer(h)
        self.theta, self.psi, self.eta = nn.Conv2d(c, h, 1), nn.Linear(h, h), nn.Linear(h, h)
        self.fuse, self.bn = nn.Conv2d(c + h, c, 1, bias=False), bn_layer(c)
        self.out = nn.Conv2d(c, c, 1)
        self.scale = h ** -0.5

    def forward(self, D):
        B, C, H, W = D.shape
        r = self.region(D)
        s = torch.sigmoid(r)
        z = F.relu(self.phi_bn(self.phi(D)))
        kappa = KAPPA * H * W
        g = z.mean((2, 3))
        les = ((z * s).sum((2, 3)) + kappa * g) / (s.sum((2, 3)) + kappa)
        bg = ((z * (1 - s)).sum((2, 3)) + kappa * g) / ((1 - s).sum((2, 3)) + kappa)
        P = torch.stack([les, bg], 1)
        att = torch.softmax(torch.bmm(self.theta(D).flatten(2).transpose(1, 2), self.psi(P).transpose(1, 2)) * self.scale, -1)
        y = torch.bmm(att, self.eta(P)).transpose(1, 2).reshape(B, -1, H, W)
        return D + self.out(F.relu(self.bn(self.fuse(torch.cat([D, y], 1))))), r


class TaskInteraction(nn.Module):
    """TIM. seg->cls: decoder features pooled under the level's lesion map update the class vector through a sigmoid
    gate. cls->seg: the class vector modulates the decoder features (FiLM) together with a class-evidence map."""

    def __init__(self, seg_c, num_classes, directions=("s2c", "c2s")):
        super().__init__()
        self.directions = tuple(directions)
        if "s2c" in self.directions:
            self.s_conv = nn.Conv2d(seg_c, seg_c // 2, 1)
            self.s_fc = nn.Linear(2 * (seg_c // 2) + 1, CLF_C)
            self.s_bn, self.clf_gate = bn_layer(CLF_C, 1), nn.Linear(CLF_C, CLF_C)
        if "c2s" in self.directions:
            self.gamma, self.beta = nn.Linear(CLF_C, seg_c), nn.Linear(CLF_C, seg_c)
            self.e_conv = nn.Conv2d(seg_c, num_classes, 1)
            self.alpha = nn.Parameter(torch.zeros(1, seg_c, 1, 1))

    def cam(self, D):
        return self.e_conv(D)

    def forward(self, D, f, a, p):
        f_enh, D_enh = f, D
        if "s2c" in self.directions:
            z = F.relu(self.s_conv(D))
            kappa = KAPPA * D.shape[-1] * D.shape[-2]
            att = ((z * a).sum((2, 3)) + kappa * z.mean((2, 3))) / (a.sum((2, 3)) + kappa)
            s = self.s_bn(F.relu(self.s_fc(torch.cat([att, z.mean((2, 3)), a.mean((2, 3))], 1))))
            f_enh = f + torch.sigmoid(self.clf_gate(s)) * s
        if "c2s" in self.directions:
            M = (p[:, :, None, None] * self.e_conv(D)).sum(1, keepdim=True)
            D_enh = D * (1.0 + torch.tanh(self.gamma(f)[:, :, None, None] + self.alpha * M)) + self.beta(f)[:, :, None, None]
        return D_enh, f_enh


class UPAGate(nn.Module):
    """UPA: per-image coefficients (w_seg, w_cls) from a five-signal descriptor of the exchange."""

    def __init__(self, seg_c, proj=64):
        super().__init__()
        self.P_s = nn.Linear(seg_c, proj, bias=False)
        self.P_c = nn.Linear(CLF_C, proj, bias=False)
        self.mlp = nn.Sequential(nn.Linear(5, 32), nn.ReLU(), nn.Linear(32, 2))

    @staticmethod
    def spatial_variation(D):
        dh = (D[:, :, 1:, :] - D[:, :, :-1, :]).abs().mean((1, 2, 3))
        dw = (D[:, :, :, 1:] - D[:, :, :, :-1]).abs().mean((1, 2, 3))
        return 0.5 * (dh + dw)

    def descriptor(self, D, D_enh, f, f_enh, extra):
        us, uc = self.P_s((D_enh - D).mean((2, 3))), self.P_c(f_enh - f)
        a = (us * uc).sum(1) / ((us.norm(dim=1) + 1e-8) * (uc.norm(dim=1) + 1e-8))
        u = torch.stack([a, self.spatial_variation(D), torch.log1p(f_enh.var(dim=1, unbiased=False))], 1)
        return torch.cat([u, extra.float()], 1)

    def forward(self, D, D_enh, f, f_enh, extra):
        logits = self.mlp(self.descriptor(D, D_enh, f, f_enh, extra))
        return torch.sigmoid(logits), logits


class FixedGate(nn.Module):
    """Ablation (e): w = 1."""

    def forward(self, D, D_enh, f, f_enh, extra=None):
        return D.new_ones((D.shape[0], 2), dtype=torch.float32), None


class ConstGate(nn.Module):
    """Ablation (f): one learned constant pair per level."""

    def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(2))

    def forward(self, D, D_enh, f, f_enh, extra=None):
        return torch.sigmoid(self.theta.float()).expand(D.shape[0], 2), None


def activation_dispersion(x):
    x = x.float()
    if x.dim() == 4:
        return x.var(dim=(2, 3), unbiased=False).mean(1, keepdim=True)
    return x.var(dim=1, unbiased=False, keepdim=True)


class AIWGate(nn.Module):
    """Ablation (h): AIW weighting, a softmax over the two streams from the dispersion of the exchanged features,
    normalised by its training-set mean after training (calibrate_aiw)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 2))
        self.register_buffer("pop_seg", torch.zeros(()))
        self.register_buffer("pop_clf", torch.zeros(()))
        self.register_buffer("calibrated", torch.zeros(()))
        self.collect = None

    def forward(self, D, D_enh, f, f_enh, extra=None):
        vs, vc = activation_dispersion(D_enh), activation_dispersion(f_enh)
        if self.collect is not None:
            self.collect.append((vs.sum().item(), vc.sum().item(), D.shape[0]))
        if not self.training and self.calibrated > 0:
            ms, mc = self.pop_seg, self.pop_clf
        else:
            ms, mc = vs.mean(), vc.mean()
        out = self.net(torch.cat([vs / (ms + 1e-8), vc / (mc + 1e-8)], 1))
        return torch.softmax(out, 1), out


class BTINet(nn.Module):
    def __init__(self, num_classes, ablation=None, pretrained=True):
        super().__init__()
        assert ablation is None or ablation in ABLATIONS, ablation
        self.ablation, self.K = ablation, num_classes
        self.tim = ablation != "no_tim"
        directions = {"no_c2s": ("s2c",), "no_s2c": ("c2s",)}.get(ablation, ("s2c", "c2s"))
        self.gate = {"gate_one": "one", "gate_const": "const", "gate_aiw": "aiw"}.get(ablation, "upa")
        self.supervised = self.tim and self.gate == "upa" and ablation != "no_gsup"
        context = ablation != "no_gctx_pres"

        self.enc = timm.create_model("tf_efficientnet_b4.aa_in1k", pretrained=pretrained, drop_path_rate=0.2)
        self.enc.classifier = nn.Identity()
        self._feats = {}
        for name, (i, j) in TAPS:
            self.enc.blocks[i][j].bn1.register_forward_hook(self._hook(name))
        self.enc.bn2.register_forward_hook(self._hook("br"))
        self.res = nn.ModuleDict({k: ResidualBlock(c, c) for k, c in ENC.items()})
        self.gctx = GlobalContext() if context else None
        self.ups = nn.ModuleList([UpBlock(ci, c) for ci, c in zip((ENC["br"],) + DEC[:-1], DEC)])
        self.gates = nn.ModuleList([AttentionGate(ENC[s], c, c // 2) for s, c in zip(SKIPS, DEC)])
        self.dblocks = nn.ModuleList([DecoderBlock(c + ENC[s] + (1 if l > 0 else 0), c) for l, (s, c) in enumerate(zip(SKIPS, DEC))])
        self.octx = nn.ModuleList([RegionContext(c) if l in REGION_LEVELS else nn.Identity() for l, c in enumerate(DEC)])
        self.clf_fc = nn.ModuleList([nn.Linear(ENC[s], CLF_C) for s in CLF_SRC])
        self.prop = nn.ModuleList([nn.Linear(CLF_C, CLF_C) for _ in DEC[1:]])
        self.ds_seg = nn.ModuleList([nn.Conv2d(c, 1, 1) for c in DEC])
        self.ds_clf = nn.ModuleList([nn.Linear(CLF_C, num_classes) for _ in DEC])
        self.tims = nn.ModuleList([TaskInteraction(c, num_classes, directions) if self.tim else nn.Identity() for c in DEC])
        gate = {"upa": UPAGate, "one": lambda c: FixedGate(), "const": lambda c: ConstGate(), "aiw": lambda c: AIWGate()}[self.gate]
        self.blend_gates = nn.ModuleList([gate(c) if self.tim else nn.Identity() for c in DEC])
        self.ms = nn.Sequential(nn.Conv2d(sum(DEC), 64, 1, bias=False), bn_layer(64), nn.ReLU(),
                                nn.Conv2d(64, 64, 3, padding=1, bias=False), bn_layer(64), nn.ReLU(), nn.Conv2d(64, 1, 1))
        self.clf_head = nn.Sequential(nn.Linear(ENC["s2"] + ENC["s3"] + ENC["s4"] + ENC["br"] + 4 * CLF_C, 256), nn.ReLU(),
                                      bn_layer(256, 1), nn.Dropout(0.3), nn.Linear(256, num_classes))
        self.pres = context
        if self.pres:
            self.pres_alpha, self.pres_beta = nn.Parameter(torch.zeros(1)), nn.Parameter(torch.zeros(1))
            self.pres_tau = nn.Parameter(torch.full((1,), math.log(math.e - 1)))
            self.register_buffer("pres_empty", torch.full((1,), -1, dtype=torch.long))
        self._empty, self._cf = -1, False
        self.log_k = float(np.log(num_classes))

        for name, m in self.named_modules():
            if name.split(".")[0] not in ("enc", "gctx") and isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for t in self.tims:
            if isinstance(t, TaskInteraction) and "c2s" in t.directions:
                for m in (t.gamma, t.beta):
                    nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
        for m in self.prop:
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
        for m in self.octx:
            if isinstance(m, RegionContext):
                nn.init.zeros_(m.out.weight); nn.init.zeros_(m.out.bias)
        with torch.no_grad():
            for blk in list(self.dblocks)[1:]:
                blk.conv1.weight[:, -1].zero_(); blk.proj.weight[:, -1].zero_()
        for m in self.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.momentum = BN_MOMENTUM
        mean, std = self.enc.pretrained_cfg["mean"], self.enc.pretrained_cfg["std"]
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1) * 255)

    def _hook(self, name):
        def f(module, inp, out):
            self._feats[name] = out
        return f

    def set_empty(self, k):
        """Index of the class whose images carry an empty mask (-1: none); enables the presence-aware posterior."""
        self._empty = int(k)
        if self.pres:
            self.pres_empty.fill_(self._empty)

    def load_state_dict(self, sd, *a, **kw):
        r = super().load_state_dict(sd, *a, **kw)
        if self.pres:
            self._empty = int(self.pres_empty.item())
        return r

    def encode(self, x):
        self._feats = {}
        self.enc.forward_features((x - self.mean) / self.std)
        e = {k: self.res[k](v) for k, v in self._feats.items()}
        if self.gctx is not None:
            e["br"] = self.gctx(e["br"])
        return e

    @staticmethod
    def pooled(e):
        return [e[k].float().mean((2, 3)) for k in ("s2", "s3", "s4", "br")]

    @staticmethod
    def blend(Z, Z_enh, w):
        w = w.view((-1,) + (1,) * (Z.dim() - 1))
        return Z.float() + w * (Z_enh.float() - Z.float())

    def level(self, lvl, d, e, f_prev, pm, regions=None):
        """Decoder level: upsampling, gated skip, the previous level's lesion map, region context; the level's class
        vector; and the TIM exchange (D_enh, f_enh)."""
        up = self.ups[lvl](d)
        parts = [up, self.gates[lvl](e[SKIPS[lvl]], up)]
        if lvl > 0:
            parts.append(F.interpolate(pm, size=up.shape[-2:], mode="bilinear", align_corners=False).to(up.dtype))
        D = self.dblocks[lvl](torch.cat(parts, 1))
        if lvl in REGION_LEVELS:
            D, r = self.octx[lvl](D)
            if regions is not None:
                regions.append(r)
        f = F.relu(self.clf_fc[lvl](e[CLF_SRC[lvl]].mean((2, 3))))
        if lvl > 0:
            f = f + self.prop[lvl - 1](f_prev)
        if self.tim:
            a = torch.sigmoid(self.ds_seg[lvl](D))
            p = torch.softmax(self.ds_clf[lvl](f).float(), 1).detach().to(D.dtype)
            D_enh, f_enh = self.tims[lvl](D, f, a, p)
        else:
            D_enh, f_enh = D, f
        return D, f, D_enh, f_enh

    def extra(self, lvl, D, f):
        """The level's class entropy and lesion-map mean (the two UPA signals from the deep-supervision heads)."""
        with torch.autocast(D.device.type, enabled=False):
            p = torch.softmax(self.ds_clf[lvl](f.float()), 1)
            h = -(p * torch.log(p.clamp_min(1e-8))).sum(1) / self.log_k
            rho = torch.sigmoid(self.ds_seg[lvl](D.float())).mean((1, 2, 3))
            return torch.stack([h, rho], 1)

    def lesion_map(self, lvl, D_fin):
        return torch.sigmoid(self.ds_seg[lvl](D_fin.float()))

    def seg_logits(self, douts, size):
        with torch.autocast(douts[-1].device.type, enabled=False):
            ref = douts[-1]
            z = torch.cat([ref.float()] + [F.interpolate(v.float(), size=ref.shape[-2:], mode="bilinear", align_corners=False)
                                           for v in douts[:-1]], 1)
            return F.interpolate(self.ms(z), size=size, mode="bilinear", align_corners=False)

    def clf_logits(self, pooled, f_fins):
        if self.training and not self._cf:
            keep = (torch.rand(f_fins[0].shape[0], 1, device=f_fins[0].device) > BYPASS_P).float() / (1 - BYPASS_P)
            pooled = [v * keep for v in pooled]
        with torch.autocast(f_fins[0].device.type, enabled=False):
            return self.clf_head(torch.cat([v.float() for v in pooled] + [v.float() for v in f_fins], 1))

    def outputs(self, douts, size, pooled, f_fins):
        """Segmentation map and class posterior; with an empty-mask class, the presence-aware posterior."""
        if not self.pres or self._empty < 0:
            clf = torch.softmax(self.clf_logits(pooled, f_fins), 1)
            return torch.sigmoid(self.seg_logits(douts, size)), clf
        lg = self.seg_logits(douts, size)
        k, K = self._empty, self.K
        other = [i for i in range(K) if i != k]
        with torch.autocast(lg.device.type, enabled=False):
            lg = lg.float()
            lo = self.clf_logits(pooled, f_fins)
            z0 = torch.logsumexp(lo[:, other], 1) - lo[:, k]
            tau = F.softplus(self.pres_tau) + 0.05
            s = tau * (torch.logsumexp(lg.flatten(1) / tau, 1) - float(np.log(lg[0].numel())))
            z = z0 + self.pres_alpha * s + self.pres_beta
            t = torch.softmax(lo[:, other], 1) * torch.sigmoid(z)[:, None]
            cols, j = [], 0
            for i in range(K):
                if i == k:
                    cols.append(torch.sigmoid(-z)[:, None])
                else:
                    cols.append(t[:, j:j + 1]); j += 1
            return torch.sigmoid(lg), torch.cat(cols, 1)

    def _gate(self, lvl, D, D_enh, f, f_enh, ex):
        with torch.autocast(D.device.type, enabled=False):
            return self.blend_gates[lvl](D.float(), D_enh.float(), f.float(), f_enh.float(), ex)

    def gate_logits(self, o, l):
        """UPA logits of level l recomputed from the cached forward pass (gradient reaches only the gate)."""
        g, c = self.blend_gates[l], o["cache"][l]
        with torch.autocast(c["D"].device.type, enabled=False):
            return g.mlp(g.descriptor(c["D"].detach().float(), c["D_enh"].detach().float(), c["f"].detach().float(),
                                      c["f_enh"].detach().float(), c["extra"].detach()))

    def forward(self, x, keep=False):
        e = self.encode(x)
        d = e["br"]
        ws, f_fins, cache, douts, aux = [], [], [], [], []
        f_prev, pm = None, None
        for lvl in range(4):
            d_in = d
            regions = [] if self.training else None
            D, f, D_enh, f_enh = self.level(lvl, d, e, f_prev, pm, regions)
            if regions:
                with torch.autocast(D.device.type, enabled=False):
                    rl = F.interpolate(regions[0].float(), size=x.shape[-2:], mode="bilinear", align_corners=False)
                    aux.append(({"seg": torch.sigmoid(rl)}, REGION_W))
            ex = self.extra(lvl, D, f).detach() if self.tim and self.gate == "upa" else None
            if self.tim:
                w, _ = self._gate(lvl, D, D_enh, f, f_enh, ex)
                wb = w.detach() if self.supervised else w
                with torch.autocast(D.device.type, enabled=False):
                    D_fin, f_fin = self.blend(D, D_enh, wb[:, 0]), self.blend(f, f_enh, wb[:, 1])
            else:
                w, D_fin, f_fin = None, D, f
            if self.training:
                with torch.autocast(D.device.type, enabled=False):
                    sl = F.interpolate(self.ds_seg[lvl](D_fin.float()), size=x.shape[-2:], mode="bilinear", align_corners=False)
                    aux.append(({"seg": torch.sigmoid(sl)}, DS_SEG_W))
                    aux.append(({"clf": torch.softmax(self.ds_clf[lvl](f_fin.float()), 1)}, DS_CLF_W))
                    t = self.tims[lvl]
                    if isinstance(t, TaskInteraction) and "c2s" in t.directions:
                        aux.append(({"clf": torch.softmax(t.cam(D.float()).mean((2, 3)), 1)}, DS_CAM_W))
            pm_in, pm = pm, self.lesion_map(lvl, D_fin)
            d = D_fin
            douts.append(d); ws.append(w); f_fins.append(f_fin)
            f_prev = f_fin
            if keep:
                cache.append({"d_in": d_in, "D": D, "D_enh": D_enh, "f": f, "f_enh": f_enh, "extra": ex, "pm_in": pm_in})
        seg, clf = self.outputs(douts, x.shape[-2:], self.pooled(e), f_fins)
        out = {"seg": seg, "clf": clf, "w": ws, "aux": aux}
        if keep:
            out.update(e=e, cache=cache, f_fin=f_fins, douts=douts)
        return out

    def rollout(self, e, o, start, k, v, size):
        """Recomputes the decoder and both outputs from level `start` with coefficient k (0: w_seg, 1: w_cls) of that
        level forced to v and all other coefficients held at their forward-pass values."""
        d = o["cache"][start]["d_in"]
        outs, ffins = list(o["douts"][:start]), list(o["f_fin"][:start])
        f_prev = o["f_fin"][start - 1] if start > 0 else None
        pm = o["cache"][start]["pm_in"]
        for lvl in range(start, 4):
            D, f, D_enh, f_enh = self.level(lvl, d, e, f_prev, pm)
            w = o["w"][lvl].detach().float()
            ws_, wc_ = w[:, 0], w[:, 1]
            if lvl == start:
                ws_, wc_ = (v, wc_) if k == 0 else (ws_, v)
            D_fin, f_fin = self.blend(D, D_enh, ws_), self.blend(f, f_enh, wc_)
            pm = self.lesion_map(lvl, D_fin)
            d = D_fin
            outs.append(d); ffins.append(f_fin); f_prev = f_fin
        return self.outputs(outs, size, self.pooled(e), ffins)


@torch.no_grad()
def calibrate_aiw(model, X, to_device, batch=24):
    """Ablation (h): replaces the batch means of the AIW gates by their training-set means."""
    model.eval()
    stats = {}
    for lvl, g in enumerate(model.blend_gates):
        if not isinstance(g, AIWGate):
            continue
        g.collect = []
        for i in range(0, len(X), batch):
            with torch.autocast(model.mean.device.type, dtype=torch.float16, enabled=model.mean.is_cuda):
                model(to_device(X[i:i + batch]))
        ss, sc, n = map(sum, zip(*g.collect))
        g.collect = None
        g.pop_seg.fill_(ss / n); g.pop_clf.fill_(sc / n); g.calibrated.fill_(1.0)
        stats["L%d" % (lvl + 1)] = (ss / n, sc / n)
    return stats
