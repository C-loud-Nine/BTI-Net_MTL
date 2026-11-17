"""The multi-task methods compared with BTI-Net (MTAN, MTI-Net, DenseMTL, MTANet, MTL-OCA, AIW), each on the shared
EfficientNet-B4 encoder."""
import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, Bottleneck

from .aiw import AdaptiveInteractionModel, calibrate_aiw

LABEL = {"mtan": "MTAN", "mti_net": "MTI-Net", "densemtl": "DenseMTL",
         "mtanet": "MTANet", "mtl_oca": "MTL-OCA", "aiw": "AIW"}
SHARED = "tf_efficientnet_b4.aa_in1k"


def conv_bn_relu(cin, cout, k=3, s=1, p=None, d=1):
    p = (k // 2) * d if p is None else p
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, p, dilation=d, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


def up_to(x, ref):
    return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)


def result(seg_logit, clf_logit, aux=(), size=224):
    s = seg_logit.float()
    if s.shape[-1] != size:
        s = F.interpolate(s, size=(size, size), mode="bilinear", align_corners=False)
    return {"seg": torch.sigmoid(s), "clf": torch.softmax(clf_logit.float(), 1), "aux": list(aux)}


def aux_seg(logit, w, size=224):
    s = logit.float()
    if s.shape[-1] != size:
        s = F.interpolate(s, size=(size, size), mode="bilinear", align_corners=False)
    return ({"seg": torch.sigmoid(s)}, w)


def aux_clf(logit, w):
    return ({"clf": torch.softmax(logit.float(), 1)}, w)


class MLPHead(nn.Sequential):
    def __init__(self, cin, n):
        super().__init__(nn.Linear(cin, 256), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(256, n))


class Encoder(nn.Module):
    def __init__(self, name=SHARED, pretrained=True, in_chans=3, **kw):
        super().__init__()
        if name.startswith("tf_efficientnet"):
            kw.setdefault("drop_path_rate", 0.2)
        self.net = timm.create_model(name, pretrained=pretrained, features_only=True, in_chans=in_chans, **kw)
        self.channels = self.net.feature_info.channels()
        self.reductions = self.net.feature_info.reduction()
        cfg = self.net.pretrained_cfg
        self.register_buffer("mean", torch.tensor(cfg["mean"]).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor(cfg["std"]).view(1, 3, 1, 1) * 255)

    def norm(self, x):
        rgb = (x[:, :3] - self.mean) / self.std
        return torch.cat([rgb, x[:, 3:]], 1) if x.shape[1] > 3 else rgb

    def forward(self, x):
        return self.net(self.norm(x))


class AttentionLayer(nn.Sequential):
    def __init__(self, cin, mid, cout):
        super().__init__(nn.Conv2d(cin, mid, 1), nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
                         nn.Conv2d(mid, cout, 1), nn.BatchNorm2d(cout), nn.Sigmoid())


class RefinementBlock(nn.Sequential):
    def __init__(self, cin, cout):
        ds = nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout))
        super().__init__(Bottleneck(cin, cout // 4, downsample=ds))


class MTAN(nn.Module):
    GROUPS = ((1,), (2,), (3, 4), (5, 6))

    def __init__(self, n_cls, pretrained=True, dec=(256, 128, 64, 32)):
        super().__init__()
        self.tasks = ("seg", "clf")
        b = timm.create_model(SHARED, pretrained=pretrained, drop_path_rate=0.2)
        self.stem = nn.Sequential(b.conv_stem, b.bn1)
        self.blocks = b.blocks
        cfg = b.pretrained_cfg
        self.register_buffer("mean", torch.tensor(cfg["mean"]).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor(cfg["std"]).view(1, 3, 1, 1) * 255)
        ch = [self.blocks[g[-1]][-1].conv_pwl.out_channels for g in self.GROUPS]
        self.ch = ch

        self.enc_att = nn.ModuleDict({t: nn.ModuleList([AttentionLayer(ch[0], ch[0] // 4, ch[0])] +
                                                       [AttentionLayer(2 * ch[i], ch[i] // 4, ch[i]) for i in range(1, 4)])
                                      for t in self.tasks})
        self.refine = nn.ModuleList([RefinementBlock(ch[i], ch[i + 1]) for i in range(3)])

        skips = [ch[2], ch[1], ch[0], 24]
        ins = [ch[3]] + list(dec[:-1])
        self.dec_a = nn.ModuleList([conv_bn_relu(ins[j] + skips[j], dec[j]) for j in range(4)])
        self.dec_b = nn.ModuleList([conv_bn_relu(dec[j], dec[j]) for j in range(4)])
        att_in = [ch[3]] + list(dec[:-1])
        self.dec_block_att = nn.ModuleList([conv_bn_relu(att_in[j], dec[j]) for j in range(4)])
        self.dec_att = nn.ModuleList([AttentionLayer(2 * dec[j], dec[j], dec[j]) for j in range(4)])
        self.seg_head = nn.Sequential(nn.Conv2d(dec[3], dec[3], 3, padding=1), nn.Conv2d(dec[3], 1, 1))
        self.clf_head = MLPHead(ch[3], n_cls)

    def forward(self, x):
        h = self.stem((x - self.mean) / self.std)
        h = self.blocks[0](h)
        s2 = h
        ub, ut = [], []
        for g in self.GROUPS:
            for i in g[:-1]:
                h = self.blocks[i](h)
            last = self.blocks[g[-1]]
            for blk in last[:-1]:
                h = blk(h)
            ub.append(h)
            h = last[-1](h)
            ut.append(h)
        att = {}
        for t in self.tasks:
            a = None
            for i in range(4):
                m = self.enc_att[t][i](ub[i] if i == 0 else torch.cat([ub[i], a], 1))
                a = m * ut[i]
                if i < 3:
                    a = F.max_pool2d(self.refine[i](a), 2, 2)
            att[t] = a
        skips = [ut[2], ut[1], ut[0], s2]
        d, A, D = ut[3], [], []
        for j in range(4):
            u = torch.cat([F.interpolate(d, size=skips[j].shape[-2:], mode="bilinear", align_corners=False), skips[j]], 1)
            a = self.dec_a[j](u)
            d = self.dec_b[j](a)
            A.append(a); D.append(d)
        y = att["seg"]
        for j in range(4):
            y = self.dec_block_att[j](F.interpolate(y, size=D[j].shape[-2:], mode="bilinear", align_corners=False))
            y = self.dec_att[j](torch.cat([A[j], y], 1)) * D[j]
        return result(self.seg_head(y), self.clf_head(att["clf"].mean((2, 3))), size=x.shape[-1])


class SEBlock(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.Sequential(nn.Linear(c, max(1, c // r)), nn.ReLU(), nn.Linear(max(1, c // r), c), nn.Sigmoid())

    def forward(self, x):
        return x * self.squeeze(x.mean((2, 3)))[:, :, None, None]


class SABlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.attention = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.Sigmoid())
        self.conv = nn.Conv2d(cin, cout, 3, padding=1, bias=False)

    def forward(self, x):
        return self.conv(x) * self.attention(x)


def basic(cin, cout):
    ds = None if cin == cout else nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout))
    return BasicBlock(cin, cout, downsample=ds)


class InitialTaskPrediction(nn.Module):
    def __init__(self, tasks, n_out, cin, c):
        super().__init__()
        self.tasks = tasks
        self.refinement = nn.ModuleDict({t: nn.Sequential(basic(cin, c), basic(c, c)) for t in tasks})
        self.decoders = nn.ModuleDict({t: nn.Conv2d(c, n_out[t], 1) for t in tasks})

    def forward(self, cur, prev=None):
        x = {t: cur if prev is None else torch.cat([cur, F.interpolate(prev[t], size=cur.shape[-2:], mode="bilinear",
                                                                      align_corners=False)], 1) for t in self.tasks}
        out = {}
        for t in self.tasks:
            out["features_" + t] = self.refinement[t](x[t])
            out[t] = self.decoders[t](out["features_" + t])
        return out


class FPM(nn.Module):
    def __init__(self, tasks, c):
        super().__init__()
        self.tasks, self.N = tasks, len(tasks)
        s = self.N * c
        self.non_linear = nn.Sequential(basic(s, s // 4), basic(s // 4, s // 4), nn.Conv2d(s // 4, s, 1))
        self.dimensionality_reduction = basic(s, c)
        self.se = nn.ModuleDict({t: SEBlock(c) for t in tasks})

    def forward(self, x):
        concat = torch.cat([x["features_" + t] for t in self.tasks], 1)
        B, C, H, W = concat.shape
        shared = self.non_linear(concat)
        mask = F.softmax(shared.view(B, C // self.N, self.N, H, W), dim=2)
        shared = (mask * concat.view(B, C // self.N, self.N, H, W)).view(B, -1, H, W)
        shared = self.dimensionality_reduction(shared)
        return {t: self.se[t](shared) + x["features_" + t] for t in self.tasks}


class Distillation(nn.Module):
    def __init__(self, tasks, c):
        super().__init__()
        self.tasks = tasks
        self.self_attention = nn.ModuleDict({t: nn.ModuleDict({a: SABlock(c, c) for a in tasks if a != t}) for t in tasks})

    def forward(self, x):
        return {t: x["features_" + t] + sum(self.self_attention[t][a](x["features_" + a]) for a in self.tasks if a != t)
                for t in self.tasks}


class HighResolutionHead(nn.Module):
    def __init__(self, chs, n_out):
        super().__init__()
        c = sum(chs)
        self.last_layer = nn.Sequential(nn.Conv2d(c, c, 1), nn.BatchNorm2d(c, momentum=0.1), nn.ReLU(inplace=False), nn.Conv2d(c, n_out, 1))

    def forward(self, xs):
        return self.last_layer(torch.cat([xs[0]] + [up_to(v, xs[0]) for v in xs[1:]], 1))


class MTINet(nn.Module):
    def __init__(self, n_cls, pretrained=True):
        super().__init__()
        self.tasks = ("seg", "clf")
        self.enc = Encoder(SHARED, pretrained, out_indices=(1, 2, 3, 4))
        c = self.enc.channels
        n_out = {"seg": 1, "clf": n_cls}
        self.scale = nn.ModuleList([InitialTaskPrediction(self.tasks, n_out, c[i] + c[i + 1], c[i]) for i in range(3)] +
                                   [InitialTaskPrediction(self.tasks, n_out, c[3], c[3])])
        self.fpm = nn.ModuleList([FPM(self.tasks, c[i]) for i in (1, 2, 3)])
        self.distil = nn.ModuleList([Distillation(self.tasks, c[i]) for i in range(4)])
        self.heads = nn.ModuleDict({t: HighResolutionHead(c, n_out[t]) for t in self.tasks})

    def forward(self, x):
        f = self.enc(x)
        x3 = self.scale[3](f[3]); p3 = self.fpm[2](x3)
        x2 = self.scale[2](f[2], p3); p2 = self.fpm[1](x2)
        x1 = self.scale[1](f[1], p2); p1 = self.fpm[0](x1)
        x0 = self.scale[0](f[0], p1)
        xs = [x0, x1, x2, x3]
        aux = []
        for s in xs:
            aux.append(aux_seg(s["seg"], 1.0, x.shape[-1]))
            aux.append(aux_clf(s["clf"].mean((2, 3)), 1.0))
        feats = [self.distil[i](xs[i]) for i in range(4)]
        seg = self.heads["seg"]([g["seg"] for g in feats])
        clf = self.heads["clf"]([g["clf"] for g in feats]).mean((2, 3))
        return result(seg, clf, aux, x.shape[-1])


class ASPPConv(nn.Sequential):
    def __init__(self, cin, cout, d):
        super().__init__(nn.Conv2d(cin, cout, 3, padding=d, dilation=d, bias=False), nn.BatchNorm2d(cout), nn.ReLU())


class ASPPPooling(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(nn.AdaptiveAvgPool2d(1), nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU())

    def forward(self, x):
        return F.interpolate(super().forward(x), size=x.shape[-2:], mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, cin, rates=(6, 12, 18), cout=256):
        super().__init__()
        self.convs = nn.ModuleList([nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU())] +
                                   [ASPPConv(cin, cout, r) for r in rates] + [ASPPPooling(cin, cout)])
        self.project = nn.Sequential(nn.Conv2d((2 + len(rates)) * cout, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(),
                                     nn.Dropout(0.5))

    def forward(self, x):
        return self.project(torch.cat([c(x) for c in self.convs], 1))


class ConvBlock(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(nn.ReflectionPad2d(1), nn.Conv2d(cin, cout, 3), nn.ELU(inplace=True))


class DecoderHead(nn.Module):
    def __init__(self, enc_ch, n_out, active_scales=(0, 1), dec=(64, 128, 128, 256, 256), last=0):
        super().__init__()
        self.enc_ch, self.dec, self.active = enc_ch, dec, tuple(active_scales)
        self.up0, self.up1, self.side = nn.ModuleDict(), nn.ModuleDict(), nn.ModuleDict()
        for i in range(4, last - 1, -1):
            cin = enc_ch[-1] if i == 4 else dec[i + 1]
            self.up0[str(i)] = ASPP(cin, cout=dec[i]) if i == 4 else ConvBlock(cin, dec[i])
            self.up1[str(i)] = ConvBlock(dec[i] + (enc_ch[i - 1] if i > 0 else 0), dec[i])
        for i in self.active:
            self.side[str(i)] = nn.Conv2d(dec[i], n_out, 1)

    def forward(self, enc, x=None, layers=(4, 3, 2, 1, 0)):
        out = {}
        if x is None:
            x = enc[-1]
        for i in range(4, -1, -1):
            if i not in layers:
                continue
            x = self.up0[str(i)](x)
            if i == 0 or x.shape[-1] < enc[i - 1].shape[-1]:
                x = F.interpolate(x, scale_factor=2, mode="nearest")
            if i > 0:
                x = torch.cat([x, enc[i - 1]], 1)
            x = self.up1[str(i)](x)
            out["upconv", i] = x
            if i in self.active:
                out[i] = self.side[str(i)](x)
        return out


class SxTAM(nn.Module):
    def __init__(self, c, s, use_alpha=True):
        super().__init__()
        self.conv_b = nn.Sequential(nn.Conv2d(c, c, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(), nn.Flatten(2))
        self.conv_c = nn.Sequential(nn.Conv2d(c, c, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(), nn.Flatten(2))
        self.conv_d = nn.Sequential(nn.Conv2d(c, c, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU())
        self.alpha = nn.Parameter(torch.zeros(1, c, 1, 1)) if use_alpha else None
        self.down = nn.MaxPool2d(s, s)
        self.d = nn.Upsample(scale_factor=1 / s)
        self.u = nn.Upsample(scale_factor=s)

    def forward(self, x, y):
        x_ = self.d(x)
        B = self.conv_b(x_).transpose(1, 2)
        C = self.conv_c(self.down(y))
        D = self.conv_d(self.down(y))
        corr = torch.softmax(B @ C / math.sqrt(B.size(2)), -1).transpose(1, 2)
        out = self.u((D.flatten(2) @ corr).view_as(x_))
        return out * self.alpha if self.alpha is not None else out


class Project(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU())


class Extractor(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU())


class XTAMBlock(nn.Module):
    def __init__(self, tasks, c, ds):
        super().__init__()
        self.tasks = tasks
        self.m = nn.ModuleDict({t: nn.ModuleDict({
            **{"extract_" + s: Extractor(c, c) for s in tasks if s != t},
            **{"SA_" + s: SABlock(c, c // 2) for s in tasks if s != t},
            **{"SxTAM_" + s: SxTAM(c, ds) for s in tasks if s != t},
            **{"proj_" + s: Project(int(1.5 * c), c) for s in tasks if s != t},
            "post": Project((len(tasks) - 1) * c, c)}) for t in tasks})

    def forward(self, code):
        out = {}
        for t in self.tasks:
            parts = []
            for s in self.tasks:
                if s == t:
                    continue
                m = self.m[t]
                f_dir = m["extract_" + s](code[s])
                parts.append(m["proj_" + s](torch.cat([m["SA_" + s](f_dir), m["SxTAM_" + s](code[t], f_dir)], 1)))
            out[t] = code[t] + self.m[t]["post"](torch.cat(parts, 1))
        return out


class DenseMTL(nn.Module):
    def __init__(self, n_cls, pretrained=True):
        super().__init__()
        self.tasks = ("seg", "clf")
        self.enc = Encoder(SHARED, pretrained)
        ch = self.enc.channels
        self.dec = nn.ModuleDict({"seg": DecoderHead(ch, 1, (0, 1)), "clf": DecoderHead(ch, 1, (), last=1)})
        self.xtam = XTAMBlock(self.tasks, 128, 4)
        self.clf_head = MLPHead(128 + ch[-1], n_cls)

    def forward(self, x):
        enc = self.enc(x)
        first = {t: self.dec[t](enc, layers=(4, 3, 2, 1)) for t in self.tasks}
        code = self.xtam({t: first[t]["upconv", 1] for t in self.tasks})
        second = self.dec["seg"](enc, x=code["seg"], layers=(0,))
        clf = self.clf_head(torch.cat([code["clf"].mean((2, 3)), enc[-1].mean((2, 3))], 1))
        return result(second[0], clf, [aux_seg(first["seg"][1], 1.0, x.shape[-1])], x.shape[-1])


class BasicConv2d(nn.Sequential):
    def __init__(self, cin, cout, k, s=1, p=0, d=1):
        super().__init__(nn.Conv2d(cin, cout, k, s, p, dilation=d, bias=False), nn.BatchNorm2d(cout))


class RFB(nn.Module):
    def __init__(self, cin, c):
        super().__init__()
        br = lambda k, d: nn.Sequential(BasicConv2d(cin, c, 1), BasicConv2d(c, c, (1, k), p=(0, k // 2)),
                                        BasicConv2d(c, c, (k, 1), p=(k // 2, 0)), BasicConv2d(c, c, 3, p=d, d=d))
        self.b0 = BasicConv2d(cin, c, 1)
        self.b1, self.b2, self.b3 = br(3, 3), br(5, 5), br(7, 7)
        self.cat = BasicConv2d(4 * c, c, 3, p=1)
        self.res = BasicConv2d(cin, c, 1)

    def forward(self, x):
        return F.relu(self.cat(torch.cat([self.b0(x), self.b1(x), self.b2(x), self.b3(x)], 1)) + self.res(x))


class Aggregation(nn.Module):
    def __init__(self, c=32):
        super().__init__()
        self.cu1, self.cu2, self.cu3, self.cu4 = (BasicConv2d(c, c, 3, p=1) for _ in range(4))
        self.cu5 = BasicConv2d(2 * c, 2 * c, 3, p=1)
        self.cc2, self.cc3 = BasicConv2d(2 * c, 2 * c, 3, p=1), BasicConv2d(3 * c, 3 * c, 3, p=1)
        self.conv4, self.conv5 = BasicConv2d(3 * c, 3 * c, 3, p=1), nn.Conv2d(3 * c, 1, 1)

    def forward(self, x1, x2, x3):
        up = lambda t, ref: F.interpolate(t, size=ref.shape[-2:], mode="bilinear", align_corners=True)
        x2_1 = self.cu1(up(x1, x2)) * x2
        x3_1 = self.cu2(up(x1, x3)) * self.cu3(up(x2, x3)) * x3
        x2_2 = self.cc2(torch.cat([x2_1, self.cu4(up(x1, x2))], 1))
        x3_2 = self.cc3(torch.cat([x3_1, self.cu5(up(x2_2, x3))], 1))
        return self.conv5(self.conv4(x3_2))


class ReverseAddition(nn.Module):
    def __init__(self, cin, c, k):
        super().__init__()
        self.c1 = BasicConv2d(cin, c, 1)
        self.c2, self.c3, self.c4 = (BasicConv2d(c, c, k, p=k // 2) for _ in range(3))
        self.c5 = BasicConv2d(c, 1, k, p=k // 2)

    def forward(self, feat, coarse):
        coarse = F.interpolate(coarse, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        x = (1 - torch.sigmoid(coarse)) * feat
        x = F.relu(self.c4(F.relu(self.c3(F.relu(self.c2(self.c1(x)))))))
        return self.c5(x) + coarse


class MTANet(nn.Module):
    def __init__(self, n_cls, pretrained=True):
        super().__init__()
        self.enc = Encoder(SHARED, pretrained, out_indices=(2, 3, 4))
        c2, c3, c4 = self.enc.channels
        self.rfb2, self.rfb3, self.rfb4 = RFB(c2, 32), RFB(c3, 32), RFB(c4, 32)
        self.agg = Aggregation(32)
        self.ra4, self.ra3, self.ra2 = ReverseAddition(c4, 256, 5), ReverseAddition(c3, 64, 3), ReverseAddition(c2, 64, 3)
        self.att = nn.Conv2d(c4, 1, 1)
        self.clf_head = MLPHead(c4, n_cls)

    def forward(self, x):
        f2, f3, f4 = self.enc(x)
        g = self.agg(self.rfb4(f4), self.rfb3(f3), self.rfb2(f2))
        m4 = self.ra4(f4, g)
        m3 = self.ra3(f3, m4)
        m2 = self.ra2(f2, m3)
        a = torch.sigmoid(self.att(f4))
        z = (f4 * a).sum((2, 3)) / a.sum((2, 3)).clamp_min(1e-6)
        return result(m2, self.clf_head(z), [aux_seg(m, 1.0, x.shape[-1]) for m in (g, m4, m3)], x.shape[-1])


class GNBlock(nn.Module):
    def __init__(self, cin, c):
        super().__init__()
        self.c1 = nn.Sequential(nn.Conv2d(cin, c, 3, padding=1, bias=False), nn.GroupNorm(8, c), nn.ReLU(inplace=True))
        self.c2 = nn.Sequential(nn.Conv2d(c, c, 3, padding=1, bias=False), nn.GroupNorm(8, c))
        self.skip = nn.Conv2d(cin, c, 1) if cin != c else nn.Identity()

    def forward(self, x):
        return F.relu(self.c2(self.c1(x)) + self.skip(x))


class OCR(nn.Module):
    def __init__(self, c, key, k=2):
        super().__init__()
        self.aux = nn.Sequential(conv_bn_relu(c, c), nn.Conv2d(c, k, 1))
        cbr = lambda a, b: conv_bn_relu(a, b, 1, p=0)
        self.f_pixel = nn.Sequential(cbr(c, key), cbr(key, key))
        self.f_object = nn.Sequential(cbr(c, key), cbr(key, key))
        self.f_down = cbr(c, key)
        self.f_up = cbr(key, c)
        self.out = nn.Sequential(cbr(2 * c, c), nn.Dropout2d(0.05))
        self.key = key

    def forward(self, x):
        B, C, H, W = x.shape
        regions = self.aux(x)
        prob = regions.flatten(2).softmax(-1)
        obj = (prob @ x.flatten(2).transpose(1, 2)).transpose(1, 2)[..., None]
        q = self.f_pixel(x).flatten(2).transpose(1, 2)
        k = self.f_object(obj).flatten(2)
        v = self.f_down(obj).flatten(2).transpose(1, 2)
        sim = torch.softmax(q @ k / math.sqrt(self.key), -1)
        ctx = self.f_up((sim @ v).transpose(1, 2).reshape(B, self.key, H, W))
        return self.out(torch.cat([ctx, x], 1)), regions


def gradient_magnitude(x, kx, ky):
    gray = x[:, :3].mean(1, keepdim=True) / 255.0
    p = kx.shape[-1] // 2
    gp = F.pad(gray, (p, p, p, p), mode="reflect")
    mag = torch.sqrt(F.conv2d(gp, kx.to(gp.dtype)) ** 2 + F.conv2d(gp, ky.to(gp.dtype)) ** 2 + 1e-12)
    return (mag - mag.mean((2, 3), keepdim=True)) / (mag.std((2, 3), keepdim=True) + 1e-6)


class MTLOCA(nn.Module):
    def __init__(self, n_cls, pretrained=True, dec=(256, 128, 64, 64, 64), sigma=1.0):
        super().__init__()
        r = int(3 * sigma)
        t = torch.arange(-r, r + 1, dtype=torch.float32)
        g = torch.exp(-t ** 2 / (2 * sigma ** 2)); g = g / g.sum()
        dg = -t / sigma ** 2 * g
        self.register_buffer("kx", (g[:, None] * dg[None, :])[None, None])
        self.register_buffer("ky", (dg[:, None] * g[None, :])[None, None])
        self.enc = Encoder(SHARED, pretrained, in_chans=4)
        ch = self.enc.channels
        skips = [ch[3], ch[2], ch[1], ch[0], 0]
        ins = [ch[4]] + list(dec[:-1])
        self.dec = nn.ModuleList([GNBlock(ins[j] + skips[j], dec[j]) for j in range(5)])
        self.ocr = OCR(dec[-1], key=dec[-1] // 2)
        self.head = nn.Conv2d(dec[-1], 1, 1)
        self.clf_head = MLPHead(ch[4], n_cls)

    def forward(self, x):
        x = torch.cat([x, gradient_magnitude(x, self.kx, self.ky)], 1)
        f = self.enc(x)
        y = f[4]
        for j, sk in enumerate([f[3], f[2], f[1], f[0], None]):
            y = F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=False)
            y = self.dec[j](y if sk is None else torch.cat([y, sk], 1))
        z, regions = self.ocr(y)
        coarse = regions[:, 1:2] - regions[:, 0:1]
        return result(self.head(z), self.clf_head(f[4].mean((2, 3))), [aux_seg(coarse, 0.4, x.shape[-1])], x.shape[-1])


class AIWNet(nn.Module):
    def __init__(self, n_cls, pretrained=True):
        super().__init__()
        self.net = AdaptiveInteractionModel(num_classes=n_cls, pretrained=pretrained)

    def forward(self, x):
        n = self.net
        n._feats = {}
        n.enc.forward_features((x - n.mean) / n.std)
        e = {k: n.res[k](v) for k, v in n._feats.items()}
        d, vecs = e["br"], []
        for lvl in range(4):
            up = n.ups[lvl](d)
            d = n.drop(n.dblocks[lvl](torch.cat([up, n.gates[lvl](e[n.skips[lvl]], up)], 1)))
            f = F.relu(n.clf_fc[lvl](e[n.clf_src[lvl]].mean((2, 3))))
            if n.tims is not None:
                D_enh, f_enh = n.tims[lvl](d, f)
                d, f, _ = n.aiws[lvl](d, D_enh, f, f_enh)
            vecs.append(f)
        with torch.autocast(x.device.type, enabled=False):
            seg = n.seg_head(d.float())
            z = torch.cat([e[k].float().mean((2, 3)) for k in ("s2", "s3", "s4", "br")] + [v.float() for v in vecs], 1)
            return result(seg, n.clf_head(z), size=x.shape[-1])

    def calibrate(self, X, batch_size=24):
        return calibrate_aiw(self.net, X, batch_size)


def build(method, n_cls, pretrained=True):
    return {"mtan": MTAN, "mti_net": MTINet, "densemtl": DenseMTL, "mtanet": MTANet, "mtl_oca": MTLOCA,
            "aiw": AIWNet}[method](n_cls, pretrained=pretrained)
