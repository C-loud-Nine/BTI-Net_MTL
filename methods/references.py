"""The reference networks: U-Net, Attention U-Net, UNet++, TransUNet, Swin-Unet and MISSFormer, and the two
single-task networks on the shared EfficientNet-B4 encoder."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .multitask import SHARED, Encoder, MLPHead, result, up_to

IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
INCEPTION = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

MODEL_INFO = {
    "unet": {"pretrained": False, "norm": None},
    "att_unet": {"pretrained": False, "norm": None},
    "unet_pp": {"pretrained": False, "norm": None},
    "transunet": {"pretrained": True, "norm": INCEPTION},
    "swin_unet": {"pretrained": True, "norm": IMAGENET},
    "missformer": {"pretrained": False, "norm": None},
}


def conv_bn_relu(cin, cout, k=3, s=1, p=None, d=1):
    p = (k // 2) * d if p is None else p
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, p, dilation=d, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class DoubleConv(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(conv_bn_relu(cin, cout), conv_bn_relu(cout, cout))


class ClfHead(nn.Module):
    def __init__(self, cin, n):
        super().__init__()
        self.net = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(cin, 256),
                                 nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(256, n))

    def forward(self, x):
        return self.net(x)


def out(seg_logit, clf_logit=None, aux=None, size=224):
    seg_logit = seg_logit.float()
    if seg_logit.shape[-1] != size:
        seg_logit = F.interpolate(seg_logit, size=(size, size), mode="bilinear", align_corners=False)
    d = {"seg": torch.sigmoid(seg_logit)}
    if clf_logit is not None:
        d["clf"] = torch.softmax(clf_logit.float(), dim=1)
    if aux:
        d["aux"] = aux
    return d


class UNet(nn.Module):
    def __init__(self, n_cls, ch=(64, 128, 256, 512, 1024), attention=False):
        super().__init__()
        self.enc = nn.ModuleList([DoubleConv(3, ch[0])] +
                                 [DoubleConv(ch[i], ch[i + 1]) for i in range(4)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, 2) for i in reversed(range(4))])
        self.dec = nn.ModuleList([DoubleConv(2 * ch[i], ch[i]) for i in reversed(range(4))])
        self.att = nn.ModuleList([AttGate(ch[i], ch[i], ch[i] // 2) for i in reversed(range(4))]) \
            if attention else None
        self.head = nn.Conv2d(ch[0], 1, 1)
        self.clf = ClfHead(ch[4], n_cls)

    def forward(self, x):
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        bottleneck = x
        for j, (up, dec) in enumerate(zip(self.up, self.dec)):
            x = up(x)
            s = skips[3 - j]
            if self.att is not None:
                s = self.att[j](s, x)
            x = dec(torch.cat([s, x], 1))
        return out(self.head(x), self.clf(bottleneck))


class AttGate(nn.Module):
    def __init__(self, c_skip, c_gate, c_int):
        super().__init__()
        self.wx = nn.Sequential(nn.Conv2d(c_skip, c_int, 1, bias=False), nn.BatchNorm2d(c_int))
        self.wg = nn.Sequential(nn.Conv2d(c_gate, c_int, 1, bias=False), nn.BatchNorm2d(c_int))
        self.psi = nn.Sequential(nn.Conv2d(c_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())

    def forward(self, x, g):
        return x * self.psi(F.relu(self.wx(x) + self.wg(g)))


class UNetPP(nn.Module):
    def __init__(self, n_cls, nb=(32, 64, 128, 256, 512)):
        super().__init__()
        self.c = nn.ModuleDict()
        for i in range(5):
            self.c[f"{i}_0"] = DoubleConv(3 if i == 0 else nb[i - 1], nb[i])
        for j in range(1, 5):
            for i in range(5 - j):
                self.c[f"{i}_{j}"] = DoubleConv(nb[i] * j + nb[i + 1], nb[i])
        self.head = nn.Conv2d(nb[0], 1, 1)
        self.clf = ClfHead(nb[4], n_cls)

    def forward(self, x):
        X = {}
        for i in range(5):
            X[(i, 0)] = self.c[f"{i}_0"](x if i == 0 else F.max_pool2d(X[(i - 1, 0)], 2))
        for j in range(1, 5):
            for i in range(5 - j):
                up = F.interpolate(X[(i + 1, j - 1)], scale_factor=2, mode="bilinear", align_corners=False)
                X[(i, j)] = self.c[f"{i}_{j}"](torch.cat([X[(i, k)] for k in range(j)] + [up], 1))
        return out(self.head(X[(0, 4)]), self.clf(X[(4, 0)]))


class TransUNet(nn.Module):
    def __init__(self, n_cls, pretrained=True):
        super().__init__()
        import timm
        self.vit = timm.create_model("vit_base_r50_s16_224.orig_in21k", pretrained=pretrained,
                                     num_classes=0)
        bb = self.vit.patch_embed.backbone
        self.stem, self.stages = bb.stem, bb.stages
        self.conv_more = conv_bn_relu(768, 512)
        dec_in = [512, 256, 128, 64]
        skip_ch = [512, 256, 64, 0]
        dec_out = [256, 128, 64, 16]
        self.blocks = nn.ModuleList([nn.Sequential(conv_bn_relu(dec_in[i] + skip_ch[i], dec_out[i]),
                                                   conv_bn_relu(dec_out[i], dec_out[i]))
                                     for i in range(4)])
        self.head = nn.Conv2d(16, 1, 3, padding=1)

    def forward(self, x):

        s = self.stem.conv(x)
        s = self.stem.norm(s)
        skip2 = s
        h = self.stem.pool(s)
        feats = []
        for st in self.stages:
            h = st(h)
            feats.append(h)
        skip4, skip8 = feats[0], feats[1]
        tok = self.vit.patch_embed.proj(h).flatten(2).transpose(1, 2)
        tok = self.vit._pos_embed(tok)
        tok = self.vit.norm_pre(tok)
        tok = self.vit.blocks(tok)
        tok = self.vit.norm(tok)
        n_pref = getattr(self.vit, "num_prefix_tokens", 1)
        tok = tok[:, n_pref:]
        B, N, C = tok.shape
        g = int(math.sqrt(N))
        y = self.conv_more(tok.transpose(1, 2).reshape(B, C, g, g))
        for blk, sk in zip(self.blocks, [skip8, skip4, skip2, None]):
            y = F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=False)
            if sk is not None:
                y = torch.cat([y, sk], 1)
            y = blk(y)
        return out(self.head(y))


def window_partition(x, w):
    B, H, W, C = x.shape
    x = x.view(B, H // w, w, W // w, w, C).permute(0, 1, 3, 2, 4, 5)
    return x.reshape(-1, w * w, C)


def window_reverse(win, w, H, W):
    B = win.shape[0] // ((H // w) * (W // w))
    x = win.view(B, H // w, W // w, w, w, -1).permute(0, 1, 3, 2, 4, 5)
    return x.reshape(B, H, W, -1)


class WindowAttention(nn.Module):
    def __init__(self, dim, w, heads):
        super().__init__()
        self.w, self.heads, self.scale = w, heads, (dim // heads) ** -0.5
        self.rpb = nn.Parameter(torch.zeros((2 * w - 1) ** 2, heads))
        nn.init.trunc_normal_(self.rpb, std=0.02)
        coords = torch.stack(torch.meshgrid(torch.arange(w), torch.arange(w), indexing="ij")).flatten(1)
        rel = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0) + (w - 1)
        self.register_buffer("rpi", rel[..., 0] * (2 * w - 1) + rel[..., 1], persistent=False)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        Bw, N, C = x.shape
        qkv = self.qkv(x).reshape(Bw, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        a = q @ k.transpose(-2, -1) + self.rpb[self.rpi.view(-1)].view(N, N, -1).permute(2, 0, 1)[None]
        if mask is not None:
            nW = mask.shape[0]
            a = a.view(Bw // nW, nW, self.heads, N, N) + mask[None, :, None]
            a = a.view(Bw, self.heads, N, N)
        x = (a.softmax(-1) @ v).transpose(1, 2).reshape(Bw, N, C)
        return self.proj(x)


class SwinBlock(nn.Module):
    def __init__(self, dim, res, heads, w=7, shift=0):
        super().__init__()
        self.res, self.w, self.shift = res, min(w, res), (0 if res <= w else shift)
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, self.w, heads)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        if self.shift:
            img = torch.zeros(1, res, res, 1)
            cnt = 0
            for hs in (slice(0, -self.w), slice(-self.w, -self.shift), slice(-self.shift, None)):
                for ws in (slice(0, -self.w), slice(-self.w, -self.shift), slice(-self.shift, None)):
                    img[:, hs, ws, :] = cnt
                    cnt += 1
            mw = window_partition(img, self.w).squeeze(-1)
            m = mw[:, None, :] - mw[:, :, None]
            self.register_buffer("mask", m.masked_fill(m != 0, -100.0).masked_fill(m == 0, 0.0),
                                 persistent=False)
        else:
            self.mask = None

    def forward(self, x):
        B, H, W, C = x.shape
        h = self.n1(x)
        if self.shift:
            h = torch.roll(h, (-self.shift, -self.shift), (1, 2))
        h = window_reverse(self.attn(window_partition(h, self.w), self.mask), self.w, H, W)
        if self.shift:
            h = torch.roll(h, (self.shift, self.shift), (1, 2))
        x = x + h
        return x + self.mlp(self.n2(x))


class PatchExpand(nn.Module):
    def __init__(self, dim, scale=2, out_dim=None):
        super().__init__()
        self.scale = scale
        out_dim = out_dim or dim // 2
        self.lin = nn.Linear(dim, out_dim * scale * scale, bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        B, H, W, _ = x.shape
        x = self.lin(x)
        s = self.scale
        x = x.view(B, H, W, s, s, -1).permute(0, 1, 3, 2, 4, 5).reshape(B, H * s, W * s, -1)
        return self.norm(x)


class SwinUNet(nn.Module):
    def __init__(self, n_cls, pretrained=True):
        super().__init__()
        import timm
        self.enc = timm.create_model("swin_tiny_patch4_window7_224", pretrained=pretrained,
                                     features_only=True, out_indices=(0, 1, 2, 3))
        dims, res, heads = [96, 192, 384, 768], [56, 28, 14, 7], [3, 6, 12, 24]
        self.up = nn.ModuleList([PatchExpand(dims[i]) for i in (3, 2, 1)])
        self.cat = nn.ModuleList([nn.Linear(2 * dims[i], dims[i]) for i in (2, 1, 0)])
        self.dec = nn.ModuleList([nn.Sequential(SwinBlock(dims[i], res[i], heads[i]),
                                                SwinBlock(dims[i], res[i], heads[i], shift=3))
                                  for i in (2, 1, 0)])
        self.norm = nn.LayerNorm(dims[0])
        self.final = PatchExpand(dims[0], scale=4, out_dim=dims[0])
        self.head = nn.Conv2d(dims[0], 1, 1)

    def forward(self, x):
        f = self.enc(x)
        f = [t if t.shape[-1] in (96, 192, 384, 768) and t.shape[1] == t.shape[2] else t.permute(0, 2, 3, 1)
             for t in f]
        y = f[3]
        for up, cat, dec, sk in zip(self.up, self.cat, self.dec, (f[2], f[1], f[0])):
            y = up(y)
            y = dec(cat(torch.cat([y, sk], -1)))
        y = self.final(self.norm(y))
        return out(self.head(y.permute(0, 3, 1, 2)))


class MFEffAttention(nn.Module):
    def __init__(self, dim, heads, r):
        super().__init__()
        self.heads, self.scale, self.r = heads, (dim // heads) ** -0.5, r
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)
        if r > 1:
            self.sr = nn.Conv2d(dim, dim, r, r)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.heads, C // self.heads).transpose(1, 2)
        if self.r > 1:
            x = self.norm(self.sr(x.transpose(1, 2).reshape(B, C, H, W)).reshape(B, C, -1).transpose(1, 2))
        kv = self.kv(x).reshape(B, -1, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        a = (q @ kv[0].transpose(-2, -1)) * self.scale
        return self.proj((a.softmax(-1) @ kv[1]).transpose(1, 2).reshape(B, N, C))

class MFMixFFNSkip(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = nn.Conv2d(c2, c2, 3, 1, 1, groups=c2)
        self.fc2 = nn.Linear(c2, c1)
        self.norm1 = nn.LayerNorm(c2)

    def forward(self, x, H, W):
        B, N, _ = x.shape
        h = self.fc1(x)
        d = self.dwconv(h.transpose(1, 2).reshape(B, -1, H, W)).flatten(2).transpose(1, 2)
        return self.fc2(F.gelu(self.norm1(d + h)))

class MFBlock(nn.Module):
    def __init__(self, dim, heads, r):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = MFEffAttention(dim, heads, r)
        self.mlp = MFMixFFNSkip(dim, dim * 4)

    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        return x + self.mlp(self.norm2(x), H, W)

class MFPatchEmbed(nn.Module):
    def __init__(self, cin, dim, k, s, p):
        super().__init__()
        self.proj = nn.Conv2d(cin, dim, k, s, p)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        return self.norm(x.flatten(2).transpose(1, 2)), H, W

class MFBridgeLayer(nn.Module):
    def __init__(self, c=64, mult=(1, 2, 5, 8), sizes=(56, 28, 14, 7), ratios=(8, 4, 2)):
        super().__init__()
        self.c, self.mult, self.sizes = c, mult, sizes
        self.lens = [s * s * m for s, m in zip(sizes, mult)]
        self.norm1 = nn.LayerNorm(c)
        self.q = nn.Linear(c, c)
        self.kv = nn.Linear(c, 2 * c)
        self.proj = nn.Linear(c, c)
        self.sr = nn.ModuleList([nn.Conv2d(c * m, c * m, r, r) for m, r in zip(mult[:3], ratios)])
        self.sr_norm = nn.LayerNorm(c)
        self.norm2 = nn.LayerNorm(c)
        self.mixffn = nn.ModuleList([MFMixFFNSkip(c * m, c * m * 4) for m in mult])

    def attn(self, x):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, 1, C).transpose(1, 2)
        parts = torch.split(x, self.lens, 1)
        red = []
        for i in range(3):
            m, s = self.mult[i], self.sizes[i]
            t = parts[i].reshape(B, s, s, C * m).permute(0, 3, 1, 2)
            red.append(self.sr[i](t).reshape(B, C, -1).permute(0, 2, 1))
        red.append(parts[3])
        kvx = self.sr_norm(torch.cat(red, 1))
        kv = self.kv(kvx).reshape(B, -1, 2, 1, C).permute(2, 0, 3, 1, 4)
        a = (q @ kv[0].transpose(-2, -1)) * C ** -0.5
        return self.proj((a.softmax(-1) @ kv[1]).transpose(1, 2).reshape(B, N, C))

    def forward(self, x):
        B, _, C = x.shape
        tx1 = x + self.attn(self.norm1(x))
        tx = self.norm2(tx1)
        outs = []
        for p, m, s, ffn in zip(torch.split(tx, self.lens, 1), self.mult, self.sizes, self.mixffn):
            outs.append(ffn(p.reshape(B, -1, C * m), s, s).reshape(B, -1, C))
        return tx1 + torch.cat(outs, 1)

class MFPatchExpand(nn.Module):
    def __init__(self, dim, scale=2):
        super().__init__()
        self.s = scale
        self.expand = nn.Linear(dim, 2 * dim if scale == 2 else 16 * dim, bias=False)
        self.norm = nn.LayerNorm(dim // 2 if scale == 2 else dim)

    def forward(self, x, H, W):
        B = x.shape[0]
        x = self.expand(x)
        s = self.s
        x = x.view(B, H, W, s, s, -1).permute(0, 1, 3, 2, 4, 5).reshape(B, H * s * W * s, -1)
        return self.norm(x)

class MFDecoderLayer(nn.Module):
    def __init__(self, in_out, heads, r, n_out=1, is_last=False, is_first=False):
        super().__init__()
        dims, out_dim = in_out
        self.is_first, self.is_last = is_first, is_last
        if not is_first:
            self.concat_linear = nn.Linear(dims * (4 if is_last else 2), out_dim)
            self.layer_former_1 = MFBlock(out_dim, heads, r)
            self.layer_former_2 = MFBlock(out_dim, heads, r)
        self.layer_up = MFPatchExpand(out_dim, 4 if is_last else 2)
        self.last_layer = nn.Conv2d(out_dim, n_out, 1) if is_last else None
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x1, H, W, x2=None):
        if x2 is None:
            return self.layer_up(x1, H, W)
        y = self.concat_linear(torch.cat([x1, x2], -1))
        y = self.layer_former_2(self.layer_former_1(y, H, W), H, W)
        y = self.layer_up(y, H, W)
        if self.last_layer is not None:
            B = y.shape[0]
            y = self.last_layer(y.view(B, 4 * H, 4 * W, -1).permute(0, 3, 1, 2))
        return y

class MISSFormer(nn.Module):
    def __init__(self, n_cls, img_size=224):
        super().__init__()
        dims, heads, sr = (64, 128, 320, 512), (1, 2, 5, 8), (8, 4, 2, 1)
        self.embeds = nn.ModuleList([MFPatchEmbed(3, dims[0], 7, 4, 3)] +
                                    [MFPatchEmbed(dims[i - 1], dims[i], 3, 2, 1) for i in range(1, 4)])
        self.enc = nn.ModuleList([nn.ModuleList([MFBlock(dims[i], heads[i], sr[i]) for _ in range(2)])
                                  for i in range(4)])
        self.enorm = nn.ModuleList([nn.LayerNorm(d) for d in dims])
        b = img_size // 4
        sizes = (b, b // 2, b // 4, b // 8)
        self.bridge = nn.ModuleList([MFBridgeLayer(64, (1, 2, 5, 8), sizes)
                                     for _ in range(4)])
        io = [[32, 64], [144, 128], [288, 320], [512, 512]]
        self.decoder_3 = MFDecoderLayer(io[3], heads[3], sr[3], is_first=True)
        self.decoder_2 = MFDecoderLayer(io[2], heads[2], sr[2])
        self.decoder_1 = MFDecoderLayer(io[1], heads[1], sr[1])
        self.decoder_0 = MFDecoderLayer(io[0], heads[0], sr[0], n_out=1, is_last=True)

    def forward(self, x):
        B = x.shape[0]
        feats, shapes, h = [], [], x
        for emb, blocks, nrm in zip(self.embeds, self.enc, self.enorm):
            t, H, W = emb(h)
            for blk in blocks:
                t = blk(t, H, W)
            t = nrm(t)
            feats.append(t); shapes.append((H, W))
            h = t.transpose(1, 2).reshape(B, -1, H, W)

        z = torch.cat([f.reshape(B, -1, 64) for f in feats], 1)
        for layer in self.bridge:
            z = layer(z)
        parts = torch.split(z, self.bridge[0].lens, 1)
        sk = [p.reshape(B, -1, d) for p, d in zip(parts, (64, 128, 320, 512))]
        (H3, W3), (H2, W2), (H1, W1), (H0, W0) = shapes[3], shapes[2], shapes[1], shapes[0]
        y = self.decoder_3(sk[3], H3, W3)
        y = self.decoder_2(y, H2, W2, sk[2])
        y = self.decoder_1(y, H1, W1, sk[1])
        y = self.decoder_0(y, H0, W0, sk[0])
        return out(y)


def build_model(name, num_classes, pretrained=None):
    pt = MODEL_INFO[name]["pretrained"] if pretrained is None else pretrained
    builders = {
        "unet": lambda: UNet(num_classes),
        "att_unet": lambda: UNet(num_classes, attention=True),
        "unet_pp": lambda: UNetPP(num_classes),
        "transunet": lambda: TransUNet(num_classes, pretrained=pt),
        "swin_unet": lambda: SwinUNet(num_classes, pretrained=pt),
        "missformer": lambda: MISSFormer(num_classes),
    }
    return builders[name]()


LABEL = {"unet": "U-Net", "att_unet": "Attention U-Net", "unet_pp": "UNet++",
         "transunet": "TransUNet", "swin_unet": "Swin-Unet", "missformer": "MISSFormer",
         "stl_seg": "Single-task segmentation (EfficientNet-B4 U-Net)", "stl_clf": "Single-task classification (EfficientNet-B4)"}
TASKS = {"unet": ("seg", "clf"), "att_unet": ("seg", "clf"), "unet_pp": ("seg", "clf"), "transunet": ("seg",),
         "swin_unet": ("seg",), "missformer": ("seg",), "stl_seg": ("seg",), "stl_clf": ("clf",)}


class Standard(nn.Module):
    def __init__(self, name, n_cls, pretrained=True):
        super().__init__()
        self.net = build_model(name, n_cls, pretrained=None if pretrained else False)
        self.n_cls = n_cls
        norm = MODEL_INFO[name]["norm"]
        self.normalise = norm is not None
        mean, std = norm if norm is not None else ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        x = x / 255.0
        if self.normalise:
            x = (x - self.mean) / self.std
        o = self.net(x)
        if "clf" not in o:
            o["clf"] = torch.full((x.shape[0], self.n_cls), 1.0 / self.n_cls, device=x.device, dtype=torch.float32)
        o["aux"] = [({k: v for k, v in a.items() if k in ("seg", "clf")}, w) for a, w in o.get("aux", [])]
        return o


class STLSeg(nn.Module):
    def __init__(self, n_cls, pretrained=True, dec=(256, 128, 64, 32)):
        super().__init__()
        self.enc = Encoder(SHARED, pretrained)
        c = self.enc.channels
        self.n_cls = n_cls
        ins = [c[4]] + list(dec[:-1])
        skips = [c[3], c[2], c[1], c[0]]
        self.dec_a = nn.ModuleList([conv_bn_relu(ins[j] + skips[j], dec[j]) for j in range(4)])
        self.dec_b = nn.ModuleList([conv_bn_relu(dec[j], dec[j]) for j in range(4)])
        self.seg_head = nn.Sequential(nn.Conv2d(dec[3], dec[3], 3, padding=1), nn.Conv2d(dec[3], 1, 1))

    def forward(self, x):
        f = self.enc(x)
        d = f[4]
        for j, s in enumerate((f[3], f[2], f[1], f[0])):
            d = self.dec_b[j](self.dec_a[j](torch.cat([up_to(d, s), s], 1)))
        o = result(self.seg_head(d), torch.zeros(x.shape[0], self.n_cls, device=x.device), (), x.shape[-1])
        return o


class STLClf(nn.Module):
    def __init__(self, n_cls, pretrained=True):
        super().__init__()
        import timm
        self.enc = timm.create_model(SHARED, pretrained=pretrained, drop_path_rate=0.2, num_classes=0)
        cfg = self.enc.pretrained_cfg
        self.register_buffer("mean", torch.tensor(cfg["mean"]).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor(cfg["std"]).view(1, 3, 1, 1) * 255)
        self.clf_head = MLPHead(self.enc.num_features, n_cls)

    def forward(self, x):
        f = self.enc((x - self.mean) / self.std)
        return {"seg": torch.zeros(x.shape[0], 1, x.shape[-2], x.shape[-1], device=x.device, dtype=torch.float32),
                "clf": torch.softmax(self.clf_head(f).float(), 1), "aux": []}


def build(method, n_cls, pretrained=True):
    if method == "stl_seg":
        return STLSeg(n_cls, pretrained)
    if method == "stl_clf":
        return STLClf(n_cls, pretrained)
    return Standard(method, n_cls, pretrained)
