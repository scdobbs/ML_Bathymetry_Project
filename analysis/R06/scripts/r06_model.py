"""R06 §1.3 — the R02 HetUNetClean encoder / decoder and FiLM blocks (het_clean.py, unchanged in depth and width:
base_ch 32, ch_mult (1, 2, 4, 4), reflect padding, per-pixel LayerNorm) with the per-pixel (mean, log-variance) head
replaced by a block head: the decoder output is cropped to the central 128 px (unless the input already is the tile, V3),
average-pooled to the 4 x 4 block grid and regressed to log10 A_fine per block. FiLM takes the 4-vector of §1.2.
V2 adds a separate, smaller encoder branch on the ship-product channels (4 values + 4 availability masks) whose pooled
output is concatenated with the pooled decoder features before the head ("fused before the head")."""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F

class PixelLN(nn.Module):
    def __init__(self, ch, eps=1e-5):
        super().__init__(); self.ln = nn.LayerNorm(ch, eps=eps)
    def forward(self, x): return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

class FiLM(nn.Module):
    def __init__(self, ch, scale_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(scale_dim, ch), nn.SiLU(), nn.Linear(ch, 2 * ch))
        nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)
    def forward(self, x, s):
        s = s.to(self.mlp[0].weight.dtype); g, b = self.mlp(s).chunk(2, dim=-1)
        return x * (1 + g[:, :, None, None]) + b[:, :, None, None]

def conv(i, o, k=3, stride=1): return nn.Conv2d(i, o, k, stride=stride, padding=k // 2, padding_mode="reflect")

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, scale_dim):
        super().__init__()
        self.norm1 = PixelLN(in_ch); self.conv1 = conv(in_ch, out_ch); self.norm2 = PixelLN(out_ch); self.conv2 = conv(out_ch, out_ch)
        self.film = FiLM(out_ch, scale_dim); self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x, s):
        h = self.conv1(F.silu(self.norm1(x))); h = self.film(h, s); h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)

class UNetBody(nn.Module):
    """HetUNetClean minus its two 1 x 1 heads: returns the chs[0]-channel decoder output at input resolution."""
    def __init__(self, base_ch=32, ch_mult=(1, 2, 4, 4), in_ch=2, scale_dim=4):
        super().__init__(); chs = [base_ch * m for m in ch_mult]; self.out_ch = chs[0]
        self.in_conv = conv(in_ch, chs[0]); self.enc = nn.ModuleList(); self.downs = nn.ModuleList()
        for i in range(len(chs)):
            self.enc.append(Block(chs[i], chs[i], scale_dim))
            self.downs.append(conv(chs[i], chs[min(i + 1, len(chs) - 1)], stride=2) if i < len(chs) - 1 else None)
        self.ups = nn.ModuleList(); self.dec = nn.ModuleList()
        for i in reversed(range(len(chs) - 1)):
            self.ups.append(conv(chs[i + 1], chs[i])); self.dec.append(Block(2 * chs[i], chs[i], scale_dim))
        self.out_norm = PixelLN(chs[0])
    def forward(self, inp, s):
        h = self.in_conv(inp); skips = []
        for i, blk in enumerate(self.enc):
            h = blk(h, s)
            if self.downs[i] is not None: skips.append(h); h = self.downs[i](h)
        for up, blk in zip(self.ups, self.dec):
            h = F.interpolate(h, scale_factor=2, mode="nearest"); h = up(h); h = blk(torch.cat([h, skips.pop()], dim=1), s)
        return F.silu(self.out_norm(h))

class ProductBranch(nn.Module):
    """V2: separate encoder on [4 normalized products, 4 availability masks]; 3 levels (16, 32, 64 ch), output at 1/4 res."""
    def __init__(self, in_ch=8, chs=(16, 32, 64), scale_dim=4):
        super().__init__(); self.out_ch = chs[-1]
        self.in_conv = conv(in_ch, chs[0]); self.blocks = nn.ModuleList(); self.downs = nn.ModuleList()
        for i, c in enumerate(chs):
            self.blocks.append(Block(c, c, scale_dim)); self.downs.append(conv(c, chs[i + 1], stride=2) if i < len(chs) - 1 else None)
        self.out_norm = PixelLN(chs[-1])
    def forward(self, prod, s):
        h = self.in_conv(prod)
        for blk, dn in zip(self.blocks, self.downs):
            h = blk(h, s)
            if dn is not None: h = dn(h)
        return F.silu(self.out_norm(h))

class BlockHeadNet(nn.Module):
    def __init__(self, base_ch=32, ch_mult=(1, 2, 4, 4), in_ch=2, scale_dim=4, products=False, crop=False, tile=128, margin=64, nb=4):
        super().__init__(); self.products, self.crop, self.tile, self.margin, self.nb = products, crop, tile, margin, nb
        self.body = UNetBody(base_ch, ch_mult, in_ch, scale_dim); c = self.body.out_ch
        if products: self.branch = ProductBranch(scale_dim=scale_dim); c += self.branch.out_ch
        self.head = nn.Sequential(nn.Conv2d(c, c, 1), nn.SiLU(), nn.Conv2d(c, 1, 1))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)       # DiT / R02-style zero-init output
    def _central(self, h, full):
        if self.crop: return h                                   # V3: the input already is the 128-px tile
        f = h.shape[-1] / full; m, t = int(round(self.margin * f)), int(round(self.tile * f))
        return h[:, :, m:m + t, m:m + t]
    def forward(self, inp, cond, prod=None):
        full = inp.shape[-1]
        h = self._central(self.body(inp, cond), full); f = F.adaptive_avg_pool2d(h, self.nb)
        if self.products:
            hp = self._central(self.branch(prod, cond), full); f = torch.cat([f, F.adaptive_avg_pool2d(hp, self.nb)], 1)
        return self.head(f)[:, 0]                                 # (B, nb, nb) predicted log10 A_fine per block

def n_params(m): return sum(p.numel() for p in m.parameters())

if __name__ == "__main__":
    for prod, crop in ((False, False), (True, False), (False, True)):
        net = BlockHeadNet(products=prod, crop=crop); n = 128 if crop else 256
        out = net(torch.zeros(2, 2, n, n), torch.zeros(2, 4), torch.zeros(2, 8, n, n) if prod else None)
        print(f"products={prod} crop={crop}: params {n_params(net):,} (body {n_params(net.body):,}; branch {n_params(net.branch) if prod else 0:,}; head {n_params(net.head):,}) out {tuple(out.shape)}")
