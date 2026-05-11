"""
Hybrid CNN/transformer U-Net for conditional bathymetric super-resolution.

Stage 3 architecture (2026-era):
    - 6-level encoder/decoder with skip connections.
    - Levels 0-3: convolutional ResBlocks with AdaLN-Zero time modulation.
    - Levels 4-5: DiT-style transformer blocks (self-attn + cross-attn + MLP),
      attending to multi-scale LR features as K/V.
    - F.scaled_dot_product_attention everywhere — the Flash-Attention backend
      dispatches automatically on H100 when shapes are favorable.
    - DiT identity initialization: AdaLN-Zero output gates and the final
      output Conv2d are zero-initialized, so the model returns zeros at init
      and gradient flow starts from a stable identity-residual configuration.

Forward signature:
    forward(x, t, lr_raw) -> Tensor
        x:      (B, 1, H, W)     noisy / interpolated HR (H = patch_size)
        t:      (B,)             continuous noise level in [0, 1]
        lr_raw: (B, 1, h, w)     raw LR conditioning (h = lr_patch_size = H/sr_ratio)
    Returns:
        (B, 1, H, W) predicted velocity (flow matching).

This module is wired up for the new architecture only. The diffusion process,
loss computation, training loop, and inference sampler all break because the
forward signature changed — Stage 4 repairs them.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Time / noise-level embedding
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal embedding of a continuous noise level t ∈ [0, 1].

    Scales t by 1000 before the sinusoidal projection so the half-dim
    frequency range covers wavelengths from O(1) to O(1000) and the
    embedding has useful resolution across the unit interval.
    """

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0, "embedding dim must be even"
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) in [0, 1]. dtype is preserved so this works in bf16/fp16.
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) *
            torch.arange(half, device=t.device, dtype=t.dtype) / (half - 1)
        )
        args = (t * 1000.0)[:, None] * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class TimeMLP(nn.Module):
    """SinusoidalTimeEmbedding → 2-layer MLP → time_dim."""

    def __init__(self, sin_dim: int, time_dim: int):
        super().__init__()
        self.sin = SinusoidalTimeEmbedding(sin_dim)
        self.mlp = nn.Sequential(
            nn.Linear(sin_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.sin(t))


# ---------------------------------------------------------------------------
# AdaLN-Zero (DiT modulation)
# ---------------------------------------------------------------------------

class AdaLNZero(nn.Module):
    """Produce (shift, scale, gate) modulation triples from a time embedding.

    The projection is zero-initialized: shift = scale = gate = 0 at init,
    so the modulated sub-block contributes zero and the residual connection
    makes the whole block an identity. This is the DiT identity-init trick.
    """

    def __init__(self, time_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(time_dim, 3 * hidden_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, c: torch.Tensor):
        # c: (B, time_dim)
        shift, scale, gate = self.proj(F.silu(c)).chunk(3, dim=-1)
        return shift, scale, gate


def _modulate_conv(x, shift, scale):
    """AdaLN modulation for (B, C, H, W) tensors."""
    return x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]


def _modulate_seq(x, shift, scale):
    """AdaLN modulation for (B, N, C) tensors."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ---------------------------------------------------------------------------
# Convolutional ResBlock with AdaLN-Zero
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Pre-norm conv ResBlock with AdaLN-Zero time modulation.

        h = Conv2d(SiLU(GroupNorm(x)))             # in_ch → out_ch
        h = h * (1 + scale) + shift                 # AdaLN
        h = Conv2d(SiLU(GroupNorm(h)))              # zero-initialized
        h = h * gate                                # AdaLN gate
        return h + skip(x)
    """

    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.adaln = AdaLNZero(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        # Identity at init: zero the second conv.
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, c):
        shift, scale, gate = self.adaln(c)
        h = self.conv1(F.silu(self.norm1(x)))
        h = _modulate_conv(h, shift, scale)
        h = self.conv2(F.silu(self.norm2(h)))
        h = h * gate[:, :, None, None]
        return h + self.skip(x)


# ---------------------------------------------------------------------------
# Multi-head attention (self- and cross-)
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    """Multi-head attention via F.scaled_dot_product_attention.

    Self-attention: kv defaults to x.
    Cross-attention: pass a separate kv tensor (typically the LR features).
    """

    def __init__(self, dim: int, n_heads: int, kv_dim: int = None):
        super().__init__()
        assert dim % n_heads == 0, f"dim ({dim}) must be divisible by n_heads ({n_heads})"
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        kv_dim = kv_dim if kv_dim is not None else dim
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(kv_dim, dim, bias=True)
        self.v_proj = nn.Linear(kv_dim, dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x, kv=None):
        kv = x if kv is None else kv
        q = rearrange(self.q_proj(x),  'b n (h d) -> b h n d', h=self.n_heads)
        k = rearrange(self.k_proj(kv), 'b m (h d) -> b h m d', h=self.n_heads)
        v = rearrange(self.v_proj(kv), 'b m (h d) -> b h m d', h=self.n_heads)
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Transformer block (DiT pattern: self-attn + cross-attn + MLP)
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """DiT-style block with three AdaLN-Zero gated sublayers."""

    def __init__(self, dim: int, n_heads: int, time_dim: int,
                 kv_dim: int = None, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = Attention(dim, n_heads)
        self.adaln1 = AdaLNZero(time_dim, dim)

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = Attention(dim, n_heads, kv_dim=kv_dim or dim)
        self.adaln2 = AdaLNZero(time_dim, dim)

        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(approximate='tanh'),
            nn.Linear(mlp_hidden, dim),
        )
        self.adaln3 = AdaLNZero(time_dim, dim)

    def forward(self, x, c, kv):
        # x: (B, N, C); c: (B, time_dim); kv: (B, M, kv_dim)
        shift, scale, gate = self.adaln1(c)
        x = x + gate.unsqueeze(1) * self.self_attn(_modulate_seq(self.norm1(x), shift, scale))

        shift, scale, gate = self.adaln2(c)
        x = x + gate.unsqueeze(1) * self.cross_attn(
            _modulate_seq(self.norm2(x), shift, scale), kv=kv
        )

        shift, scale, gate = self.adaln3(c)
        x = x + gate.unsqueeze(1) * self.mlp(_modulate_seq(self.norm3(x), shift, scale))
        return x


# ---------------------------------------------------------------------------
# LR conditioning encoder (multi-scale K/V for cross-attention)
# ---------------------------------------------------------------------------

class LRConditioningEncoder(nn.Module):
    """
    Lift raw LR (B, 1, h, w) to two K/V feature scales:
        f_l4: (B, ch_l4, h,   w)
        f_l5: (B, ch_l5, h/2, w/2)
    Channels are matched to the transformer levels they feed.
    """

    def __init__(self, in_ch: int, ch_l4: int, ch_l5: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, ch_l4, 3, padding=1),
            nn.GroupNorm(32, ch_l4),
            nn.SiLU(),
            nn.Conv2d(ch_l4, ch_l4, 3, padding=1),
            nn.GroupNorm(32, ch_l4),
            nn.SiLU(),
        )
        self.down = nn.Sequential(
            nn.Conv2d(ch_l4, ch_l5, 3, stride=2, padding=1),
            nn.GroupNorm(32, ch_l5),
            nn.SiLU(),
        )

    def forward(self, lr_raw):
        f_l4 = self.stem(lr_raw)                              # (B, ch_l4, h, w)
        f_l5 = self.down(f_l4)                                # (B, ch_l5, h/2, w/2)
        f_l4_seq = rearrange(f_l4, 'b c h w -> b (h w) c')    # (B, h*w, ch_l4)
        f_l5_seq = rearrange(f_l5, 'b c h w -> b (h w) c')    # (B, (h/2)*(w/2), ch_l5)
        return f_l4_seq, f_l5_seq


# ---------------------------------------------------------------------------
# Down / Up sample with optional channel change
# ---------------------------------------------------------------------------

class Downsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = None):
        super().__init__()
        out_ch = out_ch if out_ch is not None else in_ch
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = None):
        super().__init__()
        out_ch = out_ch if out_ch is not None else in_ch
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


# ---------------------------------------------------------------------------
# BathyUNet
# ---------------------------------------------------------------------------

class BathyUNet(nn.Module):
    """Hybrid CNN/transformer U-Net for conditional bathymetric SR.

    Args:
        base_ch:            Base channel count.
        ch_mult:            Per-level multipliers (length 6 for this stage).
        n_heads:            Attention heads at transformer levels.
        transformer_levels: Levels (0-indexed) that use transformer blocks.
                            Stage 3 wires LR features to levels 4 and 5.
        transformer_depth:  Transformer blocks per transformer level.
        n_res_blocks:       ResBlocks per non-transformer level.
        time_dim:           Time embedding dimension.
        in_ch / out_ch:     Input / output channels (1 for elevation).
        lr_in_ch:           Channels in the raw LR input (1).
    """

    def __init__(
        self,
        base_ch: int = 64,
        ch_mult=(1, 2, 4, 6, 8, 12),
        n_heads: int = 8,
        transformer_levels=(4, 5),
        transformer_depth: int = 2,
        n_res_blocks: int = 2,
        time_dim: int = 256,
        in_ch: int = 1,
        out_ch: int = 1,
        lr_in_ch: int = 1,
    ):
        super().__init__()
        n_levels = len(ch_mult)
        assert n_levels == 6, "Stage 3 architecture spec requires 6 levels"
        transformer_levels = tuple(transformer_levels)
        assert set(transformer_levels).issubset(range(n_levels))
        # The LR conditioning encoder currently produces features at exactly two
        # scales (lr resolution and lr/2). Wiring up arbitrary transformer-level
        # placements would need more encoder branches; Stage 3 hardwires (4, 5).
        assert transformer_levels == (4, 5), (
            "Stage 3 LR-feature plumbing assumes transformer_levels = (4, 5)"
        )
        self.transformer_levels = set(transformer_levels)
        self.transformer_depth = transformer_depth
        self.n_res_blocks = n_res_blocks
        self.n_levels = n_levels

        channels = [base_ch * m for m in ch_mult]
        self.channels = channels
        ch_l4, ch_l5 = channels[4], channels[5]

        # Time + LR conditioning
        self.time_mlp = TimeMLP(sin_dim=time_dim, time_dim=time_dim)
        self.lr_encoder = LRConditioningEncoder(lr_in_ch, ch_l4, ch_l5)

        # Input projection
        self.in_conv = nn.Conv2d(in_ch, channels[0], 3, padding=1)

        # ------ Encoder ------
        # Channel changes between levels happen inside Downsample (and Upsample
        # in the decoder). All level blocks therefore operate at a single
        # constant channel count — convenient for both ResBlocks and
        # transformer blocks, which require a fixed channel dim.
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        for lvl in range(n_levels):
            level_ch = channels[lvl]
            level_blocks = nn.ModuleList()
            if lvl in self.transformer_levels:
                for _ in range(transformer_depth):
                    level_blocks.append(TransformerBlock(level_ch, n_heads, time_dim))
            else:
                for _ in range(n_res_blocks):
                    level_blocks.append(ResBlock(level_ch, level_ch, time_dim))
            self.enc_blocks.append(level_blocks)

            if lvl < n_levels - 1:
                next_ch = channels[lvl + 1]
                self.downs.append(Downsample(level_ch, next_ch))
            else:
                self.downs.append(None)

        # ------ Decoder ------
        self.dec_blocks = nn.ModuleList()      # one per decoder level
        self.dec_skip_proj = nn.ModuleList()   # 1×1 conv after cat at transformer levels (None elsewhere)
        self.ups = nn.ModuleList()
        # Iterate decoder levels lvl = n_levels-2 .. 0 (the bottleneck stays single-sided)
        for lvl in reversed(range(n_levels - 1)):
            level_ch = channels[lvl]
            higher_ch = channels[lvl + 1]
            self.ups.append(Upsample(higher_ch, level_ch))
            cat_ch = 2 * level_ch  # after concat with the encoder skip
            level_blocks = nn.ModuleList()
            if lvl in self.transformer_levels:
                # Project cat_ch → level_ch before entering transformer blocks.
                self.dec_skip_proj.append(nn.Conv2d(cat_ch, level_ch, 1))
                for _ in range(transformer_depth):
                    level_blocks.append(TransformerBlock(level_ch, n_heads, time_dim))
            else:
                self.dec_skip_proj.append(None)
                level_blocks.append(ResBlock(cat_ch, level_ch, time_dim))
                for _ in range(n_res_blocks - 1):
                    level_blocks.append(ResBlock(level_ch, level_ch, time_dim))
            self.dec_blocks.append(level_blocks)

        # Output: GroupNorm → SiLU → 1×1 Conv2d, zero-initialized
        self.out_norm = nn.GroupNorm(32, channels[0])
        self.out_conv = nn.Conv2d(channels[0], out_ch, 1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def _apply_level(self, h, level_blocks, lvl, c, lr_feats):
        """Run all blocks at `lvl`. Transformer levels operate in (B, N, C)."""
        if lvl in self.transformer_levels:
            B, C, H, W = h.shape
            h_seq = rearrange(h, 'b c h w -> b (h w) c')
            kv = lr_feats[lvl]
            for block in level_blocks:
                h_seq = block(h_seq, c, kv)
            h = rearrange(h_seq, 'b (h w) c -> b c h w', h=H, w=W)
        else:
            for block in level_blocks:
                h = block(h, c)
        return h

    def forward(self, x, t, lr_raw):
        c = self.time_mlp(t)                              # (B, time_dim)
        f_l4, f_l5 = self.lr_encoder(lr_raw)
        lr_feats = {4: f_l4, 5: f_l5}

        h = self.in_conv(x)

        # Encoder: keep a skip after every level
        skips = []
        for lvl in range(self.n_levels):
            h = self._apply_level(h, self.enc_blocks[lvl], lvl, c, lr_feats)
            skips.append(h)
            if self.downs[lvl] is not None:
                h = self.downs[lvl](h)

        # The bottleneck skip is the bottleneck output itself — discard.
        skips.pop()

        # Decoder: lvl runs n_levels-2 .. 0 in that order.
        for i, lvl in enumerate(reversed(range(self.n_levels - 1))):
            h = self.ups[i](h)
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            if lvl in self.transformer_levels:
                h = self.dec_skip_proj[i](h)
            h = self._apply_level(h, self.dec_blocks[i], lvl, c, lr_feats)

        return self.out_conv(F.silu(self.out_norm(h)))
