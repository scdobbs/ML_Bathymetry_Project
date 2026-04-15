"""
Conditional U-Net for the diffusion model.

Architecture:
    - Encoder–decoder with skip connections
    - Input: concatenation of [noisy HR (1ch), upsampled LR (1ch)] = 2 channels
    - Time embedding added to every residual block
    - Self-attention at the two lowest spatial resolutions
    - Output: predicted noise epsilon (1ch), same spatial size as input

Reference: Ho et al. 2020 (DDPM), Saharia et al. 2021 (SR3)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Time / positional embedding
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional encoding for diffusion timestep."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: (B,) integer timesteps
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)


class TimeEmbedMLP(nn.Module):
    """Project sinusoidal embedding to model hidden dimension."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim * 4),
            nn.SiLU(),
            nn.Linear(out_dim * 4, out_dim),
        )

    def forward(self, t_emb):
        return self.net(t_emb)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """
    Residual block with GroupNorm, SiLU, and time embedding injection.
    Optionally changes number of channels via 1×1 conv.
    """

    def __init__(self, in_ch, out_ch, time_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Multi-head self-attention for spatial feature maps."""

    def __init__(self, ch, n_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(min(32, ch), ch)
        self.attn = nn.MultiheadAttention(ch, n_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)
        h, _ = self.attn(h, h, h)
        h = h.permute(0, 2, 1).view(B, C, H, W)
        return x + h


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """
    Conditional U-Net for bathymetry diffusion SR.

    Args:
        in_ch:       Input channels (2 = noisy HR + upsampled LR)
        out_ch:      Output channels (1 = predicted noise)
        base_ch:     Base number of feature channels
        ch_mult:     Channel multiplier at each resolution level
        n_res_blocks: Residual blocks per resolution level
        attn_levels: Which encoder levels use self-attention (0-indexed from bottom)
        time_dim:    Dimension of time embedding
        dropout:     Dropout probability in res blocks
    """

    def __init__(
        self,
        in_ch=2,
        out_ch=1,
        base_ch=64,
        ch_mult=(1, 2, 4, 8),
        n_res_blocks=2,
        attn_levels=(2, 3),
        time_dim=256,
        dropout=0.1,
    ):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(base_ch)
        self.time_mlp = TimeEmbedMLP(base_ch, time_dim)

        channels = [base_ch * m for m in ch_mult]   # e.g. [64, 128, 256, 512]
        n_levels = len(channels)

        # Input projection
        self.in_conv = nn.Conv2d(in_ch, channels[0], 3, padding=1)

        # Encoder
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        in_ch_enc = channels[0]
        enc_ch_list = [in_ch_enc]   # track channels for skip connections

        for level, ch in enumerate(channels):
            for _ in range(n_res_blocks):
                block = ResBlock(in_ch_enc, ch, time_dim, dropout)
                self.enc_blocks.append(block)
                if level in attn_levels:
                    self.enc_blocks.append(SelfAttention(ch))
                else:
                    self.enc_blocks.append(nn.Identity())
                in_ch_enc = ch
                enc_ch_list.append(ch)
            if level < n_levels - 1:
                self.downs.append(Downsample(ch))
                enc_ch_list.append(ch)  # after downsample, same ch
            else:
                self.downs.append(nn.Identity())  # no downsample at bottleneck input

        # Bottleneck
        self.mid_block1 = ResBlock(channels[-1], channels[-1], time_dim, dropout)
        self.mid_attn   = SelfAttention(channels[-1])
        self.mid_block2 = ResBlock(channels[-1], channels[-1], time_dim, dropout)

        # Decoder
        self.dec_blocks = nn.ModuleList()
        self.ups = nn.ModuleList()
        in_ch_dec = channels[-1]

        for level in reversed(range(n_levels)):
            ch = channels[level]
            for i in range(n_res_blocks + 1):  # +1 for the skip-connection merge step
                skip_ch = enc_ch_list.pop()
                block = ResBlock(in_ch_dec + skip_ch, ch, time_dim, dropout)
                self.dec_blocks.append(block)
                if level in attn_levels:
                    self.dec_blocks.append(SelfAttention(ch))
                else:
                    self.dec_blocks.append(nn.Identity())
                in_ch_dec = ch
            if level > 0:
                self.ups.append(Upsample(ch))
            else:
                self.ups.append(nn.Identity())

        # Output projection
        self.out_norm = nn.GroupNorm(min(32, channels[0]), channels[0])
        self.out_conv = nn.Conv2d(channels[0], out_ch, 1)

    def forward(self, x, t):
        """
        Args:
            x: (B, 2, H, W) — concatenation of noisy HR and upsampled LR
            t: (B,) — diffusion timestep indices
        Returns:
            (B, 1, H, W) — predicted noise epsilon
        """
        t_emb = self.time_mlp(self.time_emb(t))  # (B, time_dim)

        h = self.in_conv(x)
        skips = [h]

        # Encode
        block_iter = iter(self.enc_blocks)
        down_iter  = iter(self.downs)
        level_sizes = self._level_block_counts()

        enc_idx = 0
        for level, n_blocks in enumerate(level_sizes):
            for _ in range(n_blocks):
                res_block = next(block_iter)
                attn_block = next(block_iter)
                if isinstance(res_block, ResBlock):
                    h = res_block(h, t_emb)
                    h = attn_block(h)
                    skips.append(h)
            down = next(down_iter)
            if not isinstance(down, nn.Identity):
                h = down(h)
                skips.append(h)

        # Bottleneck
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # Decode
        dec_block_iter = iter(self.dec_blocks)
        up_iter = iter(self.ups)
        for level, n_blocks in reversed(list(enumerate(level_sizes))):
            for _ in range(n_blocks + 1):
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                res_block = next(dec_block_iter)
                attn_block = next(dec_block_iter)
                h = res_block(h, t_emb)
                h = attn_block(h)
            up = next(up_iter)
            if not isinstance(up, nn.Identity):
                h = up(h)

        return self.out_conv(F.silu(self.out_norm(h)))

    def _level_block_counts(self):
        """Return n_res_blocks per level (inferred from enc_blocks length)."""
        # enc_blocks has (n_res_blocks * 2) entries per level (ResBlock + Attn/Identity)
        # We can recover this from the module list; here we store it at init time instead.
        # This is a helper used in forward — we store n_res_blocks directly.
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Simpler, explicit U-Net (recommended for this project)
# ---------------------------------------------------------------------------

class BathyUNet(nn.Module):
    """
    Explicit 4-level U-Net for bathymetric diffusion SR.
    Hard-coded structure avoids the complexity of the generic UNet above
    while being easy to read and modify.

    Spatial sizes at each level (for 64×64 input):
        Level 0: 64×64,  ch=base_ch
        Level 1: 32×32,  ch=base_ch*2
        Level 2: 16×16,  ch=base_ch*4  + self-attention
        Level 3: 8×8,    ch=base_ch*8  + self-attention  (bottleneck)
    """

    def __init__(self, in_ch=2, out_ch=1, base_ch=64, time_dim=256, dropout=0.1):
        super().__init__()
        ch = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]
        td = time_dim

        self.time_emb = SinusoidalTimeEmbedding(base_ch)
        self.time_mlp = TimeEmbedMLP(base_ch, td)

        self.in_conv = nn.Conv2d(in_ch, ch[0], 3, padding=1)

        # Encoder
        self.enc0a = ResBlock(ch[0], ch[0], td, dropout)
        self.enc0b = ResBlock(ch[0], ch[0], td, dropout)
        self.down0  = Downsample(ch[0])

        self.enc1a = ResBlock(ch[0], ch[1], td, dropout)
        self.enc1b = ResBlock(ch[1], ch[1], td, dropout)
        self.down1  = Downsample(ch[1])

        self.enc2a = ResBlock(ch[1], ch[2], td, dropout)
        self.enc2b = ResBlock(ch[2], ch[2], td, dropout)
        self.attn2  = SelfAttention(ch[2])
        self.down2  = Downsample(ch[2])

        # Bottleneck
        self.bot1 = ResBlock(ch[2], ch[3], td, dropout)
        self.bot_attn = SelfAttention(ch[3])
        self.bot2 = ResBlock(ch[3], ch[3], td, dropout)

        # Decoder
        self.up2   = Upsample(ch[3])
        self.dec2a = ResBlock(ch[3] + ch[2], ch[2], td, dropout)
        self.dec2b = ResBlock(ch[2], ch[2], td, dropout)
        self.dattn2 = SelfAttention(ch[2])

        self.up1   = Upsample(ch[2])
        self.dec1a = ResBlock(ch[2] + ch[1], ch[1], td, dropout)
        self.dec1b = ResBlock(ch[1], ch[1], td, dropout)

        self.up0   = Upsample(ch[1])
        self.dec0a = ResBlock(ch[1] + ch[0], ch[0], td, dropout)
        self.dec0b = ResBlock(ch[0], ch[0], td, dropout)

        self.out_norm = nn.GroupNorm(min(32, ch[0]), ch[0])
        self.out_conv = nn.Conv2d(ch[0], out_ch, 1)

    def forward(self, x, t):
        """
        Args:
            x: (B, 2, H, W) — [noisy_hr | lr_upsampled]
            t: (B,) — diffusion timestep
        Returns:
            (B, 1, H, W) — predicted noise
        """
        t_emb = self.time_mlp(self.time_emb(t))

        h = self.in_conv(x)

        # Encode
        h = s0 = self.enc0b(self.enc0a(h, t_emb), t_emb)   # 64×64
        h = self.down0(h)
        h = s1 = self.enc1b(self.enc1a(h, t_emb), t_emb)   # 32×32
        h = self.down1(h)
        h = self.enc2a(h, t_emb)
        h = s2 = self.attn2(self.enc2b(h, t_emb))           # 16×16
        h = self.down2(h)

        # Bottleneck 8×8
        h = self.bot_attn(self.bot1(h, t_emb))
        h = self.bot2(h, t_emb)

        # Decode
        h = self.up2(h)
        h = torch.cat([h, s2], dim=1)
        h = self.dattn2(self.dec2b(self.dec2a(h, t_emb), t_emb))   # 16×16

        h = self.up1(h)
        h = torch.cat([h, s1], dim=1)
        h = self.dec1b(self.dec1a(h, t_emb), t_emb)                # 32×32

        h = self.up0(h)
        h = torch.cat([h, s0], dim=1)
        h = self.dec0b(self.dec0a(h, t_emb), t_emb)                # 64×64

        return self.out_conv(F.silu(self.out_norm(h)))
