"""
All training hyperparameters in one place.
Import Config and pass it through your training scripts.

Stage 2 expansion: adds first-class sr_ratio, flow-matching knobs, CFG
plumbing, transformer architecture knobs, and DDP / mixed-precision /
DataLoader knobs. Legacy DDPM/DDIM fields are kept for back-compat
during the staged rewrite and will be removed in Stage 4.
"""

import os
from dataclasses import dataclass, field
from typing import Tuple


def _scratch(*parts: str) -> str:
    """Build a path under the running user's Sherlock scratch space."""
    return os.path.expandvars(os.path.join('/scratch/users/$USER', *parts))


@dataclass
class Config:
    # ---- Data ---------------------------------------------------------------
    patches_dir: str = field(default_factory=lambda: _scratch('bathy_patches'))
    lr_tif: str = 'data_for_model/MorroBay_all_2020_interp_32m_converted.tif'
    patch_size: int = 512            # HR patch size (pixels at 2m resolution)
    lr_patch_size: int = 32          # LR patch size before upsampling (= patch_size // sr_ratio)
    sr_ratio: int = 16               # Super-resolution ratio (32m -> 2m)
    stride: int = 256                # Patch extraction stride (50% overlap at 512)

    # ---- Model architecture ------------------------------------------------
    base_ch: int = 64                # Base channel count in U-Net
    ch_mult: Tuple[int, ...] = (1, 2, 4, 6, 8, 12)  # Per-level channel multipliers (6 levels)
    n_heads: int = 8                 # Attention heads
    transformer_levels: Tuple[int, ...] = (4, 5)    # Levels that use transformer blocks (0-indexed)
    transformer_depth: int = 2       # Transformer blocks per transformer level
    time_dim: int = 256              # Time / noise-level embedding dimension
    dropout: float = 0.0             # Dropout (flow matching does not need it)

    # ---- Flow matching ------------------------------------------------------
    time_sampling: str = "logit_normal"   # "uniform" or "logit_normal"
    logit_normal_mean: float = 0.0
    logit_normal_std: float = 1.0
    n_sampling_steps: int = 50            # Replaces ddim_steps
    sampler: str = "heun"                 # "heun" or "euler"
    churn: float = 0.0                    # Langevin churn at inference (0 = pure deterministic)

    # ---- Classifier-free guidance ------------------------------------------
    cfg_dropout_prob: float = 0.1    # Conditioning-dropout probability during training
    cfg_scale: float = 1.5           # Default guidance scale at inference

    # ---- Loss --------------------------------------------------------------
    spectral_loss_weight: float = 0.0          # Default off, configurable on
    spectral_loss_mode: str = "radial"         # "radial" or "2d"
    spectral_loss_snr_weighting: bool = True

    # ---- Training infrastructure -------------------------------------------
    batch_size: int = 16
    learning_rate: float = 2e-4
    n_epochs: int = 500
    grad_clip: float = 1.0
    precision: str = "bf16"          # "bf16" or "fp32"
    compile_model: bool = True       # torch.compile the model
    grad_accum_steps: int = 1
    ema_decay: float = 0.9999
    ema_warmup_steps: int = 1000     # EMA updates begin after this many steps

    # ---- Inference ---------------------------------------------------------
    n_realizations_per_forward: int = 10  # Cap on batched realizations per forward pass
    tiles_per_batch: int = 1              # Tiles processed per forward (memory-bounded)

    # ---- Logging / checkpointing -------------------------------------------
    out_dir: str = field(default_factory=lambda: _scratch('bathy_diffusion'))
    log_every: int = 100
    sample_every: int = 1000
    save_every: int = 5000
    n_sample_display: int = 4

    # ---- DataLoader --------------------------------------------------------
    num_workers: int = 16
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4

    # ---- Distributed -------------------------------------------------------
    distributed: bool = True              # DDP scaffolding (works on 1 GPU too)
    ddp_find_unused_params: bool = False

    def __post_init__(self):
        # SR-ratio invariant
        assert self.patch_size % self.sr_ratio == 0, (
            f"patch_size ({self.patch_size}) must be divisible by "
            f"sr_ratio ({self.sr_ratio})"
        )
        assert self.patch_size // self.sr_ratio == self.lr_patch_size, (
            f"lr_patch_size ({self.lr_patch_size}) must equal "
            f"patch_size // sr_ratio = {self.patch_size // self.sr_ratio}"
        )
        # transformer_levels must be valid indices into ch_mult
        max_level = len(self.ch_mult) - 1
        for lvl in self.transformer_levels:
            assert 0 <= lvl <= max_level, (
                f"transformer_level {lvl} out of range [0, {max_level}] "
                f"for ch_mult of length {len(self.ch_mult)}"
            )
        # Validate enum-like string fields
        assert self.time_sampling in ("uniform", "logit_normal"), self.time_sampling
        assert self.sampler in ("heun", "euler"), self.sampler
        assert self.precision in ("bf16", "fp32"), self.precision
        assert self.spectral_loss_mode in ("radial", "2d"), self.spectral_loss_mode
