"""
All training hyperparameters in one place.
Import Config and pass it through your training scripts.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Config:
    # ---- Data ---------------------------------------------------------------
    patches_dir: str = '/scratch/users/scdobbs/bathy_patches'
    lr_tif: str = 'data_for_model/MorroBay_all_2020_interp_32m_converted.tif'
    patch_size: int = 64          # HR patch size (pixels at 2m resolution)
    lr_patch_size: int = 4        # LR patch size before upsampling (patch_size // 16)

    # ---- Model --------------------------------------------------------------
    base_ch: int = 64             # Base channel count in U-Net
    time_dim: int = 256           # Time embedding dimension
    dropout: float = 0.1

    # ---- Diffusion ----------------------------------------------------------
    n_timesteps: int = 1000       # Total DDPM diffusion steps
    beta_start: float = 1e-4
    beta_end: float = 0.02
    ddim_steps: int = 50          # Steps used during DDIM inference

    # ---- Training -----------------------------------------------------------
    batch_size: int = 16
    learning_rate: float = 2e-4
    n_epochs: int = 500
    grad_clip: float = 1.0        # Gradient norm clipping
    ema_decay: float = 0.9999     # EMA of model weights (critical for quality)
    spectral_loss_weight: float = 0.1

    # ---- Logging / checkpointing -------------------------------------------
    out_dir: str = '/scratch/users/scdobbs/bathy_diffusion'
    log_every: int = 100          # Log loss every N iterations
    sample_every: int = 1000      # Save sample images every N iterations
    save_every: int = 5000        # Save checkpoint every N iterations
    n_sample_display: int = 4     # Number of patches to visualize during training

    # ---- Hardware ----------------------------------------------------------
    num_workers: int = 4
    pin_memory: bool = False   # Must be False for MPS (Apple Silicon); True only for CUDA

    def __post_init__(self):
        assert self.patch_size // 16 == self.lr_patch_size, \
            "lr_patch_size must equal patch_size // 16 (for 16x SR ratio)"
