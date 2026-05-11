"""
Main training loop for the bathymetric super-resolution diffusion model.

Run locally for a quick smoke test:
    python training/train.py --patches_dir /path/to/patches --out_dir /tmp/test_run --n_epochs 1

Run on Sherlock:
    sbatch sherlock/train.sbatch
"""

import os
import sys
import argparse
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

# Make sure parent directory is on the path when running as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet import BathyUNet
from models.diffusion import GaussianDiffusion
from models.losses import CombinedDiffusionLoss
from preprocessing.data_utils import BathymetryPatchDataset
from training.config import Config


# ---------------------------------------------------------------------------
# EMA helper
# ---------------------------------------------------------------------------

class EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()

    @torch.no_grad()
    def update(self, model):
        for s_param, param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def get_model(self):
        return self.shadow


# ---------------------------------------------------------------------------
# Sampling utility for TensorBoard visualization
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_and_log(diffusion_ema, lr_batch, writer, step, n_display):
    """Generate one realization per display patch and log to TensorBoard."""
    diffusion_ema.eval()
    lr_display = lr_batch[:n_display]
    samples = []
    for i in range(n_display):
        lr_i = lr_display[i:i+1]
        s = diffusion_ema.sample_ddim(lr_i, n_realizations=1, n_steps=50)
        samples.append(s[0])     # (1, H, W)
    samples = torch.stack(samples)  # (n_display, 1, H, W)

    # Normalize to [0, 1] for display
    def norm_for_display(t):
        t = t - t.min()
        t = t / (t.max() + 1e-8)
        return t

    writer.add_images('lr_conditioning', norm_for_display(lr_display), step)
    writer.add_images('hr_generated', norm_for_display(samples), step)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: Config):
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Using device: {device}")

    # Data — patches_dir may be a single path or a list of paths
    dirs = cfg.patches_dir if isinstance(cfg.patches_dir, list) else [cfg.patches_dir]
    train_dataset = BathymetryPatchDataset(
        [os.path.join(d, 'train') for d in dirs], augment=True)
    val_dataset   = BathymetryPatchDataset(
        [os.path.join(d, 'val')   for d in dirs], augment=False)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size,
                              shuffle=True, num_workers=cfg.num_workers,
                              pin_memory=cfg.pin_memory, drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=cfg.batch_size,
                              shuffle=False, num_workers=cfg.num_workers,
                              pin_memory=cfg.pin_memory)

    print(f"Train: {len(train_dataset)} patches | Val: {len(val_dataset)} patches")

    # Model
    unet = BathyUNet(
        in_ch=2, out_ch=1,
        base_ch=cfg.base_ch, time_dim=cfg.time_dim, dropout=cfg.dropout
    ).to(device)

    diffusion = GaussianDiffusion(unet, n_timesteps=cfg.n_timesteps,
                                  beta_start=cfg.beta_start,
                                  beta_end=cfg.beta_end).to(device)

    n_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # EMA model (for inference — higher quality than raw weights)
    ema = EMA(diffusion, decay=cfg.ema_decay)

    # Loss
    criterion = CombinedDiffusionLoss(
        spectral_weight=cfg.spectral_loss_weight,
        gradient_weight=cfg.gradient_loss_weight,
        x0_weight=cfg.x0_loss_weight,
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), lr=cfg.learning_rate,
                                  betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.n_epochs * len(train_loader), eta_min=1e-6)

    # Output directories
    ckpt_dir = os.path.join(cfg.out_dir, 'checkpoints')
    log_dir  = os.path.join(cfg.out_dir, 'logs')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # Grab fixed display batch from validation set
    display_batch = next(iter(val_loader))
    display_lr = display_batch['lr'][:cfg.n_sample_display].to(device)
    display_hr = display_batch['hr'][:cfg.n_sample_display].to(device)

    # Resume or initialise from checkpoint
    start_epoch = 0
    global_step = 0
    latest_ckpt = _find_latest_checkpoint(ckpt_dir)
    if latest_ckpt:
        # Full resume (same out_dir): restore model + optimiser + scheduler
        print(f"Resuming from {latest_ckpt}")
        state = torch.load(latest_ckpt, map_location=device, weights_only=False)
        unet.load_state_dict(state['model'])
        ema.shadow.load_state_dict(state['ema'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start_epoch = state['epoch'] + 1
        global_step = state['global_step']
    elif cfg.pretrained_ckpt:
        # Fine-tune init: load model weights only, reset optimiser/scheduler
        print(f"Fine-tune initialisation from {cfg.pretrained_ckpt}")
        state = torch.load(cfg.pretrained_ckpt, map_location=device, weights_only=False)
        unet.load_state_dict(state['model'])
        ema.shadow.load_state_dict(state['ema'])
        print("  Loaded model weights. Optimiser and scheduler reset (fresh fine-tune).")

    # Training
    for epoch in range(start_epoch, cfg.n_epochs):
        unet.train()
        epoch_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.n_epochs}",
                          leave=False):
            lr  = batch['lr'].to(device)   # (B, 1, H, W)
            hr  = batch['hr'].to(device)   # (B, 1, H, W)

            # Sample random timesteps
            B = hr.shape[0]
            t = torch.randint(0, cfg.n_timesteps, (B,), device=device)

            # Residual mode: train on (HR - LR) so model learns to add texture
            x0 = (hr - lr) if cfg.residual_mode else hr

            # Add noise
            predicted_noise, target_noise = diffusion.p_losses(x0, lr, t)

            # Compute loss (with spectral component)
            sac_t  = diffusion.sqrt_alphas_cumprod[t][:, None, None, None]
            soac_t = diffusion.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
            x_t = sac_t * x0 + soac_t * target_noise

            loss, l_noise, l_spectral, l_gradient, l_x0 = criterion(
                predicted_noise, target_noise, x0, sac_t, soac_t, x_t)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(unet.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            ema.update(unet)

            epoch_loss += loss.item()
            global_step += 1

            # Log scalars
            if global_step % cfg.log_every == 0:
                lr_now = scheduler.get_last_lr()[0]
                writer.add_scalar('loss/total',    loss.item(),       global_step)
                writer.add_scalar('loss/noise',    l_noise.item(),    global_step)
                writer.add_scalar('loss/spectral', l_spectral.item(), global_step)
                writer.add_scalar('loss/gradient', l_gradient.item(), global_step)
                writer.add_scalar('loss/x0',       l_x0.item(),       global_step)
                writer.add_scalar('lr', lr_now, global_step)

            # Sample and display
            if global_step % cfg.sample_every == 0:
                sample_and_log(ema.get_model(), display_lr, writer,
                               global_step, cfg.n_sample_display)
                writer.add_images('hr_groundtruth',
                                  _norm_display(display_hr), global_step)

            # Save checkpoint
            if global_step % cfg.save_every == 0:
                _save_checkpoint(ckpt_dir, unet, ema, optimizer, scheduler,
                                 epoch, global_step)

        # Validation loss (every epoch)
        val_loss = _evaluate(diffusion, criterion, val_loader, device, cfg)
        writer.add_scalar('loss/val', val_loss, global_step)
        print(f"Epoch {epoch+1}: train_loss={epoch_loss/len(train_loader):.4f}  "
              f"val_loss={val_loss:.4f}")

    # Final save
    _save_checkpoint(ckpt_dir, unet, ema, optimizer, scheduler,
                     cfg.n_epochs - 1, global_step, tag='final')
    writer.close()
    print("Training complete.")


def _evaluate(diffusion, criterion, loader, device, cfg):
    """Compute average loss over the validation set (no sampling)."""
    diffusion.eval()
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            lr = batch['lr'].to(device)
            hr = batch['hr'].to(device)
            B  = hr.shape[0]
            t  = torch.randint(0, cfg.n_timesteps, (B,), device=device)
            x0 = (hr - lr) if cfg.residual_mode else hr
            predicted_noise, target_noise = diffusion.p_losses(x0, lr, t)
            sac_t  = diffusion.sqrt_alphas_cumprod[t][:, None, None, None]
            soac_t = diffusion.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
            x_t = sac_t * x0 + soac_t * target_noise
            loss, *_ = criterion(predicted_noise, target_noise, x0, sac_t, soac_t, x_t)
            total += loss.item()
    diffusion.train()
    return total / len(loader)


def _norm_display(t):
    t = t - t.min()
    return t / (t.max() + 1e-8)


def _find_latest_checkpoint(ckpt_dir):
    import glob
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, 'ckpt_step*.pt')))
    return ckpts[-1] if ckpts else None


def _save_checkpoint(ckpt_dir, model, ema, optimizer, scheduler,
                     epoch, step, tag=None):
    suffix = tag if tag else f'step{step:08d}'
    path = os.path.join(ckpt_dir, f'ckpt_{suffix}.pt')
    torch.save({
        'model':     model.state_dict(),
        'ema':       ema.shadow.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch':     epoch,
        'global_step': step,
    }, path)
    print(f"Saved checkpoint: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train bathymetry diffusion SR model')
    parser.add_argument('--patches_dir',    default=None, nargs='+')
    parser.add_argument('--out_dir',        default=None)
    parser.add_argument('--batch_size',     type=int,   default=None)
    parser.add_argument('--n_epochs',       type=int,   default=None)
    parser.add_argument('--lr',             type=float, default=None)
    parser.add_argument('--base_ch',        type=int,   default=None)
    parser.add_argument('--pretrained',     default=None,
                        help='Checkpoint to load weights from for fine-tuning (resets optimiser)')
    parser.add_argument('--no_residual', action='store_true',
                        help='Disable residual mode (train on full HR instead of HR-LR)')
    args = parser.parse_args()

    cfg = Config()
    if args.patches_dir:
        cfg.patches_dir = args.patches_dir if len(args.patches_dir) > 1 else args.patches_dir[0]
    if args.out_dir:     cfg.out_dir = args.out_dir
    if args.batch_size:  cfg.batch_size = args.batch_size
    if args.n_epochs:    cfg.n_epochs = args.n_epochs
    if args.lr:          cfg.learning_rate = args.lr
    if args.base_ch:     cfg.base_ch = args.base_ch
    if args.pretrained:  cfg.pretrained_ckpt = args.pretrained
    if args.no_residual: cfg.residual_mode = False

    train(cfg)


if __name__ == '__main__':
    main()
