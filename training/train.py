"""
Main training loop for the rectified-flow bathymetric SR model.

Stage 4: wired to the new RectifiedFlow + FlowMatchingLoss. The DataLoader,
autocast/compile/DDP plumbing is the same as Stage 1's; Stage 5 will add
mixed precision, torch.compile, distributed training, and the persistent /
prefetching DataLoader knobs from cfg.

Run locally for a smoke test (from the project root):
    python -m training.train --patches_dir /path/to/patches --out_dir /tmp/test --n_epochs 1

Run on Sherlock:
    sbatch sherlock/train.sbatch
"""

import os
import argparse
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.unet import BathyUNet
from models.diffusion import RectifiedFlow
from models.losses import FlowMatchingLoss
from preprocessing.data_utils import BathymetryPatchDataset
from training.config import Config
from utils.device import get_device


# ---------------------------------------------------------------------------
# EMA helper
# ---------------------------------------------------------------------------

class EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s_param, param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def get_model(self):
        return self.shadow


# ---------------------------------------------------------------------------
# Sampling utility for TensorBoard
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_and_log(flow_ema, lr_raw_batch, writer, step, n_display,
                   display_steps: int = 20):
    """Generate one realization per display patch and log to TensorBoard.

    Uses a shortened step count and no CFG to keep training-time sampling
    affordable. Production-quality samples come from inference/.
    """
    flow_ema.eval()
    lr_display = lr_raw_batch[:n_display]
    samples = []
    for i in range(n_display):
        lr_i = lr_display[i:i + 1]
        s = flow_ema.sample(lr_i, n_realizations=1, n_steps=display_steps,
                            cfg_scale=1.0)
        samples.append(s[0])
    samples = torch.stack(samples)

    def norm_for_display(t):
        t = t - t.amin(dim=(1, 2, 3), keepdim=True)
        t = t / (t.amax(dim=(1, 2, 3), keepdim=True) + 1e-8)
        return t

    writer.add_images('lr_conditioning', norm_for_display(lr_display), step)
    writer.add_images('hr_generated',    norm_for_display(samples),    step)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: Config):
    device = get_device()
    print(f"Using device: {device}")

    # Data
    train_dataset = BathymetryPatchDataset(
        os.path.join(cfg.patches_dir, 'train'), augment=True)
    val_dataset = BathymetryPatchDataset(
        os.path.join(cfg.patches_dir, 'val'),   augment=False)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size,
                              shuffle=True, num_workers=cfg.num_workers,
                              pin_memory=cfg.pin_memory, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size,
                            shuffle=False, num_workers=cfg.num_workers,
                            pin_memory=cfg.pin_memory)

    print(f"Train: {len(train_dataset)} patches | Val: {len(val_dataset)} patches")

    # Model
    unet = BathyUNet(
        base_ch=cfg.base_ch,
        ch_mult=cfg.ch_mult,
        n_heads=cfg.n_heads,
        transformer_levels=cfg.transformer_levels,
        transformer_depth=cfg.transformer_depth,
        time_dim=cfg.time_dim,
    ).to(device)
    flow = RectifiedFlow(unet, cfg).to(device)

    n_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}  ({n_params/1e6:.2f} M)")

    # EMA on the full flow (includes null_lr buffer)
    ema = EMA(flow, decay=cfg.ema_decay)

    criterion = FlowMatchingLoss(cfg)
    optimizer = torch.optim.AdamW(flow.parameters(), lr=cfg.learning_rate,
                                  betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.n_epochs * len(train_loader), eta_min=1e-6)

    ckpt_dir = os.path.join(cfg.out_dir, 'checkpoints')
    log_dir  = os.path.join(cfg.out_dir, 'logs')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # Fixed display batch for TensorBoard visualization
    display_batch = next(iter(val_loader))
    display_lr_raw = display_batch['lr_raw'][:cfg.n_sample_display].to(device)
    display_hr     = display_batch['hr'][:cfg.n_sample_display].to(device)

    # Resume
    start_epoch = 0
    global_step = 0
    latest_ckpt = _find_latest_checkpoint(ckpt_dir)
    if latest_ckpt:
        print(f"Resuming from {latest_ckpt}")
        state = torch.load(latest_ckpt, map_location=device)
        flow.load_state_dict(state['model'])
        ema.shadow.load_state_dict(state['ema'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start_epoch = state['epoch'] + 1
        global_step = state['global_step']

    for epoch in range(start_epoch, cfg.n_epochs):
        flow.train()
        epoch_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.n_epochs}",
                          leave=False):
            hr     = batch['hr'].to(device)
            lr_raw = batch['lr_raw'].to(device)

            output = flow.training_step(hr, lr_raw)
            losses = criterion(output)
            loss = losses['total']

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(flow.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            if global_step >= cfg.ema_warmup_steps:
                ema.update(flow)

            epoch_loss += loss.item()
            global_step += 1

            if global_step % cfg.log_every == 0:
                writer.add_scalar('loss/total',    loss.item(),            global_step)
                writer.add_scalar('loss/fm',       losses['fm'].item(),    global_step)
                writer.add_scalar('loss/spectral', losses['spectral'].item(), global_step)
                writer.add_scalar('lr',            scheduler.get_last_lr()[0], global_step)
                print(f"  step {global_step:6d}  loss={loss.item():.4f}  "
                      f"fm={losses['fm'].item():.4f}  "
                      f"spec={losses['spectral'].item():.4f}  "
                      f"lr={scheduler.get_last_lr()[0]:.2e}", flush=True)

            if global_step % cfg.sample_every == 0:
                sample_and_log(ema.get_model(), display_lr_raw, writer,
                               global_step, cfg.n_sample_display)
                writer.add_images('hr_groundtruth',
                                  _norm_display(display_hr), global_step)

            if global_step % cfg.save_every == 0:
                _save_checkpoint(ckpt_dir, flow, ema, optimizer, scheduler,
                                 epoch, global_step)

        val_loss = _evaluate(flow, criterion, val_loader, device)
        writer.add_scalar('loss/val', val_loss, global_step)
        print(f"Epoch {epoch+1}: train_loss={epoch_loss/len(train_loader):.4f}  "
              f"val_loss={val_loss:.4f}")

    _save_checkpoint(ckpt_dir, flow, ema, optimizer, scheduler,
                     cfg.n_epochs - 1, global_step, tag='final')
    writer.close()
    print("Training complete.")


def _evaluate(flow, criterion, loader, device):
    """Mean validation loss (no sampling)."""
    flow.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            hr     = batch['hr'].to(device)
            lr_raw = batch['lr_raw'].to(device)
            # training_step uses self.training to gate CFG dropout, so eval()
            # automatically disables it. The arithmetic is otherwise identical.
            output = flow.training_step(hr, lr_raw)
            losses = criterion(output)
            total += losses['total'].item()
            n += 1
    flow.train()
    return total / max(n, 1)


def _norm_display(t):
    t = t - t.amin(dim=(1, 2, 3), keepdim=True)
    return t / (t.amax(dim=(1, 2, 3), keepdim=True) + 1e-8)


def _find_latest_checkpoint(ckpt_dir):
    import glob
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, 'ckpt_step*.pt')))
    return ckpts[-1] if ckpts else None


def _save_checkpoint(ckpt_dir, flow, ema, optimizer, scheduler,
                     epoch, step, tag=None):
    suffix = tag if tag else f'step{step:08d}'
    path = os.path.join(ckpt_dir, f'ckpt_{suffix}.pt')
    torch.save({
        'model':       flow.state_dict(),       # full flow state (unet + null_lr)
        'ema':         ema.shadow.state_dict(), # full EMA shadow state
        'optimizer':   optimizer.state_dict(),
        'scheduler':   scheduler.state_dict(),
        'epoch':       epoch,
        'global_step': step,
    }, path)
    print(f"Saved checkpoint: {path}")


def main():
    parser = argparse.ArgumentParser(description='Train rectified-flow bathymetry SR model')
    parser.add_argument('--patches_dir', default=None)
    parser.add_argument('--out_dir',     default=None)
    parser.add_argument('--batch_size',  type=int, default=None)
    parser.add_argument('--n_epochs',    type=int, default=None)
    parser.add_argument('--lr',          type=float, default=None)
    parser.add_argument('--base_ch',     type=int, default=None)
    parser.add_argument('--log_every',   type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=None,
                        help='DataLoader worker count (set to 0 for MPS / debugging)')
    args = parser.parse_args()

    cfg = Config()
    if args.patches_dir: cfg.patches_dir = args.patches_dir
    if args.out_dir:     cfg.out_dir = args.out_dir
    if args.batch_size:  cfg.batch_size = args.batch_size
    if args.n_epochs:    cfg.n_epochs = args.n_epochs
    if args.lr:          cfg.learning_rate = args.lr
    if args.base_ch:     cfg.base_ch = args.base_ch
    if args.log_every:   cfg.log_every = args.log_every
    if args.num_workers is not None: cfg.num_workers = args.num_workers

    train(cfg)


if __name__ == '__main__':
    main()
