"""
Main training loop for the rectified-flow bathymetric SR model.

Stage 5 adds the H100-class plumbing:
    - bf16 mixed precision via torch.autocast (CUDA only — no GradScaler
      needed for bf16).
    - torch.compile on the underlying U-Net (not the RectifiedFlow wrapper).
    - Single-node DistributedDataParallel that scales transparently from 1
      GPU (WORLD_SIZE=1) to N GPUs without code changes.
    - Optimized DataLoader: pin_memory, persistent_workers, prefetch_factor.
    - foreach-fused EMA update with warmup gating.
    - Gradient accumulation with DDP no_sync() on intermediate micro-batches.
    - Main-process gating for all I/O (TensorBoard, prints, checkpoints).

Single-GPU launch (laptop, solo):
    python -m training.train --patches_dir ... --out_dir ...

Multi-GPU launch (Sherlock):
    torchrun --standalone --nproc_per_node=$SLURM_GPUS_ON_NODE --module \
        training.train --patches_dir ... --out_dir ...
"""

import contextlib
import copy
import os
import argparse

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.unet import BathyUNet
from models.diffusion import RectifiedFlow
from models.losses import FlowMatchingLoss
from preprocessing.data_utils import BathymetryPatchDataset
from training.config import Config
from utils.device import get_device
from utils.distributed import (
    setup_distributed, cleanup_distributed,
    is_main_process, all_reduce_mean,
)


# ---------------------------------------------------------------------------
# Wrapper-peeling for checkpoints
# ---------------------------------------------------------------------------

def unwrap_model(m: nn.Module) -> nn.Module:
    """Strip DDP and torch.compile wrappers in any order."""
    while True:
        if hasattr(m, '_orig_mod'):           # torch.compile -> OptimizedModule
            m = m._orig_mod
        elif isinstance(m, DistributedDataParallel):
            m = m.module
        else:
            return m


# ---------------------------------------------------------------------------
# EMA with foreach-fused updates
# ---------------------------------------------------------------------------

class EMA:
    """Exponential moving average of a RectifiedFlow's parameters.

    The shadow is a deepcopy taken BEFORE DDP/compile wrapping, so the
    state dict has clean keys. Updates use torch._foreach for fused
    multi-tensor arithmetic.

    Warm-up: callers gate update() on global_step >= cfg.ema_warmup_steps.
    Before warmup, the shadow stays as the initial weights — appropriate
    because there's no useful signal to average that early.
    """

    def __init__(self, flow: nn.Module, decay: float):
        self.decay = decay
        self.shadow = copy.deepcopy(flow)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)
        # Cache shadow params once; flow params are looked up per update
        # (the same tensor objects are shared across DDP / compile wrappers,
        # so this works whether or not flow is wrapped).
        self._shadow_params = list(self.shadow.parameters())

    @torch.no_grad()
    def update(self, flow: nn.Module) -> None:
        flow_params = list(flow.parameters())
        assert len(flow_params) == len(self._shadow_params), (
            f"param count mismatch: flow={len(flow_params)} "
            f"shadow={len(self._shadow_params)}"
        )
        torch._foreach_mul_(self._shadow_params, self.decay)
        torch._foreach_add_(self._shadow_params, flow_params,
                            alpha=1.0 - self.decay)


# ---------------------------------------------------------------------------
# Sampling utility for TensorBoard
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_and_log(flow_ema, lr_raw_batch, writer, step, n_display,
                   display_steps: int = 20):
    """Generate one realization per display patch and log to TensorBoard."""
    flow_ema.eval()
    lr_display = lr_raw_batch[:n_display]
    samples = []
    for i in range(n_display):
        s = flow_ema.sample(lr_display[i:i + 1], n_realizations=1,
                            n_steps=display_steps, cfg_scale=1.0)
        samples.append(s[0])
    samples = torch.stack(samples)

    def norm_for_display(t):
        t = t - t.amin(dim=(1, 2, 3), keepdim=True)
        return t / (t.amax(dim=(1, 2, 3), keepdim=True) + 1e-8)

    writer.add_images('lr_conditioning', norm_for_display(lr_display), step)
    writer.add_images('hr_generated',    norm_for_display(samples),    step)


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def _save_checkpoint(ckpt_dir, flow, ema, optimizer, scheduler,
                     epoch, step, tag=None):
    if not is_main_process():
        return
    suffix = tag if tag else f'step{step:08d}'
    path = os.path.join(ckpt_dir, f'ckpt_{suffix}.pt')
    torch.save({
        'flow.model':   unwrap_model(flow.model).state_dict(),
        'flow.null_lr': flow.null_lr.detach().cpu(),
        'ema.model':    unwrap_model(ema.shadow.model).state_dict(),
        'ema.null_lr':  ema.shadow.null_lr.detach().cpu(),
        'optimizer':    optimizer.state_dict(),
        'scheduler':    scheduler.state_dict(),
        'epoch':        epoch,
        'global_step':  step,
    }, path)
    print(f"Saved checkpoint: {path}", flush=True)


def _load_checkpoint(path, flow, ema, optimizer, scheduler, device):
    """Load a checkpoint INTO an unwrapped flow/ema (call before DDP/compile)."""
    state = torch.load(path, map_location=device)
    unwrap_model(flow.model).load_state_dict(state['flow.model'])
    flow.null_lr.data.copy_(state['flow.null_lr'].to(device))
    unwrap_model(ema.shadow.model).load_state_dict(state['ema.model'])
    ema.shadow.null_lr.data.copy_(state['ema.null_lr'].to(device))
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    return state['epoch'], state['global_step']


def _find_latest_checkpoint(ckpt_dir):
    """Locate the most recent checkpoint to resume from.

    Step-numbered checkpoints (ckpt_step00000000.pt) take precedence — they
    are intermediate saves and the highest step number is the freshest. If
    no intermediate exists, fall back to ckpt_final.pt (clean completion
    of a previous run), which still supports incremental training.
    """
    import glob
    step_ckpts = sorted(glob.glob(os.path.join(ckpt_dir, 'ckpt_step*.pt')))
    if step_ckpts:
        return step_ckpts[-1]
    final = os.path.join(ckpt_dir, 'ckpt_final.pt')
    return final if os.path.isfile(final) else None


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: Config):
    rank, local_rank, world_size = setup_distributed()
    if torch.cuda.is_available() and world_size > 1:
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device(get_device())
    if is_main_process():
        print(f"Distributed: rank={rank}/{world_size}  device={device}", flush=True)

    # ---- Data ----
    train_dataset = BathymetryPatchDataset(
        os.path.join(cfg.patches_dir, 'train'), augment=True)
    val_dataset = BathymetryPatchDataset(
        os.path.join(cfg.patches_dir, 'val'),   augment=False)

    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size,
                                           rank=rank, shuffle=True, drop_last=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size,
                                         rank=rank, shuffle=False)
        train_shuffle = False
    else:
        train_sampler = val_sampler = None
        train_shuffle = True

    # persistent_workers / prefetch_factor are only valid when num_workers > 0
    loader_kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    if cfg.num_workers > 0:
        loader_kwargs['persistent_workers'] = cfg.persistent_workers
        loader_kwargs['prefetch_factor'] = cfg.prefetch_factor

    train_loader = DataLoader(train_dataset, sampler=train_sampler,
                              shuffle=train_shuffle, drop_last=True,
                              **loader_kwargs)
    val_loader = DataLoader(val_dataset, sampler=val_sampler,
                            shuffle=False, **loader_kwargs)

    if is_main_process():
        print(f"Train: {len(train_dataset)} patches | "
              f"Val: {len(val_dataset)} patches", flush=True)

    # ---- Model (built unwrapped so checkpoint I/O is clean) ----
    unet = BathyUNet(
        base_ch=cfg.base_ch,
        ch_mult=cfg.ch_mult,
        n_heads=cfg.n_heads,
        transformer_levels=cfg.transformer_levels,
        transformer_depth=cfg.transformer_depth,
        time_dim=cfg.time_dim,
    ).to(device)
    flow = RectifiedFlow(unet, cfg).to(device)

    if is_main_process():
        n = sum(p.numel() for p in flow.parameters() if p.requires_grad)
        print(f"Model parameters: {n:,}  ({n/1e6:.2f} M)", flush=True)

    # EMA shadow is built on the UNWRAPPED flow.
    ema = EMA(flow, decay=cfg.ema_decay)

    criterion = FlowMatchingLoss(cfg)
    optimizer = torch.optim.AdamW(flow.parameters(), lr=cfg.learning_rate,
                                  betas=(0.9, 0.999), weight_decay=1e-4)
    # Cosine schedule across optimizer steps (i.e., post grad-accum reductions)
    steps_per_epoch = max(1, len(train_loader) // max(1, cfg.grad_accum_steps))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.n_epochs * steps_per_epoch, eta_min=1e-6)

    ckpt_dir = os.path.join(cfg.out_dir, 'checkpoints')
    log_dir = os.path.join(cfg.out_dir, 'logs')
    if is_main_process():
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir) if is_main_process() else None

    # ---- Resume BEFORE wrapping ----
    start_epoch, global_step = 0, 0
    latest_ckpt = _find_latest_checkpoint(ckpt_dir)
    if latest_ckpt:
        if is_main_process():
            print(f"Resuming from {latest_ckpt}", flush=True)
        last_epoch, last_step = _load_checkpoint(
            latest_ckpt, flow, ema, optimizer, scheduler, device)
        start_epoch = last_epoch + 1
        global_step = last_step

    # ---- Wrap with torch.compile, then DDP ----
    # Order matters: compile-then-DDP is the documented combination.
    if cfg.compile_model and device.type == 'cuda':
        if is_main_process():
            print("Compiling flow.model (torch.compile mode='default')...",
                  flush=True)
        flow.model = torch.compile(flow.model, mode='default')

    if world_size > 1:
        flow.model = DistributedDataParallel(
            flow.model,
            device_ids=[local_rank] if device.type == 'cuda' else None,
            find_unused_parameters=cfg.ddp_find_unused_params,
        )

    # ---- Display batch for TensorBoard ----
    display_lr_raw = display_hr = None
    if is_main_process():
        display_batch = next(iter(val_loader))
        display_lr_raw = display_batch['lr_raw'][:cfg.n_sample_display].to(device)
        display_hr = display_batch['hr'][:cfg.n_sample_display].to(device)

    # ---- Autocast context (bf16 on CUDA only) ----
    use_autocast = cfg.precision == 'bf16' and device.type == 'cuda'
    if use_autocast:
        def autocast():
            return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    else:
        autocast = contextlib.nullcontext

    # ---- Training loop ----
    done = False  # set when cfg.max_steps is reached mid-epoch
    for epoch in range(start_epoch, cfg.n_epochs):
        if done:
            break
        flow.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        running = {'total': 0.0, 'fm': 0.0, 'spectral': 0.0}
        running_count = 0
        accum_count = 0
        optimizer.zero_grad(set_to_none=True)

        loop = train_loader
        if is_main_process():
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.n_epochs}",
                        leave=False)

        for batch in loop:
            hr = batch['hr'].to(device, non_blocking=True)
            lr_raw = batch['lr_raw'].to(device, non_blocking=True)

            is_final_in_cycle = (accum_count == cfg.grad_accum_steps - 1)
            # DDP: skip all-reduce on intermediate micro-batches
            if world_size > 1 and not is_final_in_cycle:
                ddp_sync = flow.model.no_sync()
            else:
                ddp_sync = contextlib.nullcontext()

            with ddp_sync, autocast():
                output = flow.training_step(hr, lr_raw)
                loss_dict = criterion(output)
                loss = loss_dict['total'] / cfg.grad_accum_steps
            loss.backward()

            # Bookkeeping (use the un-scaled loss for logging)
            running['total']    += loss_dict['total'].item()
            running['fm']       += loss_dict['fm'].item()
            running['spectral'] += loss_dict['spectral'].item()
            running_count += 1

            accum_count += 1
            if not is_final_in_cycle:
                continue

            # ---- Optimizer step on the final micro-batch of the cycle ----
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(flow.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            accum_count = 0

            if global_step >= cfg.ema_warmup_steps:
                ema.update(flow)
            global_step += 1

            # ---- Logging ----
            if global_step % cfg.log_every == 0:
                losses_reduced = {
                    k: all_reduce_mean(
                        torch.tensor(running[k] / running_count, device=device)
                    ).item()
                    for k in ('total', 'fm', 'spectral')
                }
                running = {'total': 0.0, 'fm': 0.0, 'spectral': 0.0}
                running_count = 0
                if is_main_process():
                    lr_now = scheduler.get_last_lr()[0]
                    writer.add_scalar('loss/total',    losses_reduced['total'],    global_step)
                    writer.add_scalar('loss/fm',       losses_reduced['fm'],       global_step)
                    writer.add_scalar('loss/spectral', losses_reduced['spectral'], global_step)
                    writer.add_scalar('lr', lr_now, global_step)
                    print(f"  step {global_step:6d}  loss={losses_reduced['total']:.4f}  "
                          f"fm={losses_reduced['fm']:.4f}  "
                          f"spec={losses_reduced['spectral']:.4f}  "
                          f"lr={lr_now:.2e}", flush=True)

            if is_main_process() and global_step % cfg.sample_every == 0:
                sample_and_log(ema.shadow, display_lr_raw, writer,
                               global_step, cfg.n_sample_display)
                writer.add_images('hr_groundtruth',
                                  _norm_display(display_hr), global_step)

            if global_step % cfg.save_every == 0:
                _save_checkpoint(ckpt_dir, flow, ema, optimizer, scheduler,
                                 epoch, global_step)

            if cfg.max_steps > 0 and global_step >= cfg.max_steps:
                done = True
                if is_main_process():
                    print(f"Reached max_steps={cfg.max_steps}; stopping early.",
                          flush=True)
                break

        if done:
            break

        # ---- Validation ----
        val_loss = _evaluate(flow, criterion, val_loader, device, autocast)
        val_loss = all_reduce_mean(
            torch.tensor(val_loss, device=device)
        ).item()
        if is_main_process():
            writer.add_scalar('loss/val', val_loss, global_step)
            mean_train = (running['total'] / max(1, running_count)) if running_count else float('nan')
            print(f"Epoch {epoch+1}: val_loss={val_loss:.4f}  "
                  f"(last-window train_loss={mean_train:.4f})", flush=True)

    _save_checkpoint(ckpt_dir, flow, ema, optimizer, scheduler,
                     cfg.n_epochs - 1, global_step, tag='final')
    if writer is not None:
        writer.close()
    if is_main_process():
        print("Training complete.", flush=True)
    cleanup_distributed()


def _evaluate(flow, criterion, loader, device, autocast):
    flow.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            hr = batch['hr'].to(device, non_blocking=True)
            lr_raw = batch['lr_raw'].to(device, non_blocking=True)
            with autocast():
                output = flow.training_step(hr, lr_raw)
                loss_dict = criterion(output)
            total += loss_dict['total'].item()
            n += 1
    flow.train()
    return total / max(n, 1)


def _norm_display(t):
    t = t - t.amin(dim=(1, 2, 3), keepdim=True)
    return t / (t.amax(dim=(1, 2, 3), keepdim=True) + 1e-8)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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
                        help='DataLoader worker count (0 for MPS / debugging)')
    parser.add_argument('--precision',   choices=['bf16', 'fp32'], default=None)
    parser.add_argument('--compile',     dest='compile_model',
                        action=argparse.BooleanOptionalAction, default=None,
                        help='Toggle torch.compile (default from Config)')
    parser.add_argument('--grad_accum_steps', type=int, default=None)
    parser.add_argument('--max_steps',   type=int, default=None,
                        help='Stop after this many optimizer steps (0 = unlimited)')
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
    if args.precision:   cfg.precision = args.precision
    if args.compile_model is not None: cfg.compile_model = args.compile_model
    if args.grad_accum_steps is not None: cfg.grad_accum_steps = args.grad_accum_steps
    if args.max_steps is not None: cfg.max_steps = args.max_steps

    train(cfg)


if __name__ == '__main__':
    main()
