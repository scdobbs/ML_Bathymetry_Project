"""
Quick benchmark: run a few training iterations at different batch sizes
and report throughput, GPU memory usage, and estimated epoch/total time.

Usage:
    python3 training/benchmark.py --patches_dir /path/to/patches
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet import BathyUNet
from models.diffusion import GaussianDiffusion
from models.losses import CombinedDiffusionLoss
from preprocessing.data_utils import BathymetryPatchDataset
from torch.utils.data import DataLoader


def benchmark_batch_size(batch_size, train_dataset, device, n_iters=20):
    """Run n_iters forward+backward passes and return stats."""
    loader = DataLoader(train_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=4, pin_memory=True,
                        drop_last=True)

    unet = BathyUNet(in_ch=2, out_ch=1, base_ch=64, time_dim=256, dropout=0.1).to(device)
    diffusion = GaussianDiffusion(unet, n_timesteps=1000).to(device)
    criterion = CombinedDiffusionLoss(spectral_weight=0.1)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=2e-4)

    n_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)

    unet.train()
    torch.cuda.reset_peak_memory_stats(device)

    # Warmup (2 iterations)
    data_iter = iter(loader)
    for _ in range(2):
        batch = next(data_iter)
        lr = batch['lr'].to(device)
        hr = batch['hr'].to(device)
        t = torch.randint(0, 1000, (hr.shape[0],), device=device)
        pred_noise, target_noise = diffusion.p_losses(hr, lr, t)
        sac_t = diffusion.sqrt_alphas_cumprod[t][:, None, None, None]
        soac_t = diffusion.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        x_t = sac_t * hr + soac_t * target_noise
        loss, _, _ = criterion(pred_noise, target_noise, hr, sac_t, soac_t, x_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()

    # Timed iterations
    start = time.time()
    for i in range(n_iters):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        lr = batch['lr'].to(device)
        hr = batch['hr'].to(device)
        t = torch.randint(0, 1000, (hr.shape[0],), device=device)
        pred_noise, target_noise = diffusion.p_losses(hr, lr, t)
        sac_t = diffusion.sqrt_alphas_cumprod[t][:, None, None, None]
        soac_t = diffusion.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        x_t = sac_t * hr + soac_t * target_noise
        loss, _, _ = criterion(pred_noise, target_noise, hr, sac_t, soac_t, x_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    elapsed = time.time() - start

    peak_mem_gb = torch.cuda.max_memory_allocated(device) / 1e9
    samples_per_sec = (n_iters * batch_size) / elapsed
    sec_per_iter = elapsed / n_iters

    n_train = len(train_dataset)
    iters_per_epoch = n_train // batch_size
    sec_per_epoch = sec_per_iter * iters_per_epoch
    hours_500_epochs = (sec_per_epoch * 500) / 3600

    # Cleanup
    del unet, diffusion, criterion, optimizer
    torch.cuda.empty_cache()

    return {
        'batch_size': batch_size,
        'n_params': n_params,
        'peak_mem_gb': peak_mem_gb,
        'samples_per_sec': samples_per_sec,
        'sec_per_iter': sec_per_iter,
        'iters_per_epoch': iters_per_epoch,
        'sec_per_epoch': sec_per_epoch,
        'hours_500_epochs': hours_500_epochs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patches_dir', required=True)
    parser.add_argument('--batch_sizes', type=str, default='8,16,32,64',
                        help='Comma-separated batch sizes to test')
    parser.add_argument('--n_iters', type=int, default=20,
                        help='Number of timed iterations per batch size')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    train_dataset = BathymetryPatchDataset(
        os.path.join(args.patches_dir, 'train'), augment=True)
    print(f"Training patches: {len(train_dataset)}")

    batch_sizes = [int(b) for b in args.batch_sizes.split(',')]
    results = []

    print(f"\nRunning {args.n_iters} timed iterations per batch size...\n")
    print(f"{'BS':>4} | {'Peak GPU':>9} | {'Samples/s':>10} | {'s/iter':>7} | "
          f"{'Iters/ep':>9} | {'Min/epoch':>10} | {'Hr/500ep':>9}")
    print('-' * 80)

    for bs in batch_sizes:
        try:
            stats = benchmark_batch_size(bs, train_dataset, device, args.n_iters)
            results.append(stats)
            print(f"{bs:>4} | {stats['peak_mem_gb']:>7.2f} GB | "
                  f"{stats['samples_per_sec']:>10.1f} | {stats['sec_per_iter']:>7.3f} | "
                  f"{stats['iters_per_epoch']:>9d} | {stats['sec_per_epoch']/60:>8.1f} m | "
                  f"{stats['hours_500_epochs']:>7.1f} h")
        except RuntimeError as e:
            if 'out of memory' in str(e):
                torch.cuda.empty_cache()
                print(f"{bs:>4} |    OOM    |        -- |      -- |        -- |         -- |        --")
            else:
                raise

    print(f"\nModel parameters: {results[0]['n_params']:,}")
    print("\nRecommendation: pick the largest batch size that fits in GPU memory")
    print("and gives good throughput (samples/s). Larger batch = fewer iterations")
    print("per epoch but each iteration processes more data.")


if __name__ == '__main__':
    main()
