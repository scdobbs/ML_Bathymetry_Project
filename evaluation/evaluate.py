"""
Evaluate the trained diffusion model on held-out test surveys.

For each test patch:
    - Generate N=50 realizations using DDIM
    - Compare mean realization to ground truth HR
    - Compute deterministic metrics: PSNR, SSIM, MAE, RMSE
    - Compute probabilistic metrics: CRPS, coverage (90%), sharpness
    - Compute geomorphological metrics: slope KS test

Prints a summary table and saves calibration plot.

Usage:
    python evaluation/evaluate.py \
        --checkpoint /scratch/.../ckpt_final.pt \
        --test_patches /scratch/.../bathy_patches/test/ \
        --out_dir results/evaluation/ \
        --n_realizations 50 \
        --ddim_steps 50
"""

import os
import sys
import argparse
import json
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet import BathyUNet
from models.diffusion import GaussianDiffusion
from preprocessing.data_utils import denormalize_patch
from training.config import Config
from evaluation.metrics import (
    psnr, ssim, mae, rmse,
    crps, coverage, sharpness, calibration_data,
    slope_distribution_comparison
)


def load_model(checkpoint_path, cfg, device):
    unet = BathyUNet(in_ch=2, out_ch=1, base_ch=cfg.base_ch,
                     time_dim=cfg.time_dim, dropout=0.0).to(device)
    diffusion = GaussianDiffusion(unet, n_timesteps=cfg.n_timesteps,
                                  beta_start=cfg.beta_start,
                                  beta_end=cfg.beta_end).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    ema_state = state.get('ema', state.get('model'))
    diffusion.load_state_dict(ema_state)
    diffusion.eval()
    return diffusion


def evaluate_patch(diffusion, lr_norm, hr_norm, stats, n_realizations, ddim_steps, device):
    """
    Generate N realizations for one patch and compute all metrics.
    Returns dict of metric values.
    """
    mean_elev, std_elev = float(stats[0]), float(stats[1])

    # Generate realizations
    lr_tensor = torch.from_numpy(lr_norm[np.newaxis, np.newaxis]).float().to(device)
    with torch.no_grad():
        samples = diffusion.sample_ddim(lr_tensor, n_realizations=n_realizations,
                                        n_steps=ddim_steps)
    samples_np = samples.squeeze(1).cpu().numpy()  # (N, H, W) normalized

    # Denormalize all outputs to elevation (meters)
    samples_m = denormalize_patch(samples_np, mean_elev, std_elev)  # (N, H, W)
    hr_m      = denormalize_patch(hr_norm,    mean_elev, std_elev)  # (H, W)
    mean_pred = samples_m.mean(axis=0)

    metrics = {
        'psnr':        psnr(mean_pred, hr_m),
        'ssim':        ssim(mean_pred, hr_m),
        'mae':         mae(mean_pred, hr_m),
        'rmse':        rmse(mean_pred, hr_m),
        'crps':        crps(samples_m, hr_m),
        'coverage_90': coverage(samples_m, hr_m, alpha=0.90),
        'sharpness_80': sharpness(samples_m, lo_pct=10, hi_pct=90),
    }

    # Slope comparison
    slope_stats = slope_distribution_comparison(mean_pred, hr_m, res=2.0)
    metrics['slope_ks'] = slope_stats['ks_statistic']

    return metrics, samples_m, hr_m


def main():
    parser = argparse.ArgumentParser(description='Evaluate diffusion SR model on test patches')
    parser.add_argument('--checkpoint',     required=True)
    parser.add_argument('--test_patches',   required=True,
                        help='Directory of test .npz patch files')
    parser.add_argument('--out_dir',        default='results/evaluation')
    parser.add_argument('--n_realizations', type=int, default=50)
    parser.add_argument('--ddim_steps',     type=int, default=50)
    parser.add_argument('--max_patches',    type=int, default=None,
                        help='Limit number of patches (for fast smoke test)')
    parser.add_argument('--base_ch',        type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Device: {device}")

    cfg = Config()
    cfg.base_ch = args.base_ch
    diffusion = load_model(args.checkpoint, cfg, device)

    # Load test patches
    patch_files = sorted(glob.glob(os.path.join(args.test_patches, '*.npz')))
    if args.max_patches:
        patch_files = patch_files[:args.max_patches]
    print(f"Evaluating on {len(patch_files)} test patches")

    all_metrics = []
    all_samples_for_calib = []  # for calibration curve
    all_targets_for_calib = []

    for pf in tqdm(patch_files, desc="Evaluating"):
        data = np.load(pf)
        lr_norm = data['lr'].squeeze()
        hr_norm = data['hr'].squeeze()
        stats   = data['stats']

        m, samples_m, hr_m = evaluate_patch(
            diffusion, lr_norm, hr_norm, stats,
            args.n_realizations, args.ddim_steps, device)
        all_metrics.append(m)

        # Store a subsample for calibration curve
        if len(all_samples_for_calib) < 20:
            all_samples_for_calib.append(samples_m)
            all_targets_for_calib.append(hr_m)

    # Aggregate results
    metric_keys = list(all_metrics[0].keys())
    summary = {}
    for k in metric_keys:
        vals = [m[k] for m in all_metrics if np.isfinite(m[k])]
        summary[k] = {
            'mean': float(np.mean(vals)),
            'std':  float(np.std(vals)),
            'median': float(np.median(vals)),
        }

    # Print summary table
    print("\n" + "=" * 60)
    print(f"{'Metric':<20}  {'Mean':>10}  {'Median':>10}  {'Std':>10}")
    print("-" * 60)
    for k, v in summary.items():
        print(f"{k:<20}  {v['mean']:>10.4f}  {v['median']:>10.4f}  {v['std']:>10.4f}")
    print("=" * 60)
    print(f"\nNote: Coverage 90% = {summary['coverage_90']['mean']:.3f} "
          f"(ideal: 0.900)")

    # Save JSON results
    results_path = os.path.join(args.out_dir, 'metrics.json')
    with open(results_path, 'w') as f:
        json.dump({'summary': summary, 'per_patch': all_metrics}, f, indent=2)
    print(f"\nSaved metrics: {results_path}")

    # Calibration plot
    if all_samples_for_calib:
        _plot_calibration(all_samples_for_calib, all_targets_for_calib, args.out_dir)


def _plot_calibration(all_samples, all_targets, out_dir):
    """Plot reliability diagram (nominal vs empirical coverage)."""
    # Stack all patches
    samples_cat = np.concatenate([s.reshape(s.shape[0], -1) for s in all_samples], axis=1)
    targets_cat = np.concatenate([t.ravel() for t in all_targets])

    nominal, empirical = calibration_data(samples_cat, targets_cat, n_bins=20)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax.plot(nominal, empirical, 'o-', color='steelblue', label='Model')
    ax.fill_between(nominal, nominal - 0.05, nominal + 0.05,
                    alpha=0.15, color='gray', label='±5% band')
    ax.set_xlabel('Nominal coverage')
    ax.set_ylabel('Empirical coverage')
    ax.set_title('Calibration plot (reliability diagram)')
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    out_path = os.path.join(out_dir, 'calibration.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved calibration plot: {out_path}")


if __name__ == '__main__':
    main()
