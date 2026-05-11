"""
Validate and visualize extracted patches to confirm alignment quality
and dataset statistics before training.

Usage:
    python -m preprocessing.validate_dataset \
        --patches_dir /path/to/patches/ \
        --n_samples 8 \
        --out_dir results/validation/
"""

import os
import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt


def load_random_patches(split_dir, n):
    """Load n random patches from a split directory."""
    files = glob.glob(os.path.join(split_dir, '*.npz'))
    if len(files) == 0:
        return []
    indices = np.random.choice(len(files), min(n, len(files)), replace=False)
    patches = []
    for i in indices:
        data = np.load(files[i])
        patches.append({
            'lr_raw': data['lr_raw'].squeeze(),
            'lr_up':  data['lr_up'].squeeze(),
            'hr':     data['hr'].squeeze(),
            'stats':  data['stats'],
            'name':   os.path.basename(files[i]),
        })
    return patches


def plot_patch_grid(patches, title, out_path):
    """Plot a grid of (lr_raw | lr_up | hr | residual | histogram) per patch."""
    n = len(patches)
    cols = 5
    rows = n

    fig = plt.figure(figsize=(17, 3.2 * rows))
    fig.suptitle(title, fontsize=14, y=1.005)

    for i, p in enumerate(patches):
        lr_raw = p['lr_raw']
        lr_up  = p['lr_up']
        hr     = p['hr']
        diff   = hr - lr_up

        vmin = min(lr_up.min(), hr.min())
        vmax = max(lr_up.max(), hr.max())

        ax_lr_raw = fig.add_subplot(rows, cols, i * cols + 1)
        ax_lr_up  = fig.add_subplot(rows, cols, i * cols + 2)
        ax_hr     = fig.add_subplot(rows, cols, i * cols + 3)
        ax_diff   = fig.add_subplot(rows, cols, i * cols + 4)
        ax_hist   = fig.add_subplot(rows, cols, i * cols + 5)

        ax_lr_raw.imshow(lr_raw, cmap='terrain', vmin=vmin, vmax=vmax,
                         interpolation='nearest')
        ax_lr_raw.set_title(f'lr_raw {lr_raw.shape[0]}×{lr_raw.shape[1]}', fontsize=8)
        ax_lr_raw.axis('off')

        ax_lr_up.imshow(lr_up, cmap='terrain', vmin=vmin, vmax=vmax)
        ax_lr_up.set_title(f'lr_up {lr_up.shape[0]}×{lr_up.shape[1]}', fontsize=8)
        ax_lr_up.axis('off')

        ax_hr.imshow(hr, cmap='terrain', vmin=vmin, vmax=vmax)
        ax_hr.set_title(f'hr {hr.shape[0]}×{hr.shape[1]}', fontsize=8)
        ax_hr.axis('off')

        dmax = np.abs(diff).max()
        ax_diff.imshow(diff, cmap='RdBu_r', vmin=-dmax, vmax=dmax)
        ax_diff.set_title('hr − lr_up', fontsize=8)
        ax_diff.axis('off')

        ax_hist.hist(hr.ravel(),     bins=50, alpha=0.55, label='hr',     color='steelblue')
        ax_hist.hist(lr_up.ravel(),  bins=50, alpha=0.55, label='lr_up',  color='orange')
        ax_hist.hist(lr_raw.ravel(), bins=50, alpha=0.55, label='lr_raw', color='green')
        ax_hist.set_title(f"μ={p['stats'][0]:.1f}m σ={p['stats'][1]:.2f}m", fontsize=8)
        ax_hist.legend(fontsize=7)
        ax_hist.set_xlabel('Normalized elevation', fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def print_split_stats(patches_dir, split):
    """Print statistics for a split."""
    split_dir = os.path.join(patches_dir, split)
    files = glob.glob(os.path.join(split_dir, '*.npz'))
    if not files:
        print(f"  {split}: 0 patches (directory not found or empty)")
        return

    means, stds, hr_ranges = [], [], []
    lr_raw_shape = lr_up_shape = hr_shape = None
    for f in files:
        data = np.load(f)
        means.append(float(data['stats'][0]))
        stds.append(float(data['stats'][1]))
        hr = data['hr'].squeeze()
        hr_ranges.append(float(hr.max() - hr.min()))
        if lr_raw_shape is None:
            lr_raw_shape = data['lr_raw'].squeeze().shape
            lr_up_shape  = data['lr_up'].squeeze().shape
            hr_shape     = hr.shape

    print(f"  {split}: {len(files)} patches")
    print(f"    Shapes: lr_raw={lr_raw_shape}  lr_up={lr_up_shape}  hr={hr_shape}")
    print(f"    Elevation mean: {np.mean(means):.1f} ± {np.std(means):.1f} m")
    print(f"    Elevation std:  {np.mean(stds):.2f} ± {np.std(stds):.2f} m")
    print(f"    HR norm range:  {np.mean(hr_ranges):.2f} ± {np.std(hr_ranges):.2f} (normalized)")


def main():
    parser = argparse.ArgumentParser(description='Validate bathymetry patch dataset')
    parser.add_argument('--patches_dir', required=True,
                        help='Root directory containing train/val/test subdirectories')
    parser.add_argument('--n_samples', type=int, default=8,
                        help='Number of sample patches to visualize per split')
    parser.add_argument('--out_dir', default='results/validation',
                        help='Directory to save validation figures')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Dataset statistics:")
    for split in ['train', 'val', 'test']:
        print_split_stats(args.patches_dir, split)

    print("\nGenerating sample visualizations...")
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(args.patches_dir, split)
        patches = load_random_patches(split_dir, args.n_samples)
        if not patches:
            continue
        out_path = os.path.join(args.out_dir, f'{split}_samples.png')
        plot_patch_grid(patches, f'{split.capitalize()} patches (random sample)',
                        out_path)


if __name__ == '__main__':
    main()
