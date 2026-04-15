"""
Normalization, augmentation, and PyTorch Dataset utilities for bathymetric patches.
PyTorch is only required for BathymetryPatchDataset; all other functions work without it.
"""

import numpy as np
import glob
import os
import random

try:
    import torch
    from torch.utils.data import Dataset as _TorchDataset
    _TORCH_AVAILABLE = True
except ImportError:
    _TorchDataset = object
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Numpy-only utilities (no PyTorch required)
# ---------------------------------------------------------------------------

def normalize_patch(arr):
    """
    Per-patch z-score normalization.
    Returns normalized array, mean, and std.
    Operates on float32 elevation arrays.
    """
    mean = float(np.nanmean(arr))
    std = float(np.nanstd(arr))
    if std < 1e-6:
        std = 1.0  # flat patch — prevent divide-by-zero
    return (arr - mean) / std, mean, std


def denormalize_patch(arr, mean, std):
    """Invert normalize_patch to recover elevation in meters."""
    return arr * std + mean


def bicubic_upsample(lr_arr, target_size):
    """
    Bicubic upsample a 2D numpy array to (target_size, target_size).
    Used to create the conditioning input for the diffusion model.
    """
    from scipy.ndimage import zoom
    factor = target_size / lr_arr.shape[0]
    return zoom(lr_arr, factor, order=3).astype(np.float32)


def augment_pair(hr, lr_up):
    """
    Apply random horizontal/vertical flips and 90-degree rotations
    to a paired (HR, LR-upsampled) patch.
    Both arrays must be 2D (H, W).
    Returns augmented (hr, lr_up).
    """
    if random.random() > 0.5:
        hr = np.fliplr(hr).copy()
        lr_up = np.fliplr(lr_up).copy()
    if random.random() > 0.5:
        hr = np.flipud(hr).copy()
        lr_up = np.flipud(lr_up).copy()
    k = random.randint(0, 3)
    if k > 0:
        hr = np.rot90(hr, k).copy()
        lr_up = np.rot90(lr_up, k).copy()
    return hr, lr_up


# ---------------------------------------------------------------------------
# PyTorch Dataset (requires torch)
# ---------------------------------------------------------------------------

class BathymetryPatchDataset(_TorchDataset):
    """
    Loads paired (LR, HR) bathymetry patches from .npz files.
    Requires PyTorch.

    Each .npz contains:
        'lr':    float32 array (1, H, W) — bicubic-upsampled LR patch, normalized
        'hr':    float32 array (1, H, W) — HR patch, normalized
        'stats': float32 array (2,) — [mean, std] used for normalization

    Args:
        patches_dir: directory containing .npz patch files
        augment: if True, apply random flips/rotations during training
    """

    def __init__(self, patches_dir, augment=False):
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for BathymetryPatchDataset")
        self.files = sorted(glob.glob(os.path.join(patches_dir, '*.npz')))
        if len(self.files) == 0:
            raise FileNotFoundError(f"No .npz patches found in {patches_dir}")
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        lr = data['lr'].squeeze()   # (H, W)
        hr = data['hr'].squeeze()   # (H, W)
        stats = data['stats']       # [mean, std]

        if self.augment:
            hr, lr = augment_pair(hr, lr)

        lr    = torch.from_numpy(lr[np.newaxis]).float()
        hr    = torch.from_numpy(hr[np.newaxis]).float()
        stats = torch.from_numpy(stats).float()

        return {'lr': lr, 'hr': hr, 'stats': stats}
