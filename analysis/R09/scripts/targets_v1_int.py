"""targets_v1_int — the interior-pixel fallback summary of the FROZEN F_hr field (directive R09 §A1). targets_v1.py and
F_hr are unchanged; this module adds one summary per tile:
  interior px  = joint-valid px whose Euclidean distance to the nearest invalid px (mosaic padded with one invalid px,
                 scipy.ndimage.distance_transform_edt on the joint-valid mask, as in R08b) is >= D px;
  D1           = ceil(lr_native / hr_res) px (the band's longest wavelength); the stress distance D125 = ceil(125 / hr_res);
  A_fine_int   = RMS of F_hr over the tile's interior px, defined only if the tile has >= MIN_INT_PX = 2048 interior px
                 (1/8 of a 128 x 128 tile); NaN otherwise.
The interior-TILE rule of R08b (>= 95 % of joint-valid px at distance >= D) is also provided for the stress test."""
from __future__ import annotations
import math
import numpy as np
from scipy.ndimage import distance_transform_edt

TILE = 128; MIN_INT_PX = 2048; INTERIOR_TILE_FRAC = 0.95

def d_px(cell_m: float, hr_res: float) -> int:
    """Interior distance in HR px for a cell size (lr_native for D1, 125 m for the stress test)."""
    return int(math.ceil(cell_m / hr_res))

def distance_map(joint: np.ndarray) -> np.ndarray:
    """Distance (HR px) from each px to the nearest invalid px; px outside the mosaic count as invalid."""
    Jp = np.pad(joint.astype(bool), 1, constant_values=False); return distance_transform_edt(Jp)[1:-1, 1:-1]

def interior_px(joint_tile: np.ndarray, dist_tile: np.ndarray, D: int) -> np.ndarray:
    return joint_tile.astype(bool) & (dist_tile >= D)

def afine_int(F_hr_tile: np.ndarray, joint_tile: np.ndarray, dist_tile: np.ndarray, D: int, min_px: int = MIN_INT_PX):
    """(A_fine_int, n_interior_px): RMS of F_hr over the interior px; NaN when fewer than min_px interior px."""
    m = interior_px(joint_tile, dist_tile, D) & np.isfinite(F_hr_tile); n = int(m.sum())
    if n < min_px: return float("nan"), n
    return float(np.sqrt(np.mean(F_hr_tile[m].astype(float) ** 2))), n

def interior_tile(joint_tile: np.ndarray, dist_tile: np.ndarray, D: int, frac: float = INTERIOR_TILE_FRAC):
    """R08b rule: (is_interior, fraction of joint-valid px at distance >= D)."""
    j = joint_tile.astype(bool); n = int(j.sum())
    if n == 0: return False, 0.0
    f = float((dist_tile[j] >= D).mean()); return bool(f >= frac), f
