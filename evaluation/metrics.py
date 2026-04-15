"""
Evaluation metrics for probabilistic super-resolution of bathymetry.

Deterministic (compare mean realization to ground truth):
    - PSNR (Peak Signal-to-Noise Ratio)
    - SSIM (Structural Similarity Index)
    - MAE  (Mean Absolute Error, in meters)
    - RMSE (Root Mean Square Error, in meters)

Probabilistic (evaluate the full predictive distribution):
    - CRPS (Continuous Ranked Probability Score) — main uncertainty metric
    - Coverage: fraction of GT values within predicted 90% CI
    - Sharpness: mean width of 80% prediction interval
    - Calibration: reliability diagram data

Geomorphological:
    - Slope error: compare slope distributions
    - Roughness spectrum: 2D power spectrum comparison
"""

import numpy as np
from scipy.signal import fftconvolve


# ---------------------------------------------------------------------------
# Deterministic metrics
# ---------------------------------------------------------------------------

def psnr(pred, target, data_range=None):
    """
    Peak Signal-to-Noise Ratio.
    Higher is better. Typical good values: >30 dB.

    Args:
        pred, target: (H, W) float arrays (elevation in meters)
        data_range:   if None, uses max(target) - min(target)
    Returns:
        PSNR in dB
    """
    if data_range is None:
        data_range = target.max() - target.min()
        if data_range == 0:
            return float('inf')
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(data_range / np.sqrt(mse))


def ssim(pred, target, window_size=11, data_range=None):
    """
    Structural Similarity Index (Wang et al. 2004).
    Range [-1, 1], higher is better.
    """
    if data_range is None:
        data_range = float(target.max() - target.min())

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    kernel = _gaussian_kernel(window_size, sigma=1.5)

    mu_x = fftconvolve(pred,   kernel, mode='valid')
    mu_y = fftconvolve(target, kernel, mode='valid')
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sig_x2 = fftconvolve(pred   * pred,   kernel, mode='valid') - mu_x2
    sig_y2 = fftconvolve(target * target, kernel, mode='valid') - mu_y2
    sig_xy = fftconvolve(pred   * target, kernel, mode='valid') - mu_xy

    numerator   = (2 * mu_xy + C1) * (2 * sig_xy + C2)
    denominator = (mu_x2 + mu_y2 + C1) * (sig_x2 + sig_y2 + C2)
    ssim_map = numerator / (denominator + 1e-10)
    return float(np.mean(ssim_map))


def mae(pred, target):
    """Mean Absolute Error in meters."""
    return float(np.nanmean(np.abs(pred - target)))


def rmse(pred, target):
    """Root Mean Square Error in meters."""
    return float(np.sqrt(np.nanmean((pred - target) ** 2)))


# ---------------------------------------------------------------------------
# Probabilistic metrics
# ---------------------------------------------------------------------------

def crps(samples, target):
    """
    Continuous Ranked Probability Score (Gneiting & Raftery 2007).
    Measures the quality of probabilistic forecasts.
    Lower is better. CRPS = 0 for a perfect point forecast.

    CRPS = E[|X - y|] - 0.5 * E[|X - X'|]
    where X, X' are independent draws from the forecast distribution
    and y is the observation.

    Args:
        samples: (N, H, W) — N realizations
        target:  (H, W)    — ground truth

    Returns:
        Scalar mean CRPS (averaged over all valid pixels)
    """
    N = samples.shape[0]
    # E[|X - y|]
    mae_term = np.nanmean(np.abs(samples - target[np.newaxis]), axis=0)  # (H, W)

    # E[|X - X'|] — approximate with pairwise differences
    # Use all N*(N-1)/2 pairs; for efficiency use a random subset if N > 100
    spread_term = np.zeros_like(mae_term)
    count = 0
    for i in range(N - 1):
        for j in range(i + 1, N):
            spread_term += np.abs(samples[i] - samples[j])
            count += 1
    spread_term /= count  # average pair distance

    crps_map = mae_term - 0.5 * spread_term
    return float(np.nanmean(crps_map))


def coverage(samples, target, alpha=0.90):
    """
    Fraction of ground truth pixels falling within the (1-alpha) prediction interval.
    Well-calibrated model: coverage ≈ alpha.

    Args:
        samples: (N, H, W)
        target:  (H, W)
        alpha:   nominal coverage level (default 0.90)

    Returns:
        Empirical coverage fraction
    """
    lo = (1 - alpha) / 2 * 100
    hi = (1 + alpha) / 2 * 100
    lower = np.nanpercentile(samples, lo, axis=0)   # (H, W)
    upper = np.nanpercentile(samples, hi, axis=0)
    within = (target >= lower) & (target <= upper)
    valid  = np.isfinite(target) & np.isfinite(lower) & np.isfinite(upper)
    return float(within[valid].mean())


def sharpness(samples, lo_pct=10, hi_pct=90):
    """
    Mean width of the (lo_pct, hi_pct) prediction interval.
    Lower is better (tighter predictions), but must be balanced against coverage.

    Args:
        samples: (N, H, W)
    Returns:
        Mean interval width in meters
    """
    lower = np.nanpercentile(samples, lo_pct, axis=0)
    upper = np.nanpercentile(samples, hi_pct, axis=0)
    width = upper - lower
    return float(np.nanmean(width))


def calibration_data(samples, target, n_bins=20):
    """
    Compute reliability diagram data: nominal vs empirical coverage.

    Returns:
        nominal_levels:  array of nominal coverage levels [0.05, 0.1, ..., 0.95]
        empirical_covs:  array of empirical coverage fractions
    """
    levels = np.linspace(0.05, 0.95, n_bins)
    empirical = [coverage(samples, target, alpha=a) for a in levels]
    return levels, np.array(empirical)


# ---------------------------------------------------------------------------
# Geomorphological metrics
# ---------------------------------------------------------------------------

def slope_map(dem, res=2.0):
    """
    Compute slope magnitude (degrees) from a DEM using central differences.

    Args:
        dem: (H, W) elevation array
        res: pixel size in meters
    """
    dy, dx = np.gradient(dem, res)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    return np.degrees(slope_rad)


def roughness_spectrum(dem, res=2.0):
    """
    Compute the radially-averaged 1D power spectrum of a DEM.
    Useful for comparing texture at different spatial scales.

    Returns:
        wavenumbers: (M,) spatial frequencies in cycles/meter
        power:       (M,) spectral power
    """
    h, w = dem.shape
    dem_centered = dem - np.nanmean(dem)
    dem_centered = np.nan_to_num(dem_centered, nan=0.0)

    fft2 = np.fft.fft2(dem_centered)
    power2d = np.abs(np.fft.fftshift(fft2)) ** 2

    # Radial average
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)

    max_r = min(cy, cx)
    power_radial = np.array([power2d[r == ri].mean() for ri in range(max_r)])
    freq = np.fft.fftfreq(max(h, w), d=res)[:max_r]

    return freq, power_radial


def slope_distribution_comparison(pred_dem, target_dem, res=2.0):
    """
    Compare slope distributions between predicted and target DEMs.
    Returns dict with KS statistic and mean/std slope for both.
    """
    from scipy.stats import ks_2samp
    pred_slope = slope_map(pred_dem, res).ravel()
    tgt_slope  = slope_map(target_dem, res).ravel()

    pred_slope = pred_slope[np.isfinite(pred_slope)]
    tgt_slope  = tgt_slope[np.isfinite(tgt_slope)]

    ks_stat, ks_pval = ks_2samp(pred_slope, tgt_slope)
    return {
        'ks_statistic': float(ks_stat),
        'ks_pvalue':    float(ks_pval),
        'pred_slope_mean': float(pred_slope.mean()),
        'pred_slope_std':  float(pred_slope.std()),
        'target_slope_mean': float(tgt_slope.mean()),
        'target_slope_std':  float(tgt_slope.std()),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gaussian_kernel(size, sigma):
    """Create a normalized 2D Gaussian kernel."""
    coords = np.arange(size) - size // 2
    g1d = np.exp(-(coords ** 2) / (2 * sigma ** 2))
    g2d = np.outer(g1d, g1d)
    return g2d / g2d.sum()
