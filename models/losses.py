"""
Loss functions for rectified-flow training.

Primary: mean-squared error between the predicted velocity and the
analytic target velocity (z - x0) of the linear interpolant.

Auxiliary: a radially-binned 2D log-power spectrum loss on the
reconstructed x0_pred. The radial binning collapses orientation, making
the loss a function of spatial wavelength only — which is what we want
for "preserve seafloor texture / roughness" supervision.

When SNR weighting is enabled, the spectral loss is scaled per-sample by
(1 - t)² so that it dominates where x0_pred is most reliable (low t,
close to clean signal) and contributes little where x0_pred is dominated
by noise (high t).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RadialSpectralLoss(nn.Module):
    """2D FFT log-power loss, optionally radially binned.

    Args:
        mode:    "radial" (n_bins L1 over radially averaged log power)
                 or "2d" (per-bin L1 over the full log power map).
        n_bins:  Number of radial bands (only used when mode == "radial").
    """

    def __init__(self, mode: str = "radial", n_bins: int = 32):
        super().__init__()
        assert mode in ("radial", "2d"), mode
        self.mode = mode
        self.n_bins = n_bins
        # Cache the bin-index map per (H, W, device, dtype) so we don't
        # recompute it every forward.
        self._cache: dict = {}

    def _bin_indices(self, H: int, W: int, device, dtype):
        key = (H, W, str(device), dtype)
        if key in self._cache:
            return self._cache[key]
        # rfft2 output spans frequency dimensions of shape (H, W//2 + 1)
        ky = torch.fft.fftfreq(H, device=device).to(dtype)
        kx = torch.fft.rfftfreq(W, device=device).to(dtype)
        ky_grid, kx_grid = torch.meshgrid(ky, kx, indexing='ij')
        kr = torch.sqrt(ky_grid ** 2 + kx_grid ** 2)
        kr_max = kr.max()
        bin_edges = torch.linspace(0.0, float(kr_max) + 1e-6,
                                   self.n_bins + 1, device=device, dtype=dtype)
        bin_idx = (torch.bucketize(kr, bin_edges) - 1).clamp(0, self.n_bins - 1)
        bin_idx_flat = bin_idx.view(-1).long()
        # Per-band count (same for every sample)
        counts = torch.zeros(self.n_bins, device=device, dtype=dtype)
        counts.scatter_add_(0, bin_idx_flat, torch.ones_like(bin_idx_flat, dtype=dtype))
        counts = counts.clamp(min=1.0)
        self._cache[key] = (bin_idx_flat, counts)
        return self._cache[key]

    def forward(self, x_pred: torch.Tensor, x_true: torch.Tensor,
                reduction: str = 'mean') -> torch.Tensor:
        """
        Args:
            x_pred, x_true: (B, C, H, W) — typically C = 1.
            reduction: 'mean', 'sum', or 'none' (per-sample loss).

        Returns:
            scalar if reduction in ('mean', 'sum'); (B,) if 'none'.
        """
        B, C, H, W = x_pred.shape
        fft_pred = torch.fft.rfft2(x_pred)
        fft_true = torch.fft.rfft2(x_true)
        power_pred = fft_pred.abs() ** 2  # (B, C, H, W//2+1)
        power_true = fft_true.abs() ** 2

        if self.mode == "2d":
            per_sample = F.l1_loss(
                torch.log(power_pred + 1e-8),
                torch.log(power_true + 1e-8),
                reduction='none',
            ).mean(dim=[1, 2, 3])  # (B,)
        else:
            bin_idx_flat, counts = self._bin_indices(
                H, W, x_pred.device, power_pred.dtype
            )
            power_pred_flat = power_pred.reshape(B, C, -1)
            power_true_flat = power_true.reshape(B, C, -1)
            idx_expanded = bin_idx_flat.unsqueeze(0).unsqueeze(0).expand(B, C, -1)

            radial_pred = torch.zeros(B, C, self.n_bins,
                                      device=x_pred.device, dtype=power_pred.dtype)
            radial_true = torch.zeros_like(radial_pred)
            radial_pred.scatter_add_(2, idx_expanded, power_pred_flat)
            radial_true.scatter_add_(2, idx_expanded, power_true_flat)
            radial_pred = radial_pred / counts
            radial_true = radial_true / counts

            per_sample = F.l1_loss(
                torch.log(radial_pred + 1e-8),
                torch.log(radial_true + 1e-8),
                reduction='none',
            ).mean(dim=[1, 2])  # (B,)

        if reduction == 'mean':
            return per_sample.mean()
        if reduction == 'sum':
            return per_sample.sum()
        return per_sample  # 'none'


class FlowMatchingLoss(nn.Module):
    """Primary flow-matching MSE + optional radial spectral loss on x0_pred."""

    def __init__(self, cfg):
        super().__init__()
        self.spectral_weight = float(cfg.spectral_loss_weight)
        self.spectral_mode = cfg.spectral_loss_mode
        self.spectral_snr_weighting = bool(cfg.spectral_loss_snr_weighting)
        if self.spectral_weight > 0.0:
            self.spectral = RadialSpectralLoss(mode=self.spectral_mode)
        else:
            self.spectral = None

    def forward(self, output: dict) -> dict:
        v_pred   = output['v_pred']
        v_target = output['v_target']
        # Primary FM loss: mean-reduced MSE.
        l_fm = F.mse_loss(v_pred, v_target)

        if self.spectral is None:
            return {'total': l_fm, 'fm': l_fm,
                    'spectral': torch.zeros((), device=l_fm.device, dtype=l_fm.dtype)}

        # Auxiliary spectral loss on the reconstructed clean signal.
        x0_pred = output['x0_pred']
        x0_true = output['x0_true']

        if self.spectral_snr_weighting:
            per_sample = self.spectral(x0_pred, x0_true, reduction='none')  # (B,)
            w = (1 - output['t']) ** 2  # (B,) — high weight where t≈0
            # Normalize by sum-of-weights so the scalar isn't artificially
            # small when most samples have high t.
            w_sum = w.sum().clamp(min=1e-6)
            l_spec = (per_sample * w).sum() / w_sum
        else:
            l_spec = self.spectral(x0_pred, x0_true, reduction='mean')

        total = l_fm + self.spectral_weight * l_spec
        return {'total': total, 'fm': l_fm, 'spectral': l_spec}
