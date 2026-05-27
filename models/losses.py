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
    """Primary flow-matching MSE + optional spectral + optional data-consistency."""

    def __init__(self, cfg):
        super().__init__()
        self.fm_loss_type = str(getattr(cfg, 'fm_loss_type', 'mse'))
        self.fm_huber_beta = float(getattr(cfg, 'fm_loss_huber_beta', 1.0))
        self.spectral_weight = float(cfg.spectral_loss_weight)
        self.spectral_mode = cfg.spectral_loss_mode
        self.spectral_snr_weighting = bool(cfg.spectral_loss_snr_weighting)
        if self.spectral_weight > 0.0:
            self.spectral = RadialSpectralLoss(mode=self.spectral_mode)
        else:
            self.spectral = None

        self.consistency_weight = float(getattr(cfg, 'consistency_loss_weight', 0.0))
        self.consistency_snr_weighting = bool(
            getattr(cfg, 'consistency_loss_snr_weighting', True))
        # Only apply consistency where x0_pred is closer to HR than to noise
        # (i.e. (1−t) > t  ⇒  t < 0.5). Above this, x0_pred can be 10×
        # the magnitude of HR for spatially correlated v_pred errors, and
        # one such sample dominates the batch.
        self.consistency_t_max = float(getattr(cfg, 'consistency_loss_t_max', 0.5))
        # Patches whose original HR std hit normalize_patch's divide-by-zero
        # floor (`std < 1e-6 → std := 1.0`) have nonsense LR normalization
        # (lr_raw is in raw meters, not z-scored). Skip them.
        self.consistency_min_std = float(getattr(cfg, 'consistency_loss_min_std', 1e-3))

    def _spectral_loss(self, output) -> torch.Tensor:
        x0_pred = output['x0_pred']
        x0_true = output['x0_true']
        if self.spectral_snr_weighting:
            per_sample = self.spectral(x0_pred, x0_true, reduction='none')  # (B,)
            w = (1 - output['t']) ** 2  # (B,) — high weight where t≈0
            w_sum = w.sum().clamp(min=1e-6)
            return (per_sample * w).sum() / w_sum
        return self.spectral(x0_pred, x0_true, reduction='mean')

    def _consistency_loss(self, output) -> torch.Tensor:
        """Huber(avg_pool(x0_pred, sr_ratio), lr_raw), gated to reliable t.

        Coarsens the predicted HR back to the LR grid and matches it against
        the raw LR conditioning. Provides a direct gradient on absolute
        elevation that the cross-attention path was failing to enforce.

        Why Huber (smooth-L1) instead of MSE: per-sample MSE averages over
        only sr_ratio² = 256 LR cells (32×32 here), so it's highly
        sensitive to outlier samples — one v_pred overshoot dominates the
        batch mean. fm (mean over ~4M pixels per batch) hides these.
        smooth_l1 with beta=1 is quadratic for |error| < 1 (same gradient
        as MSE in the healthy regime) and linear for |error| ≥ 1, so a
        runaway sample contributes ~|error|·N_cells instead of error²·N_cells.

        Mask layers (all multiplicative on the per-sample weight):
          (1) t-gate: t < consistency_t_max. Above that x0_pred is not yet
              a reliable HR reconstruction.
          (2) SNR weighting (1−t)² within the gate.
          (3) CFG-keep: skip samples whose lr_raw was replaced by the null
              token during CFG dropout.
        """
        x0_pred = output['x0_pred']
        lr_raw = output['lr_raw']
        H, W = x0_pred.shape[-2:]
        h, w = lr_raw.shape[-2:]
        kh = H // h
        kw = W // w
        x0_pred_lr = F.avg_pool2d(x0_pred, kernel_size=(kh, kw))
        per_sample = F.smooth_l1_loss(
            x0_pred_lr, lr_raw, reduction='none', beta=1.0
        ).mean(dim=(1, 2, 3))  # (B,)

        t = output['t']
        w_t = (t < self.consistency_t_max).to(per_sample.dtype)  # (B,)
        if self.consistency_snr_weighting:
            w_t = w_t * (1 - t) ** 2
        keep_mask = output.get('cfg_keep_mask', None)
        if keep_mask is not None:
            w_t = w_t * keep_mask.to(w_t.dtype)
        w_sum = w_t.sum().clamp(min=1e-6)
        return (per_sample * w_t).sum() / w_sum

    def forward(self, output: dict) -> dict:
        v_pred   = output['v_pred']
        v_target = output['v_target']
        # Primary FM loss. Huber (smooth_l1, beta=1) keeps gradients bounded
        # when a small tail of outlier patches has |v_target| >> 1, which
        # otherwise destabilizes training (observed catastrophic divergence
        # around epoch 20 in residual-mode runs with pure MSE).
        if self.fm_loss_type == 'huber':
            l_fm = F.smooth_l1_loss(v_pred, v_target, beta=self.fm_huber_beta)
        else:
            l_fm = F.mse_loss(v_pred, v_target)

        zero = torch.zeros((), device=l_fm.device, dtype=l_fm.dtype)
        l_spec = self._spectral_loss(output) if self.spectral is not None else zero
        l_cons = (self._consistency_loss(output)
                  if self.consistency_weight > 0.0 and 'lr_raw' in output
                  else zero)

        total = l_fm + self.spectral_weight * l_spec + self.consistency_weight * l_cons
        return {'total': total, 'fm': l_fm, 'spectral': l_spec, 'consistency': l_cons}
