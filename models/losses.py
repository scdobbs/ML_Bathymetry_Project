"""
Loss functions for training the diffusion model.

Primary: L1 loss on predicted noise (more robust than L2 for elevation outliers).
Auxiliary: 2D spectral (power spectrum) loss to penalize mismatch in spatial
           frequency content — directly targets the "blurry output" failure mode.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisePredictionLoss(nn.Module):
    """
    L1 loss between predicted and actual noise (epsilon-parameterization).
    This is the standard DDPM training objective.
    L1 is preferred over L2 here because elevation data can have sharp
    local features (fault scarps, channel walls) where L2 would over-penalize.
    """

    def __init__(self):
        super().__init__()

    def forward(self, predicted_noise, target_noise):
        return F.l1_loss(predicted_noise, target_noise)


class SpectralLoss(nn.Module):
    """
    2D power spectrum loss.

    Computes the 2D FFT of both predicted and target, then penalizes
    mismatch in the log power spectrum. This forces the model to reproduce
    the correct spatial frequency content (roughness, texture wavelengths)
    rather than just minimizing pixel-wise error (which causes blurring).

    Applied to x0_pred (predicted clean HR), not to the noise.
    """

    def __init__(self, weight=0.1):
        super().__init__()
        self.weight = weight

    def forward(self, x0_pred, x0_true):
        # Compute 2D FFT
        fft_pred = torch.fft.rfft2(x0_pred)
        fft_true = torch.fft.rfft2(x0_true)

        # Power spectrum (magnitude squared)
        power_pred = fft_pred.abs().pow(2) + 1e-8
        power_true = fft_true.abs().pow(2) + 1e-8

        # Loss on log power spectrum (log scale captures range of frequencies)
        loss = F.l1_loss(power_pred.log(), power_true.log())
        return self.weight * loss


class CombinedDiffusionLoss(nn.Module):
    """
    Combined loss: L1 noise prediction + spectral loss on predicted x0.

    The spectral loss requires estimating x0 from the noise prediction,
    which uses the diffusion model's noise schedule.

    Args:
        spectral_weight: Relative weight of spectral loss (default: 0.1)
    """

    def __init__(self, spectral_weight=0.1):
        super().__init__()
        self.noise_loss = NoisePredictionLoss()
        self.spectral_loss = SpectralLoss(weight=spectral_weight)

    def forward(self, predicted_noise, target_noise, x0_true,
                sqrt_alphas_cumprod_t, sqrt_one_minus_alphas_cumprod_t, x_t):
        """
        Args:
            predicted_noise:               (B, 1, H, W)
            target_noise:                  (B, 1, H, W)
            x0_true:                       (B, 1, H, W) — clean HR
            sqrt_alphas_cumprod_t:         (B, 1, 1, 1) — schedule values at timestep t
            sqrt_one_minus_alphas_cumprod_t: (B, 1, 1, 1)
            x_t:                           (B, 1, H, W) — noisy input

        Returns:
            total_loss, noise_loss_val, spectral_loss_val
        """
        l_noise = self.noise_loss(predicted_noise, target_noise)

        # Reconstruct predicted x0 from noise prediction:
        # x_t = sqrt(ᾱ_t)*x0 + sqrt(1-ᾱ_t)*eps  =>  x0 = (x_t - sqrt(1-ᾱ_t)*eps) / sqrt(ᾱ_t)
        x0_pred = (x_t - sqrt_one_minus_alphas_cumprod_t * predicted_noise) / \
                  sqrt_alphas_cumprod_t
        x0_pred = x0_pred.clamp(-4, 4)

        l_spectral = self.spectral_loss(x0_pred, x0_true)

        return l_noise + l_spectral, l_noise, l_spectral
