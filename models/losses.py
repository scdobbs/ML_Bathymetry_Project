"""
Loss functions for training the diffusion model.

Primary: L1 loss on predicted noise (more robust than L2 for elevation outliers).
Auxiliary losses applied to predicted x0 (denoised image estimate):
  - Spectral loss: 2D power spectrum — penalises wrong spatial frequency content
  - Gradient loss: Sobel edge loss — directly penalises missing sharp features
  - x0 reconstruction loss: direct L1 on predicted vs true clean image
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisePredictionLoss(nn.Module):
    """L1 loss between predicted and actual noise (epsilon-parameterization)."""

    def __init__(self):
        super().__init__()

    def forward(self, predicted_noise, target_noise):
        return F.l1_loss(predicted_noise, target_noise)


class SpectralLoss(nn.Module):
    """
    2D power spectrum loss on log scale.
    Forces the model to reproduce correct spatial frequency content
    (roughness, texture wavelengths) rather than blurring.
    Applied to x0_pred (predicted clean HR).
    """

    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def forward(self, x0_pred, x0_true):
        fft_pred = torch.fft.rfft2(x0_pred)
        fft_true = torch.fft.rfft2(x0_true)
        power_pred = fft_pred.abs().pow(2) + 1e-8
        power_true = fft_true.abs().pow(2) + 1e-8
        loss = F.l1_loss(power_pred.log(), power_true.log())
        return self.weight * loss


class GradientLoss(nn.Module):
    """
    Sobel gradient loss on predicted x0.

    Penalises mismatch in image gradients — directly targets the failure
    mode where the model produces smooth outputs that miss canyon walls,
    fault scarps, and small bedforms present in the HR ground truth.
    """

    def __init__(self, weight=0.5):
        super().__init__()
        self.weight = weight
        # Sobel kernels (fixed, not learned)
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                          dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                          dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('kx', kx)
        self.register_buffer('ky', ky)

    def forward(self, x0_pred, x0_true):
        gx_pred = F.conv2d(x0_pred, self.kx, padding=1)
        gy_pred = F.conv2d(x0_pred, self.ky, padding=1)
        gx_true = F.conv2d(x0_true, self.kx, padding=1)
        gy_true = F.conv2d(x0_true, self.ky, padding=1)
        loss = F.l1_loss(gx_pred, gx_true) + F.l1_loss(gy_pred, gy_true)
        return self.weight * loss


class CombinedDiffusionLoss(nn.Module):
    """
    Combined loss:
        L1 noise prediction
      + spectral loss on predicted x0   (weight: spectral_weight)
      + gradient (Sobel) loss on x0     (weight: gradient_weight)
      + direct x0 reconstruction loss   (weight: x0_weight)

    The three auxiliary terms all operate on the x0 estimate derived from
    the noise prediction, so they don't change the model's parameterisation —
    they just add richer supervision signal about what the clean image should
    look like.
    """

    def __init__(self, spectral_weight=1.0, gradient_weight=0.5, x0_weight=0.5):
        super().__init__()
        self.noise_loss    = NoisePredictionLoss()
        self.spectral_loss = SpectralLoss(weight=spectral_weight)
        self.gradient_loss = GradientLoss(weight=gradient_weight)
        self.x0_weight     = x0_weight

    def forward(self, predicted_noise, target_noise, x0_true,
                sqrt_alphas_cumprod_t, sqrt_one_minus_alphas_cumprod_t, x_t):
        """
        Args:
            predicted_noise:                 (B, 1, H, W)
            target_noise:                    (B, 1, H, W)
            x0_true:                         (B, 1, H, W) — clean HR patch
            sqrt_alphas_cumprod_t:           (B, 1, 1, 1)
            sqrt_one_minus_alphas_cumprod_t: (B, 1, 1, 1)
            x_t:                             (B, 1, H, W) — noisy input

        Returns:
            total_loss, noise_loss_val, spectral_loss_val, gradient_loss_val, x0_loss_val
        """
        l_noise = self.noise_loss(predicted_noise, target_noise)

        # Reconstruct predicted x0:  x0 = (x_t - sqrt(1-ᾱ)*eps) / sqrt(ᾱ)
        x0_pred = (x_t - sqrt_one_minus_alphas_cumprod_t * predicted_noise) / \
                  sqrt_alphas_cumprod_t
        x0_pred = x0_pred.clamp(-4, 4)

        l_spectral  = self.spectral_loss(x0_pred, x0_true)
        l_gradient  = self.gradient_loss(x0_pred, x0_true)
        l_x0        = self.x0_weight * F.l1_loss(x0_pred, x0_true)

        total = l_noise + l_spectral + l_gradient + l_x0
        return total, l_noise, l_spectral, l_gradient, l_x0
