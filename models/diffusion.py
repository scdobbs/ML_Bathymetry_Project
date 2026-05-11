"""
DDPM (Ho et al. 2020) forward and reverse diffusion process,
with DDIM (Song et al. 2020) accelerated sampling.

The model is conditioned on the bicubic-upsampled LR image throughout
all diffusion steps, following SR3 (Saharia et al. 2021).

Key design choices for bathymetric SR:
    - Linear noise schedule (works well for the smooth elevation domain)
    - Noise prediction (epsilon-parameterization) — numerically stable
    - DDIM sampling with 50 steps for fast inference (vs 1000 for DDPM)
    - Multiple independent samples → stochastic realizations → uncertainty maps
"""

import torch
import torch.nn as nn
import numpy as np

from models.unet import BathyUNet


class GaussianDiffusion(nn.Module):
    """
    Wraps the U-Net model with the DDPM diffusion process.

    Args:
        model:       BathyUNet (or any model with forward(x, t) -> noise)
        n_timesteps: Total diffusion steps T (typically 1000)
        beta_start:  Starting noise level
        beta_end:    Ending noise level
    """

    def __init__(self, model, n_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.model = model
        self.T = n_timesteps

        # Linear noise schedule: beta_t linearly from beta_start to beta_end
        betas = torch.linspace(beta_start, beta_end, n_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        # Register as buffers so they move to GPU with .to(device)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', alphas_cumprod.sqrt())
        self.register_buffer('sqrt_one_minus_alphas_cumprod', (1 - alphas_cumprod).sqrt())
        self.register_buffer('sqrt_recip_alphas', (1.0 / alphas).sqrt())

        # Posterior variance q(x_{t-1} | x_t, x_0)
        posterior_var = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_var)
        # Clamp log to avoid log(0) at t=0
        self.register_buffer('posterior_log_variance_clipped',
                             posterior_var.clamp(min=1e-20).log())

    # -----------------------------------------------------------------------
    # Forward process (training)
    # -----------------------------------------------------------------------

    def q_sample(self, x0, t, noise=None):
        """
        Sample x_t from q(x_t | x_0) = N(sqrt(ᾱ_t)*x0, (1-ᾱ_t)*I).

        Args:
            x0:    (B, 1, H, W) clean HR patch (normalized)
            t:     (B,) integer timestep indices
            noise: optional pre-sampled noise tensor

        Returns:
            x_t:   (B, 1, H, W) noisy sample at timestep t
            noise: (B, 1, H, W) the noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ac = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_om_ac = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_ac * x0 + sqrt_om_ac * noise, noise

    def p_losses(self, x0, lr_up, t):
        """
        Compute training loss: predict the noise added at timestep t.

        Args:
            x0:    (B, 1, H, W) clean HR patch
            lr_up: (B, 1, H, W) bicubic-upsampled LR (conditioning)
            t:     (B,) integer timesteps

        Returns:
            Scalar loss value
        """
        noise = torch.randn_like(x0)
        x_t, _ = self.q_sample(x0, t, noise)

        # Condition: concatenate x_t (noisy HR) with lr_up
        model_input = torch.cat([x_t, lr_up], dim=1)  # (B, 2, H, W)
        predicted_noise = self.model(model_input, t)

        return predicted_noise, noise

    # -----------------------------------------------------------------------
    # Reverse process (inference — DDPM)
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def p_sample(self, x_t, lr_up, t_scalar):
        """
        Single DDPM reverse step: sample x_{t-1} from p(x_{t-1} | x_t, lr_up).

        Args:
            x_t:      (B, 1, H, W) noisy sample at step t
            lr_up:    (B, 1, H, W) conditioning LR
            t_scalar: integer scalar timestep

        Returns:
            x_{t-1}: (B, 1, H, W)
        """
        B = x_t.shape[0]
        t = torch.full((B,), t_scalar, device=x_t.device, dtype=torch.long)

        model_input = torch.cat([x_t, lr_up], dim=1)
        eps = self.model(model_input, t)

        beta_t = self.betas[t_scalar]
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[t_scalar]
        sqrt_omacp_t = self.sqrt_one_minus_alphas_cumprod[t_scalar]

        # Predicted mean of p(x_{t-1} | x_t)
        mean = sqrt_recip_alpha_t * (x_t - beta_t / sqrt_omacp_t * eps)

        if t_scalar == 0:
            return mean
        else:
            noise = torch.randn_like(x_t)
            log_var = self.posterior_log_variance_clipped[t_scalar]
            return mean + (0.5 * log_var).exp() * noise

    @torch.no_grad()
    def sample_ddpm(self, lr_up, n_realizations=1):
        """
        Full DDPM reverse process: T steps from pure noise to HR.

        Args:
            lr_up:          (1, 1, H, W) or (B, 1, H, W) conditioning LR
            n_realizations: number of independent samples to draw

        Returns:
            (n_realizations, 1, H, W) HR realizations
        """
        device = lr_up.device
        B, _, H, W = lr_up.shape
        lr_rep = lr_up.expand(n_realizations, -1, -1, -1)

        x = torch.randn(n_realizations, 1, H, W, device=device)
        for t in reversed(range(self.T)):
            x = self.p_sample(x, lr_rep, t)
        return x

    # -----------------------------------------------------------------------
    # DDIM sampling (accelerated, 50 steps)
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def sample_ddim(self, lr_up, n_realizations=1, n_steps=50, eta=0.0):
        """
        DDIM accelerated sampling (Song et al. 2020).
        Deterministic when eta=0, stochastic when eta=1 (recovers DDPM).

        Args:
            lr_up:          (1, 1, H, W) conditioning LR patch
            n_realizations: number of independent samples
            n_steps:        number of DDIM steps (default 50)
            eta:            stochasticity level (0=deterministic, 1=full DDPM noise)

        Returns:
            (n_realizations, 1, H, W) HR realizations
        """
        device = lr_up.device
        B, _, H, W = lr_up.shape
        lr_rep = lr_up.expand(n_realizations, -1, -1, -1)

        # Select evenly-spaced timesteps
        step_seq = np.linspace(0, self.T - 1, n_steps, dtype=int)
        step_seq = list(reversed(step_seq))   # go from T-1 down to 0

        x = torch.randn(n_realizations, 1, H, W, device=device)

        for i, t_cur in enumerate(step_seq):
            t_prev = step_seq[i + 1] if i + 1 < len(step_seq) else -1

            t_tensor = torch.full((n_realizations,), t_cur, device=device, dtype=torch.long)
            model_input = torch.cat([x, lr_rep], dim=1)
            eps = self.model(model_input, t_tensor)

            ac_cur  = self.alphas_cumprod[t_cur]
            ac_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.ones(1, device=device)

            # Predicted x0
            x0_pred = (x - (1 - ac_cur).sqrt() * eps) / ac_cur.sqrt()
            x0_pred = x0_pred.clamp(-4, 4)  # prevent runaway predictions

            # Direction pointing to x_t
            sigma = eta * ((1 - ac_prev) / (1 - ac_cur)).sqrt() * (1 - ac_cur / ac_prev).sqrt()
            dir_xt = (1 - ac_prev - sigma ** 2).sqrt() * eps

            noise = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)
            x = ac_prev.sqrt() * x0_pred + dir_xt + sigma * noise

        return x

    # -----------------------------------------------------------------------
    # Convenience: generate N realizations for a batch of LR patches
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def generate(self, lr_up, n_realizations=50, method='ddim', ddim_steps=50):
        """
        Generate n_realizations stochastic HR samples for each LR in the batch.

        Args:
            lr_up:          (B, 1, H, W) batch of conditioning LR patches
            n_realizations: number of HR samples per LR patch
            method:         'ddim' (fast) or 'ddpm' (slow but exact)
            ddim_steps:     number of DDIM steps

        Returns:
            (B, n_realizations, 1, H, W) HR realizations
        """
        self.model.eval()
        B = lr_up.shape[0]
        results = []
        for b in range(B):
            lr_b = lr_up[b:b+1]   # (1, 1, H, W)
            if method == 'ddim':
                samples = self.sample_ddim(lr_b, n_realizations, ddim_steps)
            else:
                samples = self.sample_ddpm(lr_b, n_realizations)
            results.append(samples)  # (n_realizations, 1, H, W)
        return torch.stack(results, dim=0)  # (B, n_realizations, 1, H, W)


def build_diffusion_from_checkpoint(checkpoint_path, cfg, device):
    """
    Build a GaussianDiffusion model and load EMA weights from a checkpoint.

    Args:
        checkpoint_path: Path to a .pt file written by training/train.py.
        cfg:             Config object (uses cfg.base_ch, cfg.time_dim,
                         cfg.n_timesteps, cfg.beta_start, cfg.beta_end).
        device:          Device string ('cuda' / 'mps' / 'cpu').

    Returns:
        GaussianDiffusion in eval() mode with EMA weights loaded.
    """
    unet = BathyUNet(
        in_ch=2, out_ch=1,
        base_ch=cfg.base_ch, time_dim=cfg.time_dim,
        dropout=0.0,  # no dropout at inference
    ).to(device)
    diffusion = GaussianDiffusion(
        unet,
        n_timesteps=cfg.n_timesteps,
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
    ).to(device)

    state = torch.load(checkpoint_path, map_location=device)
    ema_state = state.get('ema', state.get('model'))
    diffusion.load_state_dict(ema_state)
    diffusion.eval()
    return diffusion
