"""
Rectified-flow training and sampling for conditional bathymetric SR.

Replaces the Stage 1 GaussianDiffusion (DDPM/DDIM) with a continuous-time
linear-interpolant flow model:

    x_t = (1 - t) * x0 + t * z         t ∈ (0, 1), z ~ N(0, I)
    v   = dx_t / dt = z - x0
    L_FM = E[ || v_θ(x_t, t, lr) - (z - x0) ||² ]

Sampling integrates dx/dt = v_θ from t=1 (pure noise) to t=0 (clean signal)
using a 2nd-order Heun step (with an optional Euler fallback). A Langevin-
style churn knob injects fresh noise mid-trajectory to trade determinism
for sample diversity. Classifier-free guidance is built in: with probability
`cfg.cfg_dropout_prob` during training the LR conditioning is swapped for a
zero null-token; at inference we extrapolate between the conditional and
unconditional predictions.

No noise-schedule buffers (betas, alphas_cumprod, etc.) — flow matching
does not need them.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.unet import BathyUNet


class RectifiedFlow(nn.Module):
    """Flow-matching wrapper around a BathyUNet (or any forward(x, t, lr_raw))."""

    def __init__(self, model: nn.Module, cfg):
        super().__init__()
        self.model = model
        # Cache the knobs we read most often.
        self.cfg = cfg
        self.time_sampling = cfg.time_sampling
        self.logit_normal_mean = cfg.logit_normal_mean
        self.logit_normal_std = cfg.logit_normal_std
        self.n_sampling_steps = cfg.n_sampling_steps
        self.sampler = cfg.sampler
        self.churn = cfg.churn
        self.cfg_dropout_prob = cfg.cfg_dropout_prob
        self.cfg_scale = cfg.cfg_scale
        self.sr_ratio = cfg.sr_ratio
        self.residual_mode = bool(getattr(cfg, 'residual_mode', False))
        # Null conditioning for CFG (a zero tensor at LR shape). Buffer so it
        # moves with the module and is deep-copyable for EMA.
        self.register_buffer(
            'null_lr',
            torch.zeros(1, 1, cfg.lr_patch_size, cfg.lr_patch_size),
        )

    # -----------------------------------------------------------------------
    # Time and interpolation
    # -----------------------------------------------------------------------

    def sample_t(self, batch_size: int, device) -> torch.Tensor:
        """Draw a continuous t ∈ (0, 1) per sample.

        Logit-normal sampling concentrates mass in the middle of the
        interval (large gradient signal for both ends of the schedule)
        and matches the SDXL / Stable Diffusion 3 training recipe.
        """
        if self.time_sampling == 'logit_normal':
            u = torch.randn(batch_size, device=device) * self.logit_normal_std \
                + self.logit_normal_mean
            return torch.sigmoid(u)
        if self.time_sampling == 'uniform':
            return torch.rand(batch_size, device=device)
        raise ValueError(f"unknown time_sampling: {self.time_sampling}")

    def interpolate(self, x0: torch.Tensor, z: torch.Tensor,
                    t: torch.Tensor) -> torch.Tensor:
        """Linear interpolant x_t = (1 - t) x0 + t z."""
        t_b = t.view(-1, 1, 1, 1)
        return (1 - t_b) * x0 + t_b * z

    def velocity_target(self, x0: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Target velocity for the linear interpolant: z - x0."""
        return z - x0

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------

    def training_step(self, hr: torch.Tensor, lr_raw: torch.Tensor,
                      stats: torch.Tensor = None) -> dict:
        """Forward training pass.

        Args:
            hr:     (B, 1, H, W) clean HR target, per-patch normalized.
            lr_raw: (B, 1, h, w) raw LR conditioning, per-patch normalized.
            stats:  (B, 2) optional [mean, std] used to normalize the patch.
                    Passed through so the data-consistency loss can skip
                    patches whose HR std hit the divide-by-zero floor.

        Returns:
            dict with v_pred, v_target, t, x0_pred (and x0_true=hr for losses).
        """
        B = hr.shape[0]
        device = hr.device

        # In residual mode the flow operates on the high-frequency residual
        # (HR − bicubic-upsampled LR), and the bicubic is concatenated as a
        # second input channel to the model. The bicubic baseline is added
        # back at sampling time.
        if self.residual_mode:
            lr_up = F.interpolate(lr_raw, scale_factor=self.sr_ratio,
                                  mode='bicubic', align_corners=False)
            signal = hr - lr_up
        else:
            lr_up = None
            signal = hr

        t = self.sample_t(B, device)
        z = torch.randn_like(signal)
        x_t = self.interpolate(signal, z, t)
        v_target = self.velocity_target(signal, z)

        # Classifier-free guidance dropout. Replace each sample's LR with the
        # null token independently with probability cfg_dropout_prob. Only
        # the cross-attention LR is dropped — the bicubic input remains as
        # an architectural component (it's "input", not "conditioning").
        if self.training and self.cfg_dropout_prob > 0.0:
            keep = (torch.rand(B, device=device) >= self.cfg_dropout_prob)
            keep_mask = keep  # (B,) bool — for downstream data-consistency masking
            keep_b = keep.view(B, 1, 1, 1).to(lr_raw.dtype)
            lr_in = keep_b * lr_raw + (1 - keep_b) * self.null_lr
        else:
            keep_mask = torch.ones(B, device=device, dtype=torch.bool)
            lr_in = lr_raw

        x_model_in = torch.cat([x_t, lr_up], dim=1) if self.residual_mode else x_t
        v_pred = self.model(x_model_in, t, lr_in)

        # x_t = x0 + t * v  ⇒  x0 = x_t - t * v  (x0 here is the signal:
        # residual in residual mode, HR otherwise)
        x0_pred_signal = x_t - t.view(-1, 1, 1, 1) * v_pred
        # For downstream losses that compare against absolute HR (consistency,
        # spectral), reconstruct the predicted HR from the residual.
        x0_pred_hr = x0_pred_signal + lr_up if self.residual_mode else x0_pred_signal

        return {
            'v_pred':         v_pred,
            'v_target':       v_target,
            't':              t,
            'x0_pred':        x0_pred_hr,   # always in HR units (residual + lr_up if applicable)
            'x0_true':        hr,           # passthrough for the spectral loss
            'lr_raw':         lr_raw,       # passthrough for data-consistency
            'cfg_keep_mask':  keep_mask,    # (B,) bool — True where lr_raw is real
            'stats':          stats,        # (B, 2) or None — for degenerate-patch filtering
        }

    # -----------------------------------------------------------------------
    # Sampling (Heun, with optional Euler and Langevin churn)
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        lr_raw: torch.Tensor,
        n_realizations: int,
        n_steps: int = None,
        cfg_scale: float = None,
        churn: float = None,
        seed: int = None,
    ) -> torch.Tensor:
        """Integrate dx/dt = v_θ from t=1 to t=0 with a Heun step.

        lr_raw can be (1, 1, h, w) — broadcast to n_realizations — or
        (n_realizations, 1, h, w) for distinct conditioning per sample.
        """
        n_steps = n_steps if n_steps is not None else self.n_sampling_steps
        cfg_scale = cfg_scale if cfg_scale is not None else self.cfg_scale
        churn = churn if churn is not None else self.churn
        device = lr_raw.device

        # Broadcast / accept (n_realizations, 1, h, w)
        if lr_raw.shape[0] == 1:
            lr_cond = lr_raw.expand(n_realizations, -1, -1, -1).contiguous()
        elif lr_raw.shape[0] == n_realizations:
            lr_cond = lr_raw
        else:
            raise ValueError(
                f"lr_raw batch ({lr_raw.shape[0]}) must be 1 or n_realizations "
                f"({n_realizations})"
            )

        # HR resolution from LR shape × sr_ratio
        h_lr, w_lr = lr_cond.shape[-2], lr_cond.shape[-1]
        H = h_lr * self.sr_ratio
        W = w_lr * self.sr_ratio

        # Residual mode: bicubic-upsample the LR conditioning once and pass
        # it as a 2nd input channel throughout the integration. The final
        # integrated signal is the residual; we add the bicubic back below.
        if self.residual_mode:
            lr_up = F.interpolate(lr_cond, scale_factor=self.sr_ratio,
                                  mode='bicubic', align_corners=False)
        else:
            lr_up = None

        # Initial noise
        if seed is not None:
            g = torch.Generator(device=device).manual_seed(seed)
            x = torch.randn(n_realizations, 1, H, W, device=device, generator=g)
        else:
            x = torch.randn(n_realizations, 1, H, W, device=device)

        # Linear time schedule from 1 → 0
        ts = torch.linspace(1.0, 0.0, n_steps + 1, device=device, dtype=x.dtype)

        # Precompute the null LR batch for CFG (only used when scale != 1.0)
        use_cfg = cfg_scale != 1.0
        null_lr = self.null_lr.expand_as(lr_cond) if use_cfg else None

        for i in range(n_steps):
            t_cur, t_next = ts[i], ts[i + 1]
            dt = t_next - t_cur   # negative (we integrate backwards in t)

            t_b = t_cur.expand(n_realizations)
            v_cur = self._velocity(x, t_b, lr_cond, null_lr, cfg_scale, use_cfg, lr_up)

            x_euler = x + dt * v_cur

            if self.sampler == 'heun' and t_next > 0:
                t_next_b = t_next.expand(n_realizations)
                v_next = self._velocity(x_euler, t_next_b, lr_cond, null_lr,
                                        cfg_scale, use_cfg, lr_up)
                x_next = x + dt * 0.5 * (v_cur + v_next)
            else:
                x_next = x_euler

            # Langevin churn: inject fresh noise scaled by sqrt(churn * |Δt|)
            if churn > 0 and t_next > 0:
                noise_scale = (churn * (t_cur - t_next).abs()).sqrt()
                x_next = x_next + noise_scale * torch.randn_like(x_next)

            x = x_next

        # In residual mode the integrated x is the predicted HR residual.
        # The bicubic baseline is added unconditionally to recover HR.
        if self.residual_mode:
            x = x + lr_up

        return x

    def _velocity(self, x, t_b, lr_cond, null_lr, cfg_scale, use_cfg, lr_up=None):
        """Predict velocity with optional CFG, in a single batched forward.

        In residual mode, lr_up is concatenated as a 2nd input channel before
        the model forward. The velocity is still 1-channel (velocity of the
        residual).
        """
        x_in = torch.cat([x, lr_up], dim=1) if self.residual_mode else x
        if not use_cfg:
            return self.model(x_in, t_b, lr_cond)
        x_pair  = torch.cat([x_in,    x_in],    dim=0)
        t_pair  = torch.cat([t_b,     t_b],     dim=0)
        lr_pair = torch.cat([lr_cond, null_lr], dim=0)
        v_pair = self.model(x_pair, t_pair, lr_pair)
        v_cond, v_uncond = v_pair.chunk(2, dim=0)
        return v_uncond + cfg_scale * (v_cond - v_uncond)


# ---------------------------------------------------------------------------
# Checkpoint factory
# ---------------------------------------------------------------------------

def build_rectified_flow_from_checkpoint(checkpoint_path, cfg, device):
    """Construct a RectifiedFlow and load EMA weights from a checkpoint.

    Replaces the Stage 1 build_diffusion_from_checkpoint factory.
    """
    in_ch = 2 if getattr(cfg, 'residual_mode', False) else 1
    unet = BathyUNet(
        base_ch=cfg.base_ch,
        ch_mult=cfg.ch_mult,
        n_heads=cfg.n_heads,
        transformer_levels=cfg.transformer_levels,
        transformer_depth=cfg.transformer_depth,
        time_dim=cfg.time_dim,
        in_ch=in_ch,
    ).to(device)
    flow = RectifiedFlow(unet, cfg).to(device)

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Training (training/train.py:_save_checkpoint) stores the U-Net and the
    # null-LR buffer under separate dotted keys: 'ema.model' / 'ema.null_lr'
    # for the EMA weights, and 'flow.model' / 'flow.null_lr' for the live
    # weights. Earlier code (and external callers) may also pass an old-style
    # flat 'ema' or 'model' state_dict — keep that path as a fallback.
    if 'ema.model' in state:
        flow.model.load_state_dict(state['ema.model'])
        if 'ema.null_lr' in state:
            flow.null_lr.data.copy_(state['ema.null_lr'].to(device))
    elif 'flow.model' in state:
        flow.model.load_state_dict(state['flow.model'])
        if 'flow.null_lr' in state:
            flow.null_lr.data.copy_(state['flow.null_lr'].to(device))
    else:
        ema_state = state.get('ema', state.get('model'))
        if ema_state is None:
            raise KeyError(
                f"Checkpoint at {checkpoint_path} has none of the expected "
                f"keys (ema.model / flow.model / ema / model). Top-level "
                f"keys present: {list(state.keys())}"
            )
        flow.load_state_dict(ema_state)

    flow.eval()
    return flow
