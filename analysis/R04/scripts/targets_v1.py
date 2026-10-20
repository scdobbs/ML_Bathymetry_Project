"""targets_v1 — FROZEN target definitions (directive R04 §1; tag phase0-freeze). Implemented once, used everywhere.

Band-passed fields (§1.1), per 256 x 256 HR-px window at stride 128, on the harmonized HR grid:
  1. least-squares plane removed over joint-valid px, invalid px set to 0;
  2. Tukey (alpha = 0.25) taper;
  3. FFT band-pass over [f_lr, f_hi] (R02 T32B: f_lr = hr_res / lr_native, f_hi = min(k f_lr, 0.45), HR cycles per px)
     with raised-cosine transitions of +/- 0.25 octave around each edge (product of a raised-cosine high-pass step at
     f_lr and a raised-cosine low-pass step at f_hi, both in log2 frequency);
  4. central 128 x 128 kept;  5. re-masked to joint-valid (NaN elsewhere).
  F_hr  = band-passed (HR - delta)              (true unresolved relief)
  F_res = band-passed (HR - delta - bicubic)    (fine-band misfit; plane excluded by construction)
Summary targets (§1.2) over joint-valid px of a tile (128) or block (32): A_fine = RMS(F_hr), T_fine = RMS(F_res),
E_1m = fraction of px with |F_hr| > E1M_THRESHOLD_M (1.0 m default).
r03_afine() is the R03/R02 convention (plane detrend, zero fill, hard annulus, no taper, RMS over the full 256 window)
used only for the §1.1 consistency check."""
from __future__ import annotations
import numpy as np
from scipy.signal.windows import tukey

WIN, TILE, MARGIN, BLOCK, STRIDE = 256, 128, 64, 32, 128
OCTAVE, TUKEY_ALPHA = 0.25, 0.25
E1M_THRESHOLD_M = 1.0
BLOCK_KEEP_FRAC = 0.75
NBLOCK = (TILE // BLOCK) ** 2          # 16

def radial_freq(n: int) -> np.ndarray:
    fy = np.fft.fftfreq(n)[:, None]; fx = np.fft.fftfreq(n)[None, :]
    return np.sqrt(fy ** 2 + fx ** 2)

def _rc_step(lf: np.ndarray, edge: float, oct: float, rising: bool) -> np.ndarray:
    """Raised-cosine step in log2 frequency: 0 -> 1 (rising) or 1 -> 0 over [edge - oct, edge + oct]."""
    x = np.clip((lf - (edge - oct)) / (2.0 * oct), 0.0, 1.0)
    s = 0.5 * (1.0 - np.cos(np.pi * x))
    return s if rising else 1.0 - s

def raised_cosine_response(fr: np.ndarray, f_lr: float, f_hi: float, oct: float = OCTAVE) -> np.ndarray:
    """Band-pass response on the FFT grid: 1 on [f_lr 2^oct, f_hi 2^-oct], raised-cosine ramps of +/- oct around each edge."""
    with np.errstate(divide="ignore"):
        lf = np.log2(np.where(fr > 0, fr, 1e-12))
    H = _rc_step(lf, np.log2(f_lr), oct, True) * _rc_step(lf, np.log2(f_hi), oct, False)
    H[fr == 0] = 0.0
    return H

def hard_annulus_response(fr: np.ndarray, f_lr: float, f_hi: float) -> np.ndarray:
    return ((fr >= f_lr) & (fr <= f_hi)).astype(float)

def tukey2d(n: int = WIN, alpha: float = TUKEY_ALPHA) -> np.ndarray:
    w = tukey(n, alpha); return np.outer(w, w)

def detrend_plane(a: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Least-squares plane over m; residual on m, 0 elsewhere (identical to the R02/R03 _detrend)."""
    H, W = a.shape; yy, xx = np.mgrid[0:H, 0:W]
    A = np.c_[xx[m], yy[m], np.ones(int(m.sum()))]
    coef, *_ = np.linalg.lstsq(A, a[m].astype(float), rcond=None)
    return np.where(m, a - (coef[0] * xx + coef[1] * yy + coef[2]), 0.0)

def bandpass(field: np.ndarray, m: np.ndarray, H: np.ndarray, taper: np.ndarray | None) -> np.ndarray:
    res = detrend_plane(field, m)
    if taper is not None: res = res * taper
    return np.real(np.fft.ifft2(np.fft.fft2(res) * H))

def window_fields(hr: np.ndarray, joint: np.ndarray, delta: float, lr_up: np.ndarray, H: np.ndarray, taper: np.ndarray):
    """§1.1 steps 1-5 for one 256 window. Returns (F_hr, F_res) central 128 x 128 float32 crops, NaN outside joint."""
    assert hr.shape == (WIN, WIN) and joint.dtype == bool
    hr0 = np.where(joint, hr, 0.0).astype(float)
    F_hr = bandpass(hr0 - delta, joint, H, taper)
    F_res = bandpass(hr0 - delta - np.where(joint, lr_up, 0.0), joint, H, taper)
    sl = slice(MARGIN, MARGIN + TILE); jc = joint[sl, sl]
    return (np.where(jc, F_hr[sl, sl], np.nan).astype(np.float32), np.where(jc, F_res[sl, sl], np.nan).astype(np.float32))

def r03_afine(hr: np.ndarray, joint: np.ndarray, delta: float, f_lr: float, f_hi: float) -> float:
    """R03 §3 / R02 T32B convention: plane detrend on valid px, zero fill, hard annulus [f_lr, f_hi], no taper, RMS over valid px."""
    if joint.sum() < 64: return float("nan")
    b = bandpass(np.where(joint, hr, 0.0) - delta, joint, hard_annulus_response(radial_freq(hr.shape[0]), f_lr, f_hi), None)
    return float(np.sqrt(np.mean(b[joint] ** 2)))

def summary(F_hr: np.ndarray, F_res: np.ndarray, m: np.ndarray, h: float = E1M_THRESHOLD_M) -> dict:
    """A_fine, T_fine, E_1m over the joint-valid px m of any support (tile or block). NaN if no valid px."""
    n = int(m.sum())
    if n == 0: return dict(n_joint=0, frac_joint=0.0, A_fine=np.nan, T_fine=np.nan, E_1m=np.nan)
    a = F_hr[m].astype(float); r = F_res[m].astype(float)
    return dict(n_joint=n, frac_joint=float(n / m.size), A_fine=float(np.sqrt(np.mean(a ** 2))), T_fine=float(np.sqrt(np.mean(r ** 2))),
                E_1m=float(np.mean(np.abs(a) > h)))

def tile_and_blocks(F_hr_t: np.ndarray, F_res_t: np.ndarray, joint_t: np.ndarray, h: float = E1M_THRESHOLD_M):
    """Targets for one 128 tile and its 16 non-overlapping 32 x 32 blocks (block kept iff >= 75 % joint-valid)."""
    assert F_hr_t.shape == (TILE, TILE)
    t = summary(F_hr_t, F_res_t, joint_t, h); blocks = []
    nb = TILE // BLOCK
    for bi in range(nb):
        for bj in range(nb):
            sl = (slice(bi * BLOCK, (bi + 1) * BLOCK), slice(bj * BLOCK, (bj + 1) * BLOCK))
            s = summary(F_hr_t[sl], F_res_t[sl], joint_t[sl], h)
            blocks.append(dict(block_id=bi * nb + bj, br=bi * BLOCK, bc=bj * BLOCK, kept=bool(s["frac_joint"] >= BLOCK_KEEP_FRAC), **s))
    return t, blocks
