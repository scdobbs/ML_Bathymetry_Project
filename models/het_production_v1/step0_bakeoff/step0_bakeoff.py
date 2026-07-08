"""Step-0 bake-off (directive 31 §1.A, four fairness guards).

Head-to-head, on IDENTICAL held-out native patches, of:
  (FLOW)  the EXISTING native flow checkpoint, sampled at the LOCKED churn=0
          (report 27: churn>0 fakes calibration with white static; the report-26
          cal number was a churn=0.5 number and is not the honest one).
  (REG)   a coarse-roughness heteroscedastic regressor: mean = lr_up (the same
          bicubic baseline the flow uses in residual mode), per-pixel spread =
          a red-noise fine-band texture whose amplitude is predicted from cheap
          LR-only coarse features [coarse RMS, coarse curvature, coarse anisotropy]
          (guard 3: no-strawman feature set), fit POOLED/LOUO (guard 2: no
          per-pair oracle) on the 1-m-native fit pairs.

Decisive metric (guard 1): held-out CONDITIONAL calibration — coverage stratified
by coarse-roughness bin — plus CRPS/sharpness. Reported per bin AND per pair.
Both engines: identical patches, identical roughness bins, identical N realizations,
identical empirical-quantile coverage + sample-CRPS code (comparison hygiene).

Held-out = the flow's genuine held-out pairs (it trained on all others):
  NA090__MGDS_31212 (1 m native, k8)  — the primary 1-m-native verdict fold.
  TN159__MGDS_21981 (harmonized ~3.2 m, k4) — reported SEPARATELY (regime differs).

Verdict is COMPARATIVE ("does the existing flow earn a retrain"), NOT an absolute
calibration certification (n=2 folds; §2.5 scope binds). Read-only OAK; scratch out.
"""
from __future__ import annotations
import argparse, glob, json, os, re
from pathlib import Path
import numpy as np
import torch

from training.config import Config
from models.diffusion import build_rectified_flow_from_checkpoint
from utils.device import get_device

ALPHAS = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
HELD_OUT = {"NA090__MGDS_31212": "native_1m", "TN159__MGDS_21981": "harmonized"}
FIT_PAIRS = ["NA080__MGDS_31290", "TN399__MGDS_30373", "RR1506__MGDS_29779",
             "AT37-13__MGDS_31199", "EW0207__MGDS_32556", "FK181031__MGDS_24618",
             "TN299__MGDS_27339", "TN299__MGDS_31253"]
FNPAT = re.compile(r'^(.*)_r\d{6}_c\d{6}\.npz$')


def pid_of(f):
    m = FNPAT.match(os.path.basename(f))
    return m.group(1) if m else ''


# ---------- band-limited helpers on a normalized 256-px patch (dx = 1 px) ----------
def _detrend(a, m):
    H, W = a.shape
    yy, xx = np.mgrid[0:H, 0:W]
    A = np.c_[xx[m], yy[m], np.ones(m.sum())]
    coef, *_ = np.linalg.lstsq(A, a[m], rcond=None)
    return np.where(m, a - (coef[0] * xx + coef[1] * yy + coef[2]), 0.0)


def _radial_freq(H, W):
    fy = np.fft.fftfreq(H)[:, None]; fx = np.fft.fftfreq(W)[None, :]
    return np.sqrt(fy ** 2 + fx ** 2)


def bandpass(a, m, f_lo, f_hi):
    """Detrended FFT band-pass (cycles/px). Returns real band-limited field."""
    res = _detrend(np.asarray(a, float), m)
    F = np.fft.fft2(res)
    fr = _radial_freq(*a.shape)
    return np.real(np.fft.ifft2(F * ((fr >= f_lo) & (fr <= f_hi))))


def band_rms(a, m, f_lo, f_hi):
    b = bandpass(a, m, f_lo, f_hi)
    return float(np.sqrt(np.mean(b[m] ** 2))) if m.sum() else np.nan


def coarse_features(lr_up, m, f_lr):
    """Cheap LR-only coarse features (guard 3): coarse-band RMS, coarse curvature
    (RMS of the coarse-band Laplacian), coarse anisotropy (power-weighted freq
    covariance eigen-ratio in the coarse annulus)."""
    f_lo, f_hi = f_lr / 2.0, f_lr
    cb = bandpass(lr_up, m, f_lo, f_hi)
    C = float(np.sqrt(np.mean(cb[m] ** 2))) if m.sum() else np.nan
    lap = (np.roll(cb, 1, 0) + np.roll(cb, -1, 0) + np.roll(cb, 1, 1)
           + np.roll(cb, -1, 1) - 4 * cb)
    curv = float(np.sqrt(np.mean(lap[m] ** 2))) if m.sum() else np.nan
    # anisotropy in the coarse annulus
    res = _detrend(np.asarray(lr_up, float), m)
    F = np.fft.fft2(res); P = np.abs(F) ** 2
    H, W = lr_up.shape
    fy = np.fft.fftfreq(H)[:, None] * np.ones((1, W))
    fx = np.fft.fftfreq(W)[None, :] * np.ones((H, 1))
    fr = np.sqrt(fy ** 2 + fx ** 2)
    ann = (fr >= f_lo) & (fr <= f_hi) & (fr > 0)
    aniso = np.nan
    if ann.sum() >= 8:
        w = P[ann]; sw = w.sum() + 1e-30
        cxx = np.sum(w * fx[ann] ** 2) / sw; cyy = np.sum(w * fy[ann] ** 2) / sw
        cxy = np.sum(w * fx[ann] * fy[ann]) / sw
        ev = np.linalg.eigvalsh([[cxx, cxy], [cxy, cyy]])
        if ev.min() > 0:
            aniso = float(np.sqrt(ev.max() / ev.min()))
    return C, curv, aniso


def fine_slope(hr, m, f_lo, f_hi):
    """Radial log-log PSD slope of HR in the fine band (for the red-texture prior)."""
    res = _detrend(np.asarray(hr, float), m)
    P = np.abs(np.fft.fft2(res)) ** 2
    fr = _radial_freq(*hr.shape)
    ann = (fr >= f_lo) & (fr <= f_hi) & (fr > 0) & (P > 0)
    if ann.sum() < 6:
        return -3.6
    s = np.polyfit(np.log10(fr[ann]), np.log10(P[ann]), 1)[0]
    return float(np.clip(s, -6.0, -1.0))


def red_texture(H, W, m, f_lo, f_hi, slope, sigma, rng):
    """Zero-mean fine-band texture with radial PSD ~ f^slope, normalized to spatial
    RMS = sigma over valid cells. Per-pixel across-draw std ~ sigma (Gaussian)."""
    fr = _radial_freq(H, W)
    ann = (fr >= f_lo) & (fr <= f_hi) & (fr > 0)
    amp = np.zeros_like(fr); amp[ann] = fr[ann] ** (slope / 2.0)
    w = rng.standard_normal((H, W))
    field = np.real(np.fft.ifft2(np.fft.fft2(w) * amp))
    r = np.sqrt(np.mean(field[m] ** 2)) if m.sum() else 0.0
    return field * (sigma / r) if r > 1e-12 else field * 0.0


# ---------- coverage / CRPS (identical for both engines) ----------
def coverage_counts(reals, y, valid):
    """reals (N,H,W); y,valid (H,W). Return per-alpha covered-pixel counts + n."""
    rv = reals.reshape(reals.shape[0], -1)[:, valid.ravel()]
    vy = y[valid]
    cc = np.zeros(len(ALPHAS))
    for i, a in enumerate(ALPHAS):
        lo = np.quantile(rv, (1 - a) / 2, axis=0); hi = np.quantile(rv, (1 + a) / 2, axis=0)
        cc[i] = np.sum((vy >= lo) & (vy <= hi))
    return cc, float(vy.size), rv, vy


def crps_samples(rv, vy):
    N = rv.shape[0]
    t1 = np.abs(rv - vy[None]).mean(axis=0)
    s = np.sort(rv, axis=0); i = np.arange(1, N + 1)[:, None]
    t2 = (2.0 / (N * N)) * ((2 * i - N - 1) * s).sum(axis=0)
    return float((t1 - 0.5 * t2).mean())


# ---------- patch feature extraction ----------
def patch_bands(d):
    """From a native npz: normalized hr, lr_up, valid mask, and pixel-freq bands."""
    hr = d['hr'].squeeze().astype(float)
    lr_up = d['lr_up'].squeeze().astype(float)
    hv = d['hr_valid'].squeeze().astype(bool)
    valid = hv
    if 'lr_valid' in d.files:
        import torch.nn.functional as F
        lv = d['lr_valid'].squeeze().astype(np.float32)
        lv_hr = F.interpolate(torch.from_numpy(lv)[None, None], size=hr.shape,
                              mode='nearest')[0, 0].numpy() > 0.5
        valid = hv & lv_hr
    stats = d['stats']
    k = float(stats[2]); lr_native = float(stats[4]); hr_res = float(stats[5]); std_m = float(stats[1])
    f_lr = hr_res / lr_native                       # cycles/px
    f_hi = min(k * f_lr, 0.45)
    return dict(hr=hr, lr_up=lr_up, valid=valid, k=k, lr_native=lr_native,
                hr_res=hr_res, std_m=std_m, f_lr=f_lr, f_hi=f_hi)


def features_and_target(pb):
    m = pb['valid']
    if m.sum() < 64 or pb['f_hi'] <= pb['f_lr']:
        return None
    C, curv, aniso = coarse_features(pb['lr_up'], m, pb['f_lr'])
    yfine = band_rms(pb['hr'], m, pb['f_lr'], pb['f_hi'])          # fine-band RMS of HR (target)
    if not (np.isfinite(C) and np.isfinite(curv) and np.isfinite(aniso) and np.isfinite(yfine)):
        return None
    return dict(C=C, curv=curv, aniso=aniso, yfine=yfine)


# ---------- regressor (pooled/LOUO) ----------
def fit_loglinear(X, y):
    """log10(y) ~ [standardized features] + 1. Return (beta, mu, sd) for predict."""
    ly = np.log10(y + 1e-9)
    mu = X.mean(0); sd = X.std(0) + 1e-9
    Xs = np.c_[(X - mu) / sd, np.ones(len(X))]
    beta, *_ = np.linalg.lstsq(Xs, ly, rcond=None)
    return beta, mu, sd


def predict_loglinear(fit, X):
    beta, mu, sd = fit
    return 10.0 ** (np.c_[(X - mu) / sd, np.ones(len(X))] @ beta)


def louo_r2(feat_by_pair):
    """Leave-one-pair-out held-out R^2 (log space), pooled across fit pairs."""
    pairs = list(feat_by_pair)
    preds, truth = [], []
    for held in pairs:
        Xtr = np.vstack([feat_by_pair[p]['X'] for p in pairs if p != held])
        ytr = np.concatenate([feat_by_pair[p]['y'] for p in pairs if p != held])
        fit = fit_loglinear(Xtr, ytr)
        Xte = feat_by_pair[held]['X']; yte = feat_by_pair[held]['y']
        preds.append(np.log10(predict_loglinear(fit, Xte) + 1e-9))
        truth.append(np.log10(yte + 1e-9))
    p = np.concatenate(preds); t = np.concatenate(truth)
    ss_res = np.sum((t - p) ** 2); ss_tot = np.sum((t - t.mean()) ** 2)
    per_pair = {}
    for held in pairs:
        Xtr = np.vstack([feat_by_pair[p]['X'] for p in pairs if p != held])
        ytr = np.concatenate([feat_by_pair[p]['y'] for p in pairs if p != held])
        fit = fit_loglinear(Xtr, ytr)
        yte = feat_by_pair[held]['y']
        pr = np.log10(predict_loglinear(fit, feat_by_pair[held]['X']) + 1e-9)
        tr = np.log10(yte + 1e-9)
        sr = np.sum((tr - pr) ** 2); st = np.sum((tr - tr.mean()) ** 2)
        per_pair[held] = round(1 - sr / st, 3) if st > 0 else None
    return (1 - ss_res / ss_tot if ss_tot > 0 else None), per_pair


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--patches_dir', required=True)
    ap.add_argument('--out_dir', default='/scratch/users/scdobbs/step0_bakeoff')
    ap.add_argument('--base_ch', type=int, default=32)
    ap.add_argument('--n_real', type=int, default=40)
    ap.add_argument('--n_steps', type=int, default=100)
    ap.add_argument('--cfg_scale', type=float, default=1.1)
    ap.add_argument('--churn', type=float, default=0.0)          # LOCKED
    ap.add_argument('--fit_cap', type=int, default=300)          # patches/pair for the fit
    ap.add_argument('--eval_cap', type=int, default=160)         # held-out patches/pair scored by flow
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    device = torch.device(get_device())

    # ---- 1. Fit the regressor: coarse features -> fine-band RMS on fit pairs (LOUO) ----
    print(">>> collecting regressor fit features (train split, fit pairs)", flush=True)
    train_files = sorted(glob.glob(os.path.join(args.patches_dir, 'train', '*.npz')))
    feat_by_pair = {}
    for pid in FIT_PAIRS:
        fs = [f for f in train_files if pid_of(f) == pid]
        if not fs:
            print(f"   WARN no train patches for fit pair {pid}"); continue
        if len(fs) > args.fit_cap:
            fs = [fs[i] for i in np.unique(np.linspace(0, len(fs) - 1, args.fit_cap).round().astype(int))]
        rows = []
        for f in fs:
            ft = features_and_target(patch_bands(np.load(f, allow_pickle=True)))
            if ft: rows.append(ft)
        if len(rows) >= 8:
            X = np.array([[np.log10(r['C'] + 1e-9), r['curv'], r['aniso']] for r in rows])
            y = np.array([r['yfine'] for r in rows])
            feat_by_pair[pid] = dict(X=X, y=y, n=len(rows))
            print(f"   {pid:22s} n={len(rows)}", flush=True)
    pooled_r2, per_pair_r2 = louo_r2(feat_by_pair)
    print(f"   POOLED LOUO held-out R^2 = {pooled_r2}  per-pair={per_pair_r2}", flush=True)
    # final regressor for held-out prediction: fit on ALL fit pairs
    Xall = np.vstack([v['X'] for v in feat_by_pair.values()])
    yall = np.concatenate([v['y'] for v in feat_by_pair.values()])
    reg_fit = fit_loglinear(Xall, yall)

    # ---- 2. Load the flow ----
    cfg = Config(base_ch=args.base_ch, scale_conditioning=True, residual_mode=True,
                 sr_ratio=16, lr_patch_size=16, compile_model=False, distributed=False,
                 native_mode=True, scale_dim=2)
    flow = build_rectified_flow_from_checkpoint(args.checkpoint, cfg, device)
    print(f">>> flow loaded (churn={args.churn}, n_real={args.n_real}, steps={args.n_steps})", flush=True)

    import pandas as pd

    def cov_curve(sub, eng):
        n = sub['n_valid'].sum()
        return [float(sub[f'cov{eng}_{i}'].sum() / max(n, 1)) for i in range(len(ALPHAS))]

    def cal_err(curve):
        return float(np.mean(np.abs(np.array(curve) - ALPHAS)))

    def aggregate_and_save(rec):
        """Build per-pair + conditional (roughness-binned) summary and write to disk.
        Called after EACH held-out pair so a partial run still yields usable results."""
        df = pd.DataFrame(rec)
        df.to_parquet(out / 'bakeoff_perpatch.parquet', index=False)
        summary = dict(alphas=ALPHAS.tolist(), pooled_louo_r2=pooled_r2,
                       per_pair_louo_r2=per_pair_r2, n_real=args.n_real, n_steps=args.n_steps,
                       churn=args.churn, per_pair={}, conditional={})
        for pid in df.pair_id.unique():
            sub = df[df.pair_id == pid]
            cf, cr = cov_curve(sub, 'flow'), cov_curve(sub, 'reg')
            summary['per_pair'][pid] = dict(
                regime=HELD_OUT.get(pid, '?'), n_patches=int(len(sub)),
                flow=dict(coverage=cf, cal_err=cal_err(cf), crps=float(sub.crps_flow.mean())),
                reg=dict(coverage=cr, cal_err=cal_err(cr), crps=float(sub.crps_reg.mean())),
                fine_amp=dict(hr=float(sub.hr_fine.mean()), flow=float(sub.flow_fine.mean()),
                              reg_sigma=float(sub.sigma_reg.mean())))
            qs = np.quantile(sub['C'], [1 / 3, 2 / 3])
            binlab = np.digitize(sub['C'], qs)
            summary['conditional'][pid] = {}
            for b in (0, 1, 2):
                s2 = sub[binlab == b]
                if len(s2) < 3:
                    continue
                cf2, cr2 = cov_curve(s2, 'flow'), cov_curve(s2, 'reg')
                summary['conditional'][pid][f'rough_bin{b}'] = dict(
                    n=int(len(s2)), C_med=float(s2['C'].median()),
                    flow_cal_err=cal_err(cf2), reg_cal_err=cal_err(cr2),
                    flow_cov=cf2, reg_cov=cr2,
                    flow_crps=float(s2.crps_flow.mean()), reg_crps=float(s2.crps_reg.mean()))
        (out / 'bakeoff_summary.json').write_text(json.dumps(summary, indent=2, default=str))
        return summary

    # ---- 3. Score both engines on identical held-out patches ----
    test_files = sorted(glob.glob(os.path.join(args.patches_dir, 'test', '*.npz')))
    rec = []
    for pid, regime in HELD_OUT.items():
        fs = [f for f in test_files if pid_of(f) == pid]
        if not fs:
            print(f"   WARN no test patches for held-out {pid}"); continue
        if len(fs) > args.eval_cap:
            fs = [fs[i] for i in np.unique(np.linspace(0, len(fs) - 1, args.eval_cap).round().astype(int))]
        print(f">>> {pid} ({regime}): {len(fs)} held-out patches", flush=True)
        for j, f in enumerate(fs):
            d = np.load(f, allow_pickle=True)
            pb = patch_bands(d)
            m = pb['valid']
            if m.sum() < 64 or pb['f_hi'] <= pb['f_lr']:
                continue
            C, curv, aniso = coarse_features(pb['lr_up'], m, pb['f_lr'])
            if not np.isfinite([C, curv, aniso]).all():
                continue
            hr = pb['hr']; H, W = hr.shape

            # --- FLOW: churn=0 native sampling ---
            scale_t = torch.from_numpy(np.atleast_1d(d['scale']).astype(np.float32)).to(device)
            lr_ctx = torch.from_numpy(d['lr_context']).float().to(device)[None]
            lr_up_t = torch.from_numpy(d['lr_up']).float().to(device)[None]
            fr = flow.sample_native(lr_ctx, lr_up_t, args.n_real, scale_t,
                                    n_steps=args.n_steps, cfg_scale=args.cfg_scale,
                                    churn=args.churn).squeeze(1).float().cpu().numpy()
            fcc, nval, frv, vy = coverage_counts(fr, hr, m)
            crps_flow = crps_samples(frv, vy)
            flow_fine = band_rms((fr - fr.mean(0)).std(0), m, pb['f_lr'], pb['f_hi'])  # inter-real fine spread

            # --- REG: mean = lr_up, red-texture drape at predicted fine RMS ---
            sigma = float(predict_loglinear(reg_fit, np.array([[np.log10(C + 1e-9), curv, aniso]]))[0])
            slope = fine_slope(hr, m, pb['f_lr'], pb['f_hi'])
            base = pb['lr_up']
            rr = np.stack([base + red_texture(H, W, m, pb['f_lr'], pb['f_hi'], slope, sigma, rng)
                           for _ in range(args.n_real)])
            rcc, _, rrv, _ = coverage_counts(rr, hr, m)
            crps_reg = crps_samples(rrv, vy)
            hr_fine = band_rms(hr, m, pb['f_lr'], pb['f_hi'])

            rec.append(dict(pair_id=pid, regime=regime, C=C, curv=curv, aniso=aniso,
                            n_valid=nval, k=pb['k'], std_m=pb['std_m'],
                            sigma_reg=sigma, hr_fine=hr_fine, flow_fine=flow_fine, slope=slope,
                            crps_flow=crps_flow, crps_reg=crps_reg,
                            **{f'covflow_{i}': fcc[i] for i in range(len(ALPHAS))},
                            **{f'covreg_{i}': rcc[i] for i in range(len(ALPHAS))}))
            if (j + 1) % 25 == 0:
                print(f"   {pid} {j+1}/{len(fs)}", flush=True)
        summary = aggregate_and_save(rec)   # incremental save after each held-out pair
        print(f"   [saved] {pid} done; summary updated", flush=True)

    print("\n================ STEP-0 BAKE-OFF SUMMARY ================")
    print(f"pooled LOUO regressor held-out R^2 = {pooled_r2}")
    for pid, r in summary['per_pair'].items():
        print(f"\n{pid} [{r['regime']}] n={r['n_patches']}")
        print(f"  FLOW cal_err={r['flow']['cal_err']:.3f} crps={r['flow']['crps']:.3f} cov={np.round(r['flow']['coverage'],3).tolist()}")
        print(f"  REG  cal_err={r['reg']['cal_err']:.3f} crps={r['reg']['crps']:.3f} cov={np.round(r['reg']['coverage'],3).tolist()}")
        print(f"  fine amp (norm): HR={r['fine_amp']['hr']:.3f} flow={r['fine_amp']['flow']:.3f} reg_sigma={r['fine_amp']['reg_sigma']:.3f}")
        for b, bd in summary['conditional'].get(pid, {}).items():
            print(f"    {b} (n={bd['n']}, C_med={bd['C_med']:.3f}): "
                  f"flow cal_err={bd['flow_cal_err']:.3f} | reg cal_err={bd['reg_cal_err']:.3f}")
    print("\nwrote", out / 'bakeoff_summary.json')


if __name__ == '__main__':
    main()
