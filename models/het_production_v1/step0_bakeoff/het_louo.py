"""Directive 33 §2b — mean-learning heteroscedastic engine, LOUO across native folds.

The real §2.2 competitor to the flow: a compact CNN predicting per-pixel (mean, log-variance)
trained with Gaussian NLL — it learns BOTH the mid-scale mean (like the flow) AND a per-pixel
uncertainty (unlike the fixed-bicubic §2.1 regressor). Run leave-one-native-pair-out to answer the
central open question report 32 raised: **can ANY engine learn cross-pair fine-band UQ, or is the
R²=0.06 (3-scalar regressor) a real cross-pair wall?** A full-spatial-field conditioner (this CNN)
is the test of the "3 scalars were just too weak" caveat.

Per fold (hold out one native leakage unit, train on the rest):
  - train HeteroscedasticUNetNative (2-vec scale) with masked Gaussian NLL;
  - evaluate on the held-out pair's patches:
      * FULL-FIELD Gaussian calibration (coverage of HR by N(mean, sigma)), roughness-binned
        — directly comparable to report-32's flow full-field cal_err 0.097;
      * amplitude tracking: predicted per-patch mean(sigma) vs actual rms(HR-mean) and hp_rms(HR)
        → pooled held-out R² across folds = the cross-pair-skill verdict (compare to 0.06).

Self-contained (no repo edit). Read-only OAK; scratch out. No canonical write.
Incremental per-fold save so a partial run survives.
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, "/scratch/users/scdobbs/step0_bakeoff")
import step0_bakeoff as b0
from utils.device import get_device

ALPHAS = b0.ALPHAS
from scipy.stats import norm

# native leakage-unit folds (TN299 pair shares an LR cruise -> held out together).
# NA090 first: it ties to §2a and validates the pipeline before the rest.
FOLDS = {
    "NA090": ["NA090__MGDS_31212"],
    "NA080": ["NA080__MGDS_31290"], "TN399": ["TN399__MGDS_30373"],
    "RR1506": ["RR1506__MGDS_29779"], "AT37-13": ["AT37-13__MGDS_31199"],
    "EW0207": ["EW0207__MGDS_32556"], "FK181031": ["FK181031__MGDS_24618"],
    "TN299": ["TN299__MGDS_27339", "TN299__MGDS_31253"],
}
ALL_NATIVE = [p for v in FOLDS.values() for p in v]


# ---------------- model: HeteroscedasticUNet with 2-vec FiLM ----------------
def _gn(ch):
    g = 32
    while ch % g != 0:
        g //= 2
    return nn.GroupNorm(g, ch)


class _FiLM(nn.Module):
    def __init__(self, ch, scale_dim=2):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(scale_dim, ch), nn.SiLU(), nn.Linear(ch, 2 * ch))
        nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x, s):
        if s is None:
            return x
        s = s.to(self.mlp[0].weight.dtype)
        g, b = self.mlp(s).chunk(2, dim=-1)
        return x * (1 + g[:, :, None, None]) + b[:, :, None, None]


class _Block(nn.Module):
    def __init__(self, in_ch, out_ch, scale_dim=2):
        super().__init__()
        self.norm1 = _gn(in_ch); self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = _gn(out_ch); self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.film = _FiLM(out_ch, scale_dim)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, s):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.film(h, s)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class HeteroscedasticUNetNative(nn.Module):
    def __init__(self, base_ch=32, ch_mult=(1, 2, 4, 4), in_ch=1, scale_dim=2,
                 logvar_min=-10.0, logvar_max=6.0):
        super().__init__()
        self.logvar_min, self.logvar_max = logvar_min, logvar_max
        chs = [base_ch * m for m in ch_mult]
        self.in_conv = nn.Conv2d(in_ch, chs[0], 3, padding=1)
        self.enc = nn.ModuleList(); self.downs = nn.ModuleList()
        for i in range(len(chs)):
            self.enc.append(_Block(chs[i], chs[i], scale_dim))
            self.downs.append(nn.Conv2d(chs[i], chs[min(i + 1, len(chs) - 1)], 3, stride=2, padding=1)
                              if i < len(chs) - 1 else None)
        self.ups = nn.ModuleList(); self.dec = nn.ModuleList()
        for i in reversed(range(len(chs) - 1)):
            self.ups.append(nn.Conv2d(chs[i + 1], chs[i], 3, padding=1))
            self.dec.append(_Block(2 * chs[i], chs[i], scale_dim))
        self.out_norm = _gn(chs[0])
        self.head_mean = nn.Conv2d(chs[0], 1, 1); self.head_logvar = nn.Conv2d(chs[0], 1, 1)
        for h in (self.head_mean, self.head_logvar):
            nn.init.zeros_(h.weight); nn.init.zeros_(h.bias)

    def forward(self, lr_up, scale=None):
        h = self.in_conv(lr_up); skips = []
        for i, blk in enumerate(self.enc):
            h = blk(h, scale)
            if self.downs[i] is not None:
                skips.append(h); h = self.downs[i](h)
        for up, blk in zip(self.ups, self.dec):
            h = F.interpolate(h, scale_factor=2, mode='nearest'); h = up(h)
            h = blk(torch.cat([h, skips.pop()], dim=1), scale)
        h = F.silu(self.out_norm(h))
        mean = lr_up + self.head_mean(h)
        log_var = self.head_logvar(h).clamp(self.logvar_min, self.logvar_max)
        return mean, log_var


def gaussian_nll(mean, log_var, target, valid=None):
    nll = 0.5 * (log_var + (target - mean) ** 2 * torch.exp(-log_var))
    if valid is not None:
        return (nll * valid).sum() / valid.sum().clamp(min=1.0)
    return nll.mean()


# ---------------- data ----------------
class DS(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(self.files[i], allow_pickle=True)
        lr_up = torch.from_numpy(d['lr_up']).float()             # (1,H,W)
        hr = torch.from_numpy(d['hr']).float()
        hv = torch.from_numpy(d['hr_valid'].astype(np.float32))  # (1,H,W)
        valid = hv
        if 'lr_valid' in d.files:
            lv = torch.from_numpy(d['lr_valid'].squeeze().astype(np.float32))[None, None]
            lv_hr = F.interpolate(lv, size=hr.shape[-2:], mode='nearest')[0] > 0.5
            valid = hv * lv_hr.float()
        scale = torch.from_numpy(np.atleast_1d(d['scale']).astype(np.float32))
        return dict(lr_up=lr_up, hr=hr, valid=valid, scale=scale)


def files_for(pairs, patches_dir, cap):
    allf = []
    for split in ('train', 'val', 'test'):
        allf += glob.glob(os.path.join(patches_dir, split, '*.npz'))
    out = []
    for pid in pairs:
        fs = sorted([f for f in allf if b0.pid_of(f) == pid])
        if cap and len(fs) > cap:
            fs = [fs[i] for i in np.unique(np.linspace(0, len(fs) - 1, cap).round().astype(int))]
        out += fs
    return out


# ---------------- train / eval ----------------
def train_fold(train_files, steps, base_ch, batch, lr, device, workers):
    ds = DS(train_files)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True, num_workers=workers,
                    pin_memory=True, persistent_workers=(workers > 0),
                    prefetch_factor=(4 if workers > 0 else None))
    model = HeteroscedasticUNetNative(base_ch=base_ch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-6)
    model.train(); it = iter(dl); done = 0; run = 0.0
    while done < steps:
        try:
            batch_d = next(it)
        except StopIteration:
            it = iter(dl); batch_d = next(it)
        lr_up = batch_d['lr_up'].to(device); hr = batch_d['hr'].to(device)
        valid = batch_d['valid'].to(device); scale = batch_d['scale'].to(device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            mean, log_var = model(lr_up, scale)
            loss = gaussian_nll(mean, log_var, hr, valid)
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        run += loss.item(); done += 1
        if done % 500 == 0:
            print(f"      step {done}/{steps} nll={run/500:.4f}", flush=True); run = 0.0
    return model


@torch.no_grad()
def eval_fold(model, held_files, device, n_real=40):
    model.eval(); rec = []
    for f in held_files:
        d = np.load(f, allow_pickle=True)
        pb = b0.patch_bands(d); m = pb['valid']
        if m.sum() < 64 or pb['f_hi'] <= pb['f_lr']:
            continue
        lr_up = torch.from_numpy(d['lr_up']).float().to(device)[None]
        scale = torch.from_numpy(np.atleast_1d(d['scale']).astype(np.float32)).to(device)[None]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            mean, log_var = model(lr_up, scale)
        mean = mean.squeeze().float().cpu().numpy()
        sigma = np.exp(0.5 * log_var.squeeze().float().cpu().numpy())
        hr = pb['hr']; f_lo = pb['f_lr']
        C = b0.coarse_features(pb['lr_up'], m, f_lo)[0]
        # full-field Gaussian coverage
        vy = hr[m]; vmu = mean[m]; vs = np.clip(sigma[m], 1e-6, None)
        cc = np.zeros(len(ALPHAS))
        for i, a in enumerate(ALPHAS):
            z = norm.ppf((1 + a) / 2); cc[i] = np.sum(np.abs(vy - vmu) <= z * vs)
        # gaussian CRPS
        zz = (vy - vmu) / vs
        crps = float(np.mean(vs * (zz * (2 * norm.cdf(zz) - 1) + 2 * norm.pdf(zz) - 1 / np.sqrt(np.pi))))
        # fine-band sampled coverage (same protocol as §2a: white draws, highpass)
        rng = np.random.default_rng(0)
        reals = mean[None] + sigma[None] * rng.standard_normal((n_real, *mean.shape))
        fine = np.stack([b0.bandpass(reals[i], m, f_lo, 0.5) for i in range(n_real)])
        tgt = b0.bandpass(hr, m, f_lo, 0.5)
        fcc, _, frv, fvy = b0.coverage_counts(fine, tgt, m)
        rec.append(dict(pair=b0.pid_of(f), C=float(C), n_valid=int(m.sum()),
                        mean_sigma=float(vs.mean()),
                        resid_rms=float(np.sqrt(np.mean((vy - vmu) ** 2))),
                        hr_fine=float(np.sqrt(np.mean(tgt[m] ** 2))),
                        crps=crps,
                        **{f'ff_{i}': cc[i] for i in range(len(ALPHAS))},
                        **{f'fine_{i}': fcc[i] for i in range(len(ALPHAS))}))
    return rec


def r2_heldout(x, y):
    lx, ly = np.log10(np.array(x) + 1e-9), np.log10(np.array(y) + 1e-9)
    ss_res = np.sum((ly - (np.polyval(np.polyfit(lx, ly, 1), lx))) ** 2)
    ss_tot = np.sum((ly - ly.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--patches_dir', required=True)
    ap.add_argument('--out_dir', default='/scratch/users/scdobbs/step0_bakeoff/het_louo')
    ap.add_argument('--steps', type=int, default=8000)
    ap.add_argument('--base_ch', type=int, default=32)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--cap_per_pair', type=int, default=2500)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--only_fold', default='', help='run a single fold by key (e.g. NA090)')
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(get_device())
    folds = {args.only_fold: FOLDS[args.only_fold]} if args.only_fold else FOLDS

    allrec = []; fold_summ = {}
    for fi, (key, held) in enumerate(folds.items()):
        t0 = time.time()
        train_pairs = [p for p in ALL_NATIVE if p not in held]
        tf = files_for(train_pairs, args.patches_dir, args.cap_per_pair)
        hf = files_for(held, args.patches_dir, 200)
        print(f"\n>>> FOLD {key}: train {len(tf)} patches ({len(train_pairs)} pairs), "
              f"held {len(hf)} patches", flush=True)
        model = train_fold(tf, args.steps, args.base_ch, args.batch, args.lr, device, args.workers)
        rec = eval_fold(model, hf, device)
        for r in rec:
            r['fold'] = key
        allrec += rec
        dt = time.time() - t0
        # per-fold full-field + fine cal_err
        import pandas as pd
        sub = pd.DataFrame(rec)
        def curve(df, tag):
            n = df['n_valid'].sum()
            return [float(df[f'{tag}_{i}'].sum() / max(n, 1)) for i in range(len(ALPHAS))]
        def ce(c):
            return float(np.mean(np.abs(np.array(c) - ALPHAS)))
        ff, fine = curve(sub, 'ff'), curve(sub, 'fine')
        fold_summ[key] = dict(n_held=len(rec), gpu_min=round(dt / 60, 1),
                              fullfield_coverage=ff, fullfield_cal_err=ce(ff),
                              fine_coverage=fine, fine_cal_err=ce(fine),
                              crps=float(sub.crps.mean()),
                              mean_sigma=float(sub.mean_sigma.mean()),
                              resid_rms=float(sub.resid_rms.mean()),
                              hr_fine=float(sub.hr_fine.mean()))
        print(f"   [{key}] {dt/60:.1f} min | full-field cal_err={ce(ff):.3f} fine cal_err={ce(fine):.3f} "
              f"crps={sub.crps.mean():.3f} mean_sigma={sub.mean_sigma.mean():.3f} "
              f"resid_rms={sub.resid_rms.mean():.3f}", flush=True)
        # incremental save
        pd.DataFrame(allrec).to_parquet(out / 'het_louo_perpatch.parquet', index=False)
        big = pd.DataFrame(allrec)
        cross = dict(
            r2_sigma_vs_resid=r2_heldout(big.mean_sigma, big.resid_rms) if len(big) > 8 else None,
            r2_sigma_vs_hrfine=r2_heldout(big.mean_sigma, big.hr_fine) if len(big) > 8 else None,
            n_patches=int(len(big)), n_folds_done=len(fold_summ))
        (out / 'het_louo_summary.json').write_text(json.dumps(
            dict(folds=fold_summ, cross_pair=cross, steps=args.steps, base_ch=args.base_ch,
                 compare_to_regressor_r2=0.06), indent=2, default=str))
        print(f"   [saved] cross-pair R^2(sigma~resid)={cross['r2_sigma_vs_resid']} "
              f"R^2(sigma~hr_fine)={cross['r2_sigma_vs_hrfine']}", flush=True)

    print("\n============ §2b HET-LOUO SUMMARY ============")
    for k, s in fold_summ.items():
        print(f"  {k:10s} ff_cal_err={s['fullfield_cal_err']:.3f} fine_cal_err={s['fine_cal_err']:.3f} "
              f"crps={s['crps']:.3f} ({s['gpu_min']} min)")
    print("wrote", out / 'het_louo_summary.json')


if __name__ == '__main__':
    main()
