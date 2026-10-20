"""R02 §2 — het_clean_r02: the heteroscedastic CNN retrained on the CLEAN patch set.

Recipe = het_louo.py (base_ch 32, FiLM 2-vec scale, residual mean, zero-init heads, 8,000 steps, batch 32,
AdamW 2e-4 wd 1e-4, cosine to 1e-6, grad-clip 1.0, bf16, per-pair cap 2,500, no augmentation), with the
R02 §2.1 changes only:
  * inputs x (ship-normalized bicubic) + lr_valid_hr, in_ch = 2; target y (offset-removed, ship-normalized)
  * reflect padding on every conv
  * GroupNorm -> per-pixel LayerNorm over channels (no spatial statistics)
  * log-variance clamp [-14, 8]
  * standard Gaussian NLL, masked by hr_valid & lr_valid_hr
  * EW0207 excluded from every training fold; folds = 7 native units + one all-7 model (scores EW0207 only)
  * fixed seed, deterministic flags, every fold's weights saved
Isolation: outputs confined to /scratch/users/scdobbs/grl_review/R02/. Reads only the clean patch set.
"""
from __future__ import annotations
import argparse, glob, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PATCHES = "/scratch/users/scdobbs/grl_review/R02/patches_clean"
OUT = "/scratch/users/scdobbs/grl_review/R02/het_clean_r02"
FOLDS = {  # EW0207 excluded (CHANGE_LEDGER row 1); TN299 a/b one unit
    "NA090": ["NA090__MGDS_31212"], "NA080": ["NA080__MGDS_31290"], "TN399": ["TN399__MGDS_30373"],
    "RR1506": ["RR1506__MGDS_29779"], "AT37-13": ["AT37-13__MGDS_31199"], "FK181031": ["FK181031__MGDS_24618"],
    "TN299": ["TN299__MGDS_27339", "TN299__MGDS_31253"],
}
EW = ["EW0207__MGDS_32556"]
ALL7 = [p for v in FOLDS.values() for p in v]


# ---------------- model ----------------
class PixelLN(nn.Module):
    """LayerNorm over the channel axis at each pixel; no spatial pooling (R02 §2.0)."""
    def __init__(self, ch, eps=1e-5):
        super().__init__(); self.ln = nn.LayerNorm(ch, eps=eps)
    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class _FiLM(nn.Module):
    def __init__(self, ch, scale_dim=2):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(scale_dim, ch), nn.SiLU(), nn.Linear(ch, 2 * ch))
        nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)
    def forward(self, x, s):
        if s is None: return x
        s = s.to(self.mlp[0].weight.dtype)
        g, b = self.mlp(s).chunk(2, dim=-1)
        return x * (1 + g[:, :, None, None]) + b[:, :, None, None]


def conv(i, o, k=3, stride=1):
    return nn.Conv2d(i, o, k, stride=stride, padding=k // 2, padding_mode="reflect")


class _Block(nn.Module):
    def __init__(self, in_ch, out_ch, scale_dim=2):
        super().__init__()
        self.norm1 = PixelLN(in_ch); self.conv1 = conv(in_ch, out_ch)
        self.norm2 = PixelLN(out_ch); self.conv2 = conv(out_ch, out_ch)
        self.film = _FiLM(out_ch, scale_dim)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x, s):
        h = self.conv1(F.silu(self.norm1(x))); h = self.film(h, s)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class HetUNetClean(nn.Module):
    def __init__(self, base_ch=32, ch_mult=(1, 2, 4, 4), in_ch=2, scale_dim=2, logvar_min=-14.0, logvar_max=8.0):
        super().__init__()
        self.logvar_min, self.logvar_max = logvar_min, logvar_max
        chs = [base_ch * m for m in ch_mult]
        self.in_conv = conv(in_ch, chs[0])
        self.enc = nn.ModuleList(); self.downs = nn.ModuleList()
        for i in range(len(chs)):
            self.enc.append(_Block(chs[i], chs[i], scale_dim))
            self.downs.append(conv(chs[i], chs[min(i + 1, len(chs) - 1)], stride=2) if i < len(chs) - 1 else None)
        self.ups = nn.ModuleList(); self.dec = nn.ModuleList()
        for i in reversed(range(len(chs) - 1)):
            self.ups.append(conv(chs[i + 1], chs[i])); self.dec.append(_Block(2 * chs[i], chs[i], scale_dim))
        self.out_norm = PixelLN(chs[0])
        self.head_mean = nn.Conv2d(chs[0], 1, 1); self.head_logvar = nn.Conv2d(chs[0], 1, 1)
        for h in (self.head_mean, self.head_logvar):
            nn.init.zeros_(h.weight); nn.init.zeros_(h.bias)

    def forward(self, inp, scale=None):
        h = self.in_conv(inp); skips = []
        for i, blk in enumerate(self.enc):
            h = blk(h, scale)
            if self.downs[i] is not None:
                skips.append(h); h = self.downs[i](h)
        for up, blk in zip(self.ups, self.dec):
            h = F.interpolate(h, scale_factor=2, mode="nearest"); h = up(h)
            h = blk(torch.cat([h, skips.pop()], dim=1), scale)
        h = F.silu(self.out_norm(h))
        mean = inp[:, :1] + self.head_mean(h)                      # residual on the ship-normalized bicubic
        log_var = self.head_logvar(h).clamp(self.logvar_min, self.logvar_max)
        return mean, log_var


def gaussian_nll(mean, log_var, target, valid):
    nll = 0.5 * (log_var + (target - mean) ** 2 * torch.exp(-log_var))
    return (nll * valid).sum() / valid.sum().clamp(min=1.0)


# ---------------- data ----------------
def pid_of(f):
    return os.path.basename(f).rsplit("_r", 1)[0]


def files_for(pairs, patches_dir, cap):
    allf = []
    for split in ("train", "val", "test"):
        allf += glob.glob(os.path.join(patches_dir, split, "*.npz"))
    out = []
    for pid in pairs:
        fs = sorted([f for f in allf if pid_of(f) == pid])
        if cap and len(fs) > cap:
            fs = [fs[i] for i in np.unique(np.linspace(0, len(fs) - 1, cap).round().astype(int))]
        out += fs
    return out


class DS(Dataset):
    """Evaluation/training loader. Reads ONLY x, lr_valid_hr, scale for the forward pass; y and hr_valid
    for the loss. Never reads stats (mu_lr, s_lr, delta, HR mean/std) — asserted in load_sample."""
    FORWARD_KEYS = ("x", "lr_valid_hr", "scale")
    def __init__(self, files): self.files = files
    def __len__(self): return len(self.files)
    @staticmethod
    def load_sample(f):
        d = np.load(f, allow_pickle=True)
        inp = torch.cat([torch.from_numpy(d["x"]).float(), torch.from_numpy(d["lr_valid_hr"].astype(np.float32))], 0)
        scale = torch.from_numpy(np.atleast_1d(d["scale"]).astype(np.float32))
        y = torch.from_numpy(d["y"]).float()
        valid = torch.from_numpy((d["hr_valid"].astype(bool) & d["lr_valid_hr"].astype(bool)).astype(np.float32))
        return dict(inp=inp, scale=scale, y=y, valid=valid)
    def __getitem__(self, i): return self.load_sample(self.files[i])


# ---------------- train ----------------
def set_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def train_fold(train_files, steps, base_ch, batch, lr, device, workers, log_path, seed):
    set_seed(seed)
    ds = DS(train_files)
    gen = torch.Generator(); gen.manual_seed(seed)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True, num_workers=workers, pin_memory=True,
                    persistent_workers=(workers > 0), prefetch_factor=(4 if workers > 0 else None), generator=gen)
    model = HetUNetClean(base_ch=base_ch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-6)
    model.train(); it = iter(dl); done = 0; run = 0.0; curve = []
    while done < steps:
        try: b = next(it)
        except StopIteration: it = iter(dl); b = next(it)
        inp = b["inp"].to(device); y = b["y"].to(device); valid = b["valid"].to(device); s = b["scale"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mean, log_var = model(inp, s)
            loss = gaussian_nll(mean.float(), log_var.float(), y, valid)
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        run += loss.item(); done += 1
        if done % 100 == 0:
            curve.append((done, run / 100)); run = 0.0
            if done % 500 == 0:
                print(f"      step {done}/{steps} nll={curve[-1][1]:.4f}", flush=True)
    json.dump(curve, open(log_path, "w"))
    return model


@torch.no_grad()
def eval_fold(model, held_files, device):
    """Per-patch scalars + per-pixel maps (single-tile inference, the secondary column of R02 §3.1)."""
    model.eval(); rec = []; mu_maps = []; sg_maps = []; keys = []
    for f in held_files:
        d = np.load(f, allow_pickle=True)
        s = DS.load_sample(f)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mean, log_var = model(s["inp"][None].to(device), s["scale"][None].to(device))
        mu = mean.squeeze().float().cpu().numpy(); sg = np.exp(0.5 * log_var.squeeze().float().cpu().numpy())
        m = s["valid"].numpy().squeeze().astype(bool); y = s["y"].numpy().squeeze()
        st = np.asarray(d["stats"]).ravel()
        rec.append(dict(file=os.path.basename(f), pair=pid_of(f), n_valid=int(m.sum()), mu_lr=float(st[0]), s_lr=float(st[1]),
                        delta=float(st[2]), resid_rms_norm=float(np.sqrt(np.mean((y[m] - mu[m]) ** 2))) if m.any() else np.nan,
                        mean_sigma_norm=float(sg[m].mean()) if m.any() else np.nan,
                        frac_clamp_hi=float((sg[m] >= np.exp(0.5 * 8.0) * 0.999).mean()) if m.any() else np.nan,
                        frac_clamp_lo=float((sg[m] <= np.exp(0.5 * -14.0) * 1.001).mean()) if m.any() else np.nan))
        mu_maps.append(mu.astype(np.float16)); sg_maps.append(sg.astype(np.float16)); keys.append(os.path.basename(f))
    return rec, mu_maps, sg_maps, keys


def code_hash():
    return hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=OUT); ap.add_argument("--patches", default=PATCHES)
    ap.add_argument("--steps", type=int, default=8000); ap.add_argument("--base_ch", type=int, default=32)
    ap.add_argument("--batch", type=int, default=32); ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--cap_per_pair", type=int, default=2500); ap.add_argument("--held_cap", type=int, default=200)
    ap.add_argument("--workers", type=int, default=10); ap.add_argument("--seed", type=int, default=20260928)
    ap.add_argument("--only_fold", default="")
    args = ap.parse_args()
    out = Path(args.out_dir); assert str(out.resolve()).startswith("/scratch/users/scdobbs/grl_review/R02"), "isolation"
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    cfg = dict(vars(args), folds=list(FOLDS), excluded=EW, code_hash=code_hash(), torch=torch.__version__,
               recipe="het_louo.py + R02 §2.1 changes (reflect pad, PixelLN, in_ch 2, clamp [-14,8], NLL, seed)")
    json.dump(cfg, open(out / "config.json", "w"), indent=1)
    folds = dict(FOLDS); folds["ALL7"] = []                      # ALL7 = trained on all 7 units, scores EW0207
    if args.only_fold: folds = {args.only_fold: folds[args.only_fold]}
    summ = {}
    for key, held in folds.items():
        t0 = time.time(); fdir = out / f"fold_{key}"; fdir.mkdir(exist_ok=True)
        train_pairs = [p for p in ALL7 if p not in held]
        tf = files_for(train_pairs, args.patches, args.cap_per_pair)
        hf = files_for(held if held else EW, args.patches, args.held_cap)
        print(f"\n>>> FOLD {key}: train {len(tf)} files ({len(train_pairs)} pairs), held {len(hf)} ({held or EW})", flush=True)
        model = train_fold(tf, args.steps, args.base_ch, args.batch, args.lr, device, args.workers, fdir / "curve.json", args.seed)
        torch.save(model.state_dict(), fdir / "ckpt.pt")
        rec, mu, sg, keys = eval_fold(model, hf, device)
        np.savez_compressed(fdir / "perpixel_tile.npz", mu=np.stack(mu), sigma=np.stack(sg), key=np.array(keys))
        json.dump(rec, open(fdir / "perpatch_tile.json", "w"), indent=1)
        dt = time.time() - t0
        r = [x for x in rec if np.isfinite(x["resid_rms_norm"])]
        summ[key] = dict(n_train=len(tf), n_held=len(rec), gpu_min=round(dt / 60, 1),
                         resid_rms_norm_mean=float(np.mean([x["resid_rms_norm"] for x in r])),
                         mean_sigma_norm=float(np.mean([x["mean_sigma_norm"] for x in r])),
                         frac_clamp_hi=float(np.mean([x["frac_clamp_hi"] for x in r])),
                         frac_clamp_lo=float(np.mean([x["frac_clamp_lo"] for x in r])))
        print(f"   [{key}] {dt/60:.1f} min  {summ[key]}", flush=True)
        json.dump(summ, open(out / "fold_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
