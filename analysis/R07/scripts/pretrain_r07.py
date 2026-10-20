"""R07 §2 — self-supervised pretraining of one body (P-V1 or P-V2) on the ship-only window cache.
  objective (§2.2, one objective): L = MSE over the px of the masked LR cells between the reconstruction and the
           normalised bicubic field of the UNMASKED window (target in the unmasked window's standardised units)
         + MSE over covered 4 x 4 blocks between the spread head and standardised log10 ship_rsd (windows from
           product-usable blocks only). 30 % of the valid LR cells are marked missing before interpolation and enter
           exactly as missing LR does in development (filled with mu_lr, validity 0; mu_lr / s_lr / FiLM from the
           remaining cells). For P-V2 the ship_rsd channel and mask are zero on every window.
  schedule (§2.3): 60,000 steps, batch 64, AdamW 2e-4 / wd 1e-4, cosine to 1e-6, grad-clip 1.0, bf16 autocast, D4
           augmentation, seed 20261007; final checkpoint, no checkpoint choice. Every training window is unique
           (one presentation each; the mask of a window is fixed by its id).
  pretext report (§2.4): train / held-out curves of both terms; on the held-out cruises the reconstruction MSE against
           the mean-fill baseline, Spearman rho between predicted and actual log10 ship_rsd per block; a QC figure.
  pilot (§2.3): --steps 2000 --eval_steps 0,500,1000,2000; HOLD if either held-out term at 2,000 is not below step 0.
Outputs: pretrain/<body>/{ckpt_final.pt, body_<body>.pt, curves.json, eval.json, pretrain_log.json, qc_<body>.png}."""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r07_common as R7, r07_windows as W
from r07_pretext_model import PretextNet, n_params
from preprocessing import extract_clean_patches as ecp

# ---------------- cache dataset ----------------
class CacheDS(Dataset):
    """One item = one cached window assembled through the R06 path: masked input (+ V2 channels), unmasked target."""
    FIELDS = ("offsets", "lr_px", "hr_res", "cell", "usable", "n_valid", "rsd_tgt", "rsd_cov", "wid", "bin")
    def __init__(self, cache, split, products, drop_rsd, rsd_mean, rsd_sd, mask_tag, limit=None):
        self.cache, self.split, self.products, self.drop_rsd, self.mu, self.sd, self.mask_tag = cache, split, products, drop_rsd, rsd_mean, rsd_sd, mask_tag
        sh = pd.read_parquet(f"{cache}/shards.parquet"); sh = sh[sh.split == split].sort_values("shard")
        self.shards = sh.shard.tolist(); self.counts = sh.n.values.astype(int); self.cum = np.concatenate([[0], np.cumsum(self.counts)])
        self.n = int(self.cum[-1]) if limit is None else min(int(self.cum[-1]), limit); self._m = {}; self.max_open = 6   # bounded shard mmaps per worker (memory)
    def __len__(self): return self.n
    def _shard(self, s):
        if s not in self._m:
            if len(self._m) >= self.max_open: self._m.pop(next(iter(self._m)))
            p = f"{self.cache}/{self.split}/shard_{self.shards[s]:04d}"; d = {k: np.load(f"{p}/{k}.npy", mmap_mode="r") for k in self.FIELDS}
            d["lr"] = np.load(f"{p}/lr_flat.npy", mmap_mode="r"); d["lrv"] = np.load(f"{p}/lrv_flat.npy", mmap_mode="r"); d["prod"] = np.load(f"{p}/prod_flat.npy", mmap_mode="r"); self._m[s] = d
        return self._m[s]
    def meta(self, i):
        s = int(np.searchsorted(self.cum, i, side="right") - 1); j = i - self.cum[s]; d = self._shard(s); return s, j, d
    def __getitem__(self, i):
        s, j, d = self.meta(i); L = int(d["lr_px"][j]); a, b = int(d["offsets"][j]), int(d["offsets"][j + 1])
        lr = np.array(d["lr"][a:b]).reshape(L, L); lrv = np.array(d["lrv"][a:b]).reshape(L, L).astype(bool); usable = bool(d["usable"][j])
        prod_a = np.array(d["prod"][:, a:b]).reshape(len(W.PRODUCTS), L, L) if (self.products and usable) else None
        cell, hr = float(d["cell"][j]), float(d["hr_res"][j]); wid = int(d["wid"][j])
        mask = W.draw_mask(lrv, R7.MASK_FRAC, np.random.default_rng([R7.PRE_SEED, self.mask_tag, wid]))
        ot = W.assemble(lr, lrv, None, cell, hr, mask=None, products=False)                         # unmasked: the target
        om = W.assemble(lr, lrv, prod_a, cell, hr, mask=mask, products=self.products, drop_rsd=self.drop_rsd)   # masked: the input
        assert ot is not None and om is not None, (wid, L, int(lrv.sum()))
        maskmap = ecp.nearest_up(mask, W.WIN).astype(np.float32)
        base = ((om["lr_up_m"] - ot["mu_lr"]) / ot["s_lr"]).astype(np.float32)                     # mean-fill baseline in target units
        inp = np.stack([om["x"], om["lrv"].astype(np.float32)], 0)
        prod = np.concatenate([om["prod"], om["prodm"].astype(np.float32)], 0) if self.products else np.zeros((0,), np.float32)
        tgt = np.array(d["rsd_tgt"][j]); cov = np.array(d["rsd_cov"][j]) & usable
        rsd = np.where(cov, (np.nan_to_num(tgt, nan=0.0) - self.mu) / self.sd, 0.0).astype(np.float32)
        return (torch.from_numpy(inp), torch.from_numpy(om["cond"]), torch.from_numpy(ot["x"]), torch.from_numpy(maskmap), torch.from_numpy(base),
                torch.from_numpy(prod), torch.from_numpy(rsd), torch.from_numpy(cov.astype(np.float32)), torch.tensor(wid), torch.from_numpy(tgt.astype(np.float32)))

# ---------------- augmentation / loss ----------------
def d4(t, k):
    if k >= 4: t = t.flip(-1)
    return torch.rot90(t, k % 4, dims=(-2, -1))

def augment(tensors, gen):
    """The 8 square symmetries, one per window, applied identically to every spatial tensor (incl. the 4 x 4 maps)."""
    B = tensors[0].shape[0]; ks = torch.randint(0, 8, (B,), generator=gen); out = [t.clone() if t.dim() >= 3 else t for t in tensors]
    for k in range(1, 8):
        sel = (ks == k).to(tensors[0].device)
        if not sel.any(): continue
        for t in out:
            if t.dim() >= 3 and t.shape[-1] > 0: t[sel] = d4(t[sel], k)
    return out

def losses(recon, spread, target, maskmap, rsd, cov):
    l_rec = (((recon - target) ** 2) * maskmap).sum() / maskmap.sum().clamp(min=1.0)
    l_spr = (((spread.reshape(spread.shape[0], -1) - rsd) ** 2) * cov).sum() / cov.sum().clamp(min=1.0)
    return l_rec, l_spr

# ---------------- evaluation on the held-out cruises ----------------
@torch.no_grad()
def evaluate(model, ds, products, device, workers, batch=128, limit=None):
    model.eval(); n = len(ds) if limit is None else min(len(ds), limit)
    dl = DataLoader(torch.utils.data.Subset(ds, range(n)), batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True)
    se_rec = se_base = n_px = 0.0; se_spr = n_blk = 0.0; preds = []; acts = []; wids = []; per_win = []
    for inp, cond, target, maskmap, base, prod, rsd, cov, wid, tgt_raw in dl:
        inp, cond, target, maskmap, base, prod, rsd, cov = [t.to(device) for t in (inp, cond, target, maskmap, base, prod, rsd, cov)]
        with torch.autocast("cuda", dtype=torch.bfloat16): recon, spread = model(inp, cond, prod if products else None)
        recon = recon.float(); spread = spread.float().reshape(spread.shape[0], -1)
        se_rec += (((recon - target) ** 2) * maskmap).sum().item(); se_base += (((base - target) ** 2) * maskmap).sum().item(); n_px += maskmap.sum().item()
        se_spr += (((spread - rsd) ** 2) * cov).sum().item(); n_blk += cov.sum().item()
        c = cov.bool().cpu().numpy(); p = spread.cpu().numpy() * ds.sd + ds.mu; a = tgt_raw.numpy()
        preds.append(p[c]); acts.append(a[c])
        for k in range(len(wid)):
            if c[k].sum() >= 8 and np.std(a[k][c[k]]) > 0: per_win.append(spearmanr(p[k][c[k]], a[k][c[k]]).correlation)
    preds = np.concatenate(preds) if preds else np.zeros(0); acts = np.concatenate(acts) if acts else np.zeros(0)
    rho = float(spearmanr(preds, acts).correlation) if len(preds) > 2 else float("nan")
    model.train()
    return dict(n_windows=n, recon_mse_masked_px=se_rec / max(n_px, 1), baseline_meanfill_mse_masked_px=se_base / max(n_px, 1), recon_over_baseline=(se_rec / max(se_base, 1e-12)),
                spread_mse_std=se_spr / max(n_blk, 1), n_covered_blocks=int(n_blk), spread_rho_blocks=rho, spread_rho_within_window_median=(float(np.nanmedian(per_win)) if per_win else float("nan")),
                n_windows_with_8_covered_blocks=len(per_win), pooled_loss=(se_rec / max(n_px, 1) + se_spr / max(n_blk, 1)))

def qc_figure(model, ds, products, device, path, n=6, seed=1):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed); cand = [i for i in rng.permutation(min(len(ds), 4096))[:400] if bool(ds.meta(i)[2]["usable"][ds.meta(i)[1]])][:n]
    fig, ax = plt.subplots(n, 5, figsize=(16, 3.1 * n)); model.eval()
    for r, i in enumerate(cand):
        inp, cond, target, maskmap, base, prod, rsd, cov, wid, tgt_raw = ds[i]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            rec, spr = model(inp[None].to(device), cond[None].to(device), prod[None].to(device) if products else None)
        rec = rec.float()[0].cpu().numpy(); spr = spr.float()[0].cpu().numpy() * ds.sd + ds.mu; s, j, d = ds.meta(i)
        v = np.nanpercentile(target.numpy(), [1, 99])
        ax[r, 0].imshow(inp[0].numpy(), vmin=v[0], vmax=v[1], cmap="viridis"); ax[r, 0].set_title(f"wid {int(wid)}: masked input x\ncell {float(d['cell'][j]):.1f} m, hr_res {float(d['hr_res'][j]):.2f} m, lr_px {int(d['lr_px'][j])}", fontsize=8)
        ax[r, 1].imshow(maskmap.numpy(), cmap="gray"); ax[r, 1].set_title("masked cells (px)", fontsize=8)
        ax[r, 2].imshow(rec, vmin=v[0], vmax=v[1], cmap="viridis"); ax[r, 2].set_title("reconstruction", fontsize=8)
        ax[r, 3].imshow(target.numpy(), vmin=v[0], vmax=v[1], cmap="viridis"); ax[r, 3].set_title("target (unmasked bicubic, normalised)", fontsize=8)
        c = cov.numpy().astype(bool); a = tgt_raw.numpy()
        ax[r, 4].scatter(a[c], spr.reshape(-1)[c]); lo, hi = (np.nanmin(a[c]), np.nanmax(a[c])) if c.any() else (0, 1); ax[r, 4].plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax[r, 4].set_xlabel("actual log10 ship_rsd (m)", fontsize=8); ax[r, 4].set_ylabel("predicted", fontsize=8); ax[r, 4].set_title(f"spread: {int(c.sum())} covered blocks", fontsize=8)
        for a_ in ax[r, :4]: a_.set_xticks([]); a_.set_yticks([])
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig); model.train()

# ---------------- training ----------------
def set_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def main(a):
    body = a.body; products = R7.BODIES[body]["products"]; out = f"{a.out}/{body}" + ("_pilot" if a.pilot else ""); os.makedirs(out, exist_ok=True)
    man = json.load(open(f"{a.cache}/cache_manifest.json")); st = man["rsd_log10_standardisation"]
    set_seed(R7.PRE_SEED); device = torch.device("cuda")
    tr = CacheDS(a.cache, "train", products, drop_rsd=products, rsd_mean=st["mean"], rsd_sd=st["sd"], mask_tag=3)
    va = CacheDS(a.cache, "val", products, drop_rsd=products, rsd_mean=st["mean"], rsd_sd=st["sd"], mask_tag=4)
    n_use = a.steps * R7.PRE_BATCH; assert n_use <= len(tr), f"cache has {len(tr)} training windows < {n_use}"
    perm = np.random.default_rng([R7.PRE_SEED, 9]).permutation(len(tr))[:n_use]
    model = PretextNet(products=products).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=R7.PRE_LR, weight_decay=R7.PRE_WD); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps, eta_min=R7.PRE_ETA_MIN)
    gen = torch.Generator(); gen.manual_seed(R7.PRE_SEED + 1); eval_steps = set(a.eval_steps)
    log = lambda s: (print(s, flush=True), open(f"{out}/log.txt", "a").write(s + "\n"))
    log(f"[{body}] products={products} params {n_params(model):,} (body {n_params(model.body):,}{', branch ' + format(n_params(model.branch), ',') if products else ''}); train windows {len(tr)} (using {n_use}), val {len(va)}; GPU {torch.cuda.get_device_name(0)}; rsd std mean {st['mean']:.3f} sd {st['sd']:.3f}")
    curves = []; evals = {}; t0 = time.time(); run = np.zeros(2); done = 0; resume_from = 0; t_prev = 0.0
    rp = f"{out}/ckpt_resume.pt"
    if a.resume and os.path.exists(rp):   # robustness only: continue the same run (same permutation, generator state, optimiser, schedule) after a job failure
        ck = torch.load(rp, map_location="cpu"); model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); gen.set_state(ck["gen"])
        curves, evals, done, t_prev = ck["curves"], {int(k): v for k, v in ck["evals"].items()}, ck["step"], ck["wall_s"]; resume_from = done; run[:] = ck["run"]
        log(f"   RESUMED from step {done} ({rp}, saved {ck['time']}); continuing the same permutation from presentation {done * R7.PRE_BATCH}")
    dl = DataLoader(tr, batch_size=R7.PRE_BATCH, sampler=perm[done * R7.PRE_BATCH:].tolist(), num_workers=a.workers, pin_memory=True, prefetch_factor=2, drop_last=True, persistent_workers=True)
    def save_resume():
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), gen=gen.get_state(), curves=curves, evals=evals, step=done, run=run.copy(), wall_s=t_prev + time.time() - t0, time=time.strftime("%Y-%m-%dT%H:%M:%S")), rp + ".tmp"); os.replace(rp + ".tmp", rp)
    if 0 in eval_steps and resume_from == 0: evals[0] = evaluate(model, va, products, device, a.eval_workers, limit=a.eval_limit); log(f"   eval step 0: {json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in evals[0].items()})}")
    model.train()
    for inp, cond, target, maskmap, base, prod, rsd, cov, wid, _ in dl:
        inp, cond, target, maskmap, prod, rsd, cov = [t.to(device, non_blocking=True) for t in (inp, cond, target, maskmap, prod, rsd, cov)]
        inp, target, maskmap, prod, rsd4, cov4 = augment([inp, target, maskmap, prod, rsd.view(-1, 4, 4), cov.view(-1, 4, 4)], gen)
        rsd, cov = rsd4.reshape(rsd.shape[0], -1), cov4.reshape(cov.shape[0], -1)
        with torch.autocast("cuda", dtype=torch.bfloat16): recon, spread = model(inp, cond, prod if products else None)
        l_rec, l_spr = losses(recon.float(), spread.float(), target, maskmap, rsd, cov); loss = l_rec + l_spr
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), R7.PRE_GRAD_CLIP); opt.step(); sched.step()
        run += (l_rec.item(), l_spr.item()); done += 1
        if done % 100 == 0:
            curves.append(dict(step=done, recon=run[0] / 100, spread=run[1] / 100, total=run.sum() / 100, lr=sched.get_last_lr()[0], wall_s=time.time() - t0)); run[:] = 0
            if done % 1000 == 0: log(f"      step {done}/{a.steps} recon={curves[-1]['recon']:.4f} spread={curves[-1]['spread']:.4f} ({(time.time()-t0)/done:.3f} s/step)")
        if done in eval_steps or (a.eval_every and done % a.eval_every == 0 and done != a.steps):
            evals[done] = evaluate(model, va, products, device, a.eval_workers, limit=a.eval_limit); log(f"   eval step {done}: {json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in evals[done].items()})}")
            json.dump(dict(curves=curves, evals=evals), open(f"{out}/curves.json", "w"), indent=1)
        if a.ckpt_every and done % a.ckpt_every == 0 and done != a.steps: save_resume()
    wall = t_prev + time.time() - t0
    evals[done] = evaluate(model, va, products, device, a.eval_workers)         # final: the whole held-out set
    log(f"   final eval step {done} (all {len(va)} held-out windows): {json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in evals[done].items()})}")
    json.dump(dict(curves=curves, evals=evals), open(f"{out}/curves.json", "w"), indent=1); json.dump(evals, open(f"{out}/eval.json", "w"), indent=1)
    torch.save(dict(state_dict=model.state_dict(), body=body, products=products, steps=done, rsd_std=st, code_hash=R7.code_hash()), f"{out}/ckpt_final.pt")
    torch.save(dict(body_state_dict=model.body_state_dict(), body=body, products=products, steps=done, rsd_std=st, code_hash=R7.code_hash(), source=f"{out}/ckpt_final.pt"), f"{out}/body_{body}.pt")
    try: qc_figure(model, va, products, device, f"{out}/qc_{body}.png")
    except Exception as e: log(f"QC figure failed: {type(e).__name__}: {e}")
    rec = dict(body=body, products=products, pilot=a.pilot, steps=done, resumed_from_step=resume_from, workers=a.workers, batch=R7.PRE_BATCH, lr=R7.PRE_LR, weight_decay=R7.PRE_WD, eta_min=R7.PRE_ETA_MIN, grad_clip=R7.PRE_GRAD_CLIP, precision="bf16", optimizer="AdamW", schedule="cosine",
               seed=R7.PRE_SEED, mask_frac=R7.MASK_FRAC, drop_rsd_input=products, n_train_windows_cache=len(tr), n_train_windows_used=n_use, n_val_windows=len(va), params=n_params(model), params_body=n_params(model.body),
               params_branch=(n_params(model.branch) if products else 0), gpu=torch.cuda.get_device_name(0), wall_train_s=wall, s_per_step=wall / max(done, 1), gpu_h=wall / 3600, final_train_100=curves[-1] if curves else None,
               final_eval=evals[done], rsd_std=st, torch=torch.__version__, host=os.uname().nodename, slurm_job=os.environ.get("SLURM_JOB_ID"), code_hash=R7.code_hash(), git_head=R7.git_head(), cache_manifest_created=man["created"])
    if a.pilot:
        e0, e1 = evals[0], evals[done]
        rec["pilot_rule"] = dict(recon_below_step0=bool(e1["recon_mse_masked_px"] < e0["recon_mse_masked_px"]), spread_below_step0=bool(e1["spread_mse_std"] < e0["spread_mse_std"]),
                                 recon_step0=e0["recon_mse_masked_px"], recon_final=e1["recon_mse_masked_px"], spread_step0=e0["spread_mse_std"], spread_final=e1["spread_mse_std"])
        rec["pilot_rule"]["verdict"] = "PASS" if rec["pilot_rule"]["recon_below_step0"] and rec["pilot_rule"]["spread_below_step0"] else "HOLD: a held-out term at 2,000 steps is not below its step-0 value"
        log(f"PILOT RULE: {rec['pilot_rule']['verdict']} (recon {e0['recon_mse_masked_px']:.4f} -> {e1['recon_mse_masked_px']:.4f}; spread {e0['spread_mse_std']:.4f} -> {e1['spread_mse_std']:.4f})")
    json.dump(rec, open(f"{out}/pretrain_log.json", "w"), indent=1)
    log(f"done {body}: {done} steps in {wall/3600:.2f} h ({rec['s_per_step']:.3f} s/step) -> {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--body", required=True, choices=list(R7.BODIES)); ap.add_argument("--steps", type=int, default=R7.PRE_STEPS)
    ap.add_argument("--eval_steps", default=None, help="comma-separated steps (default: 0 and every --eval_every)"); ap.add_argument("--eval_every", type=int, default=R7.EVAL_EVERY)
    ap.add_argument("--eval_limit", type=int, default=4096, help="held-out windows for the periodic curve points (the final eval uses all)"); ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--workers", type=int, default=16); ap.add_argument("--eval_workers", type=int, default=8); ap.add_argument("--cache", default=R7.CACHE); ap.add_argument("--out", default=R7.PRETRAIN)
    ap.add_argument("--ckpt_every", type=int, default=5000, help="resume checkpoint interval (0 = none); robustness only"); ap.add_argument("--resume", action="store_true", help="continue from <out>/ckpt_resume.pt if present"); a = ap.parse_args()
    a.eval_steps = [int(s) for s in a.eval_steps.split(",")] if a.eval_steps else [0]
    if a.pilot: a.steps = R7.PILOT_STEPS; a.eval_steps = list(R7.PILOT_EVAL_STEPS); a.eval_every = 0
    R7.start(f"pretrain_r07_{a.body}{'_pilot' if a.pilot else ''}")
    assert os.path.exists(f"{a.cache}/DONE"), "cache not built"
    main(a)
