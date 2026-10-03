"""R06 §1.4-1.5, §2 — train ONE model (variant, seed, fold) on the frozen target and predict the held-out unit.
  fold = a development unit id (leave-one-unit-out: the unit never enters training) or 'ALL' (all 16 development units;
         the model predicts EW0207 only, §3.5). EW0207 is never trained on.
  sampler  unit-balanced (§1.4): unit ~ uniform over training units, pair ~ uniform within the unit, 8 windows ~ uniform
           within the pair (with replacement only if the pair has < 8 windows) = one group; a batch is 4 groups.
  aug      the 8 symmetries of the square, drawn per window, applied identically to every input channel and the 4 x 4
           target / kept maps.
  schedule R02 recipe: 8,000 steps, AdamW 2e-4 wd 1e-4, cosine to 1e-6, grad-clip 1.0, bf16 autocast, fixed seed; the
           final checkpoint is scored (no early stopping, no checkpoint choice).
  loss     V0: MSE on log10 A_fine over kept blocks. V1-V3: within each group, MSE between (pred - group mean of pred)
           and (target - group mean of target) over the group's kept blocks.
  V2       ship-product branch; channel dropout p = 0.3 per product (value and mask zeroed together) during training.
  V3       input cropped to the central 128 px before the network.
Outputs: models/<variant>/seed<seed>/fold_<fold>/{ckpt.pt, curve.json, train_log.json, pred.parquet}."""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r06_common as R, r04_common as C4
from r06_model import BlockHeadNet, n_params

# ---------------- data ----------------
class PairStore:
    def __init__(self, pid, products):
        self.d = f"{R.INPUTS}/{pid}"; self.meta = pd.read_parquet(f"{self.d}/meta.parquet"); self.n = len(self.meta)
        self.cond = self.meta[["cond0", "cond1", "cond2", "cond3"]].values.astype(np.float32)
        self.y = np.load(f"{self.d}/y.npy"); self.kept = np.load(f"{self.d}/kept.npy")
        self.usable = bool(self.meta.usable.iloc[0]); self.products = products; self._x = self._lv = self._pr = self._pm = None
    def _open(self):
        self._x = np.load(f"{self.d}/x.npy", mmap_mode="r"); self._lv = np.load(f"{self.d}/lrv.npy", mmap_mode="r")
        if self.products and self.usable: self._pr = np.load(f"{self.d}/prod.npy", mmap_mode="r"); self._pm = np.load(f"{self.d}/prodm.npy", mmap_mode="r")
    def get(self, i):
        if self._x is None: self._open()
        inp = np.stack([np.asarray(self._x[i]), np.asarray(self._lv[i]).astype(np.float32)], 0)
        if self.products:
            prod = np.concatenate([np.asarray(self._pr[i]), np.asarray(self._pm[i]).astype(np.float32)], 0) if self.usable else np.zeros((2 * len(R.PRODUCTS), R.WIN, R.WIN), np.float32)
        else: prod = np.zeros((0,), np.float32)
        return inp, self.cond[i], np.nan_to_num(self.y[i], nan=0.0), self.kept[i], prod

class WindowDS(Dataset):
    def __init__(self, pairs, products):
        self.stores = {p: PairStore(p, products) for p in pairs}; self.index = []; self.offset = {}; self.n_of = {}
        for p in pairs:
            self.offset[p] = len(self.index); self.n_of[p] = self.stores[p].n; self.index += [(p, i) for i in range(self.stores[p].n)]
    def __len__(self): return len(self.index)
    def __getitem__(self, g):
        p, i = self.index[g]; inp, cond, y, kept, prod = self.stores[p].get(i)
        return torch.from_numpy(inp), torch.from_numpy(cond), torch.from_numpy(y), torch.from_numpy(kept), torch.from_numpy(prod)

class GroupSampler(Sampler):
    """§1.4 unit-balanced groups; yields lists of global indices, GROUPS_PER_BATCH x WINDOWS_PER_GROUP long."""
    def __init__(self, ds, unit_pairs, n_steps, seed):
        self.ds, self.units, self.unit_pairs, self.n_steps, self.seed = ds, sorted(unit_pairs), unit_pairs, n_steps, seed
    def __len__(self): return self.n_steps
    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        for _ in range(self.n_steps):
            batch = []
            for _ in range(R.GROUPS_PER_BATCH):
                u = self.units[rng.integers(len(self.units))]; ps = self.unit_pairs[u]; p = ps[rng.integers(len(ps))]; n = self.ds.n_of[p]
                idx = rng.choice(n, R.WINDOWS_PER_GROUP, replace=(n < R.WINDOWS_PER_GROUP)); batch += [self.ds.offset[p] + int(j) for j in idx]
            yield batch

# ---------------- augmentation / loss ----------------
def d4(t, k):
    """k in 0..7: rotation by 90 k (mod 4) degrees after a horizontal flip if k >= 4; acts on the last two axes."""
    if k >= 4: t = t.flip(-1)
    return torch.rot90(t, k % 4, dims=(-2, -1))

def augment(inp, y, kept, prod, gen):
    B = inp.shape[0]; ks = torch.randint(0, 8, (B,), generator=gen); y = y.view(B, R.NB, R.NB).clone(); kept = kept.view(B, R.NB, R.NB).clone()
    inp = inp.clone(); prod = prod.clone() if prod.dim() == 4 else prod
    for k in range(1, 8):
        sel = (ks == k).to(inp.device)
        if not sel.any(): continue
        inp[sel] = d4(inp[sel], k); y[sel] = d4(y[sel], k); kept[sel] = d4(kept[sel], k)
        if prod.dim() == 4: prod[sel] = d4(prod[sel], k)
    return inp, y.reshape(B, -1), kept.reshape(B, -1), prod, ks

def product_dropout(prod, p, gen):
    """Zero a product's value AND mask channel together with probability p, per window and product."""
    B, Cc = prod.shape[:2]; nprod = Cc // 2
    keep = (torch.rand(B, nprod, generator=gen) >= p).to(prod.device, prod.dtype)
    return prod * torch.cat([keep, keep], 1)[:, :, None, None]

def loss_fn(pred, y, kept, centred):
    k = kept.float(); pred = pred.reshape(pred.shape[0], -1)
    if not centred: return (((pred - y) ** 2) * k).sum() / k.sum().clamp(min=1.0)
    G = pred.shape[0] // R.WINDOWS_PER_GROUP
    p = pred.reshape(G, -1); t = y.reshape(G, -1); kk = k.reshape(G, -1); cnt = kk.sum(1, keepdim=True).clamp(min=1.0)
    pm = (p * kk).sum(1, keepdim=True) / cnt; tm = (t * kk).sum(1, keepdim=True) / cnt
    return ((((p - pm) - (t - tm)) ** 2) * kk).sum() / kk.sum().clamp(min=1.0)

def crop_tile(t):
    return t[..., R.MARGIN:R.MARGIN + R.TILE, R.MARGIN:R.MARGIN + R.TILE] if t.dim() == 4 else t

# ---------------- train / predict ----------------
def set_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def train(variant, fold, seed, steps, workers, out, log=print, timing_only=0, smoke_units=None):
    V = R.VARIANTS[variant]; units = R.fold_units(fold); unit_pairs = {u: list(C4.UNITS[u]) for u in units}
    if smoke_units:   # smoke test only: restrict training to the units whose inputs exist
        units = [u for u in units if u in smoke_units]; unit_pairs = {u: [p for p in unit_pairs[u] if os.path.exists(f"{R.INPUTS}/{p}/meta.parquet")] for u in units}
    assert "lu16" not in units and not any(p in C4.EW0207["lu16"] for ps in unit_pairs.values() for p in ps), "EW0207 must never be trained on"
    C4.assert_no_sealed([p for ps in unit_pairs.values() for p in ps])
    set_seed(seed); device = torch.device("cuda"); ds = WindowDS([p for u in units for p in unit_pairs[u]], V["products"])
    sampler = GroupSampler(ds, unit_pairs, steps if not timing_only else timing_only, seed)
    dl = DataLoader(ds, batch_sampler=sampler, num_workers=workers, pin_memory=True, prefetch_factor=(4 if workers else None), persistent_workers=False)
    model = BlockHeadNet(R.BASE_CH, R.CH_MULT, 2, 4, products=V["products"], crop=V["crop"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=R.LR, weight_decay=R.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=R.ETA_MIN)
    gen = torch.Generator(); gen.manual_seed(seed + 1)
    log(f"  [{variant} seed {seed} fold {fold}] train units {len(units)}, pairs {len(ds.stores)}, windows {len(ds)}; params {n_params(model):,}; GPU {torch.cuda.get_device_name(0)}")
    model.train(); curve = []; run = 0.0; t0 = time.time(); done = 0; kept_frac = 0.0
    for inp, cond, y, kept, prod in dl:
        inp, cond, y, kept, prod = inp.to(device, non_blocking=True), cond.to(device), y.to(device), kept.to(device), prod.to(device, non_blocking=True)
        inp, y, kept, prod, _ = augment(inp, y, kept, prod, gen)
        if V["products"]: prod = product_dropout(prod, R.PRODUCT_DROPOUT, gen)
        if V["crop"]: inp = crop_tile(inp); prod = crop_tile(prod)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(inp, cond, prod if V["products"] else None)
        loss = loss_fn(pred.float(), y, kept, V["loss"] == "centred")
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), R.GRAD_CLIP); opt.step(); sched.step()
        run += loss.item(); done += 1; kept_frac += kept.float().mean().item()
        if done % 100 == 0:
            curve.append((done, run / 100, time.time() - t0)); run = 0.0
            if done % 1000 == 0 or timing_only: log(f"      step {done}/{steps} loss={curve[-1][1]:.4f} ({(time.time()-t0)/done:.3f} s/step)")
    wall = time.time() - t0
    if timing_only: return dict(variant=variant, steps=done, s_per_step=wall / max(done, 1), gpu=torch.cuda.get_device_name(0), params=n_params(model))
    os.makedirs(out, exist_ok=True); json.dump(curve, open(f"{out}/curve.json", "w"))
    torch.save(dict(state_dict=model.state_dict(), variant=variant, fold=fold, seed=seed, steps=steps, config=dict(base_ch=R.BASE_CH, ch_mult=R.CH_MULT, products=V["products"], crop=V["crop"])), f"{out}/ckpt.pt")
    return model, dict(variant=variant, fold=fold, seed=seed, steps=steps, batch=R.GROUPS_PER_BATCH * R.WINDOWS_PER_GROUP, groups_per_batch=R.GROUPS_PER_BATCH, windows_per_group=R.WINDOWS_PER_GROUP,
                       lr=R.LR, weight_decay=R.WEIGHT_DECAY, eta_min=R.ETA_MIN, grad_clip=R.GRAD_CLIP, precision="bf16", optimizer="AdamW", schedule="cosine", product_dropout=(R.PRODUCT_DROPOUT if V["products"] else None),
                       train_units=units, n_train_pairs=len(ds.stores), n_train_windows=len(ds), params=n_params(model), gpu=torch.cuda.get_device_name(0), wall_train_s=wall, s_per_step=wall / steps,
                       final_loss_100=curve[-1][1] if curve else None, mean_kept_frac=kept_frac / max(done, 1), torch=torch.__version__, host=os.uname().nodename, slurm_job=os.environ.get("SLURM_JOB_ID"))

@torch.no_grad()
def predict(model, variant, pairs, workers=4, batch=64):
    V = R.VARIANTS[variant]; device = next(model.parameters()).device; model.eval(); ds = WindowDS(pairs, V["products"]); rows = []
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True)
    preds = []
    for inp, cond, y, kept, prod in dl:
        inp, cond, prod = inp.to(device), cond.to(device), prod.to(device)
        if V["crop"]: inp = crop_tile(inp); prod = crop_tile(prod)
        with torch.autocast("cuda", dtype=torch.bfloat16): p = model(inp, cond, prod if V["products"] else None)
        preds.append(p.float().reshape(p.shape[0], -1).cpu().numpy())
    P = np.concatenate(preds, 0); assert P.shape == (len(ds), R.NB * R.NB)
    meta = pd.concat([ds.stores[p].meta[["tile_id", "unit", "pair"]] for p in pairs], ignore_index=True)
    out = meta.copy()
    for j in range(R.NB * R.NB): out[f"blk{j}"] = P[:, j]                      # log10 A_fine per block
    out["tile_pred_m"] = np.sqrt(np.mean(10 ** (2 * P), axis=1))             # §1.1 RMS over the 16 blocks, equal weights
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--variant", required=True, choices=list(R.VARIANTS)); ap.add_argument("--fold", required=True); ap.add_argument("--seed", type=int, default=R.SEED_MAIN)
    ap.add_argument("--steps", type=int, default=R.STEPS); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--out", default=R.MODELS); ap.add_argument("--timing_only", type=int, default=0)
    ap.add_argument("--force", action="store_true"); ap.add_argument("--smoke_units", default=None, help="smoke test only: comma-separated units; inputs dir from R06_INPUTS"); a = ap.parse_args()
    if os.environ.get("R06_INPUTS"): R.INPUTS = os.environ["R06_INPUTS"]
    m = R.start(f"train_r06_{a.variant}_{a.fold}_{a.seed}")
    out = f"{a.out}/{a.variant}/seed{a.seed}/fold_{a.fold}"
    if a.timing_only:
        r = train(a.variant, a.fold, a.seed, a.steps, a.workers, out, timing_only=a.timing_only, smoke_units=(a.smoke_units.split(",") if a.smoke_units else None)); print("TIMING", json.dumps(r), flush=True); sys.exit(0)
    if os.path.exists(f"{out}/pred.parquet") and not a.force: print(f"exists, skipping: {out}", flush=True); sys.exit(0)
    t0 = time.time(); model, rec = train(a.variant, a.fold, a.seed, a.steps, a.workers, out, smoke_units=(a.smoke_units.split(",") if a.smoke_units else None))
    held = C4.EW0207["lu16"] if a.fold == "ALL" else list(C4.UNITS[a.fold]); t1 = time.time()
    pred = predict(model, a.variant, held, workers=min(a.workers, 4)); pred.to_parquet(f"{out}/pred.parquet")
    rec.update(held_pairs=held, n_held_windows=len(pred), wall_infer_s=time.time() - t1, wall_total_s=time.time() - t0, code_hash=R.code_hash(), git_head=R.git_head())
    json.dump(rec, open(f"{out}/train_log.json", "w"), indent=1)
    print(f"  done {a.variant} seed {a.seed} fold {a.fold}: train {rec['wall_train_s']/60:.1f} min, infer {rec['wall_infer_s']:.0f} s, {len(pred)} held windows, final loss {rec['final_loss_100']}", flush=True)
