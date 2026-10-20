"""R07 §3 — fine-tuning: the phase2-r1 training code (train_r06.py) with ONE added option, --init, that loads initial
body (and, for V2, product-branch) weights from a pretrained checkpoint before training. Everything else is the R06
code verbatim: inputs (R06 input arrays), unit-balanced group sampler, D4 augmentation, within-pair centred loss,
8,000 steps, AdamW 2e-4 / wd 1e-4 / cosine to 1e-6 / grad-clip 1.0 / bf16, final checkpoint, product dropout 0.3 (V2).
  --arm P-V1 : variant V1 + --init <P-V1 body>      --arm P-V2 : variant V2 + --init <P-V2 body>
  --arm V2   : variant V2, init none (scratch seed completion)     --arm V1 : variant V1, init none (§3.3 equivalence test only)
Outputs: models/<arm>/seed<seed>/fold_<fold>/{ckpt.pt, curve.json, train_log.json, pred.parquet} (R06 layout)."""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r07_common as R7, r06_common as R, r04_common as C4
from r06_model import BlockHeadNet, n_params

# ---------------- data (train_r06.py, unchanged) ----------------
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

# ---------------- augmentation / loss (train_r06.py, unchanged) ----------------
def d4(t, k):
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

# ---------------- the one addition: initial weights ----------------
def load_init(model, init_path):
    """Load body.* (and branch.*) tensors from a pretrained checkpoint into a fresh BlockHeadNet; the head keeps its
    zero-initialised R06 weights. Every body / branch tensor must be present and no unexpected key may remain."""
    ck = torch.load(init_path, map_location="cpu"); sd = ck["body_state_dict"]
    assert all(k.startswith(("body.", "branch.")) for k in sd), [k for k in sd if not k.startswith(("body.", "branch."))]
    res = model.load_state_dict(sd, strict=False)
    assert not res.unexpected_keys, res.unexpected_keys
    assert all(k.startswith("head.") for k in res.missing_keys), res.missing_keys
    n_loaded = sum(v.numel() for v in sd.values())
    return dict(path=init_path, sha256=R7.sha256_file(init_path), body=ck.get("body"), pretrain_steps=ck.get("steps"), n_tensors=len(sd), n_params_loaded=n_loaded, head_keys_fresh=sorted(res.missing_keys))

# ---------------- train / predict (train_r06.py; + init) ----------------
def set_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def train(variant, fold, seed, steps, workers, out, init=None, log=print, timing_only=0):
    V = R.VARIANTS[variant]; units = R.fold_units(fold); unit_pairs = {u: list(C4.UNITS[u]) for u in units}
    assert "lu16" not in units and not any(p in C4.EW0207["lu16"] for ps in unit_pairs.values() for p in ps), "EW0207 must never be trained on"
    C4.assert_no_sealed([p for ps in unit_pairs.values() for p in ps])
    set_seed(seed); device = torch.device("cuda"); ds = WindowDS([p for u in units for p in unit_pairs[u]], V["products"])
    sampler = GroupSampler(ds, unit_pairs, steps if not timing_only else timing_only, seed)
    dl = DataLoader(ds, batch_sampler=sampler, num_workers=workers, pin_memory=True, prefetch_factor=(4 if workers else None), persistent_workers=False)
    model = BlockHeadNet(R.BASE_CH, R.CH_MULT, 2, 4, products=V["products"], crop=V["crop"]).to(device)
    init_rec = None
    if init: init_rec = load_init(model, init); model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=R.LR, weight_decay=R.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=R.ETA_MIN)
    gen = torch.Generator(); gen.manual_seed(seed + 1)
    log(f"  [{variant} seed {seed} fold {fold} init {'none' if not init else os.path.basename(init)}] train units {len(units)}, pairs {len(ds.stores)}, windows {len(ds)}; params {n_params(model):,}; GPU {torch.cuda.get_device_name(0)}")
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
    torch.save(dict(state_dict=model.state_dict(), variant=variant, fold=fold, seed=seed, steps=steps, init=init_rec, config=dict(base_ch=R.BASE_CH, ch_mult=R.CH_MULT, products=V["products"], crop=V["crop"])), f"{out}/ckpt.pt")
    return model, dict(variant=variant, fold=fold, seed=seed, steps=steps, batch=R.GROUPS_PER_BATCH * R.WINDOWS_PER_GROUP, groups_per_batch=R.GROUPS_PER_BATCH, windows_per_group=R.WINDOWS_PER_GROUP,
                       lr=R.LR, weight_decay=R.WEIGHT_DECAY, eta_min=R.ETA_MIN, grad_clip=R.GRAD_CLIP, precision="bf16", optimizer="AdamW", schedule="cosine", product_dropout=(R.PRODUCT_DROPOUT if V["products"] else None),
                       init=init_rec, train_units=units, n_train_pairs=len(ds.stores), n_train_windows=len(ds), params=n_params(model), gpu=torch.cuda.get_device_name(0), wall_train_s=wall, s_per_step=wall / steps,
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
    for j in range(R.NB * R.NB): out[f"blk{j}"] = P[:, j]
    out["tile_pred_m"] = np.sqrt(np.mean(10 ** (2 * P), axis=1))
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--arm", required=True, choices=["P-V1", "P-V2", "V2", "V1"]); ap.add_argument("--fold", required=True); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--init", default=None, help="pretrained body checkpoint (required for P-V1 / P-V2, forbidden otherwise)")
    ap.add_argument("--steps", type=int, default=R.STEPS); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--out", default=R7.MODELS); ap.add_argument("--timing_only", type=int, default=0); ap.add_argument("--force", action="store_true"); a = ap.parse_args()
    variant = {"P-V1": "V1", "P-V2": "V2", "V2": "V2", "V1": "V1"}[a.arm]
    if a.arm.startswith("P-"): assert a.init, f"{a.arm} needs --init"
    else: assert not a.init, f"{a.arm} is a scratch arm: no --init"
    m = R7.start(f"train_r07_{a.arm}_{a.fold}_{a.seed}", check_corpus_files=False)
    out = f"{a.out}/{a.arm}/seed{a.seed}/fold_{a.fold}"
    if a.timing_only:
        r = train(variant, a.fold, a.seed, a.steps, a.workers, out, init=a.init, timing_only=a.timing_only); print("TIMING", json.dumps(r), flush=True); sys.exit(0)
    if os.path.exists(f"{out}/pred.parquet") and not a.force: print(f"exists, skipping: {out}", flush=True); sys.exit(0)
    t0 = time.time(); model, rec = train(variant, a.fold, a.seed, a.steps, a.workers, out, init=a.init)
    held = C4.EW0207["lu16"] if a.fold == "ALL" else list(C4.UNITS[a.fold]); t1 = time.time()
    pred = predict(model, variant, held, workers=min(a.workers, 4)); pred.to_parquet(f"{out}/pred.parquet")
    rec.update(arm=a.arm, held_pairs=held, n_held_windows=len(pred), wall_infer_s=time.time() - t1, wall_total_s=time.time() - t0, code_hash=R7.code_hash(), r06_training_code_hash=R.code_hash(), git_head=R7.git_head())
    json.dump(rec, open(f"{out}/train_log.json", "w"), indent=1)
    print(f"  done {a.arm} seed {a.seed} fold {a.fold}: train {rec['wall_train_s']/60:.1f} min, infer {rec['wall_infer_s']:.0f} s, {len(pred)} held windows, final loss {rec['final_loss_100']}", flush=True)
