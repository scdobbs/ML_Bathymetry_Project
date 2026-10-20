"""R06 §1.2 — model inputs for every tiles_v1.1 window (16 development units + EW0207), ship side only, through the
UNCHANGED R02 path (extract_clean_patches.tile_ship_side + build_inputs, the representation recorded in R04 §7):
  x           = (bicubic(area-averaged LR, invalid cells <- mu_lr) - mu_lr) / s_lr      256 x 256 float32
  lrv         = LR validity nearest-upsampled to HR px                                   256 x 256 uint8
  cond        = [log2 lr_native, log2 hr_res, log2 s_lr, log2 |mu_lr|]                   FiLM vector (§1.2)
  prod / prodm (V2 only, pairs with ship_products_usable and not overridden): the 4 contract products area-averaged onto
                the window's lr_px grid, normalized (ship_rsd -> log10 max(rsd, 1 mm); ship_count -> log10 count;
                ship_xtrack_frac as is; ship_beam_angle / 90), unavailable cells <- mean of the available cells, bicubic to
                HR px; prodm = per-product availability mask nearest-upsampled (contract §6: never fill without the mask).
  y / kept    = log10 A_fine of the 16 blocks (blocks_v1.1; NaN where not kept) and the kept mask            (labels)
Per pair: inputs/<pair>/{x,lrv,y,kept[,prod,prodm]}.npy (row i = tiles_v1.1 row i of the pair) + meta.parquet.
Checks: mu_lr / s_lr equal the frozen tiles_v1.1 values and the central 128 px of the bicubic equal the R04 bicubic_m.tif
mosaic on every window; R04 field mosaics verified against the OAK SHA256SUMS; leak test (§0.2 rule 3): 100 windows
recomputed with the HR grid, HR validity and joint mask replaced by noise must give byte-identical x, lrv, cond, prod, prodm."""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
import numpy as np, pandas as pd, rasterio
from multiprocessing import Pool
from rasterio.enums import Resampling
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r06_common as R, r04_common as C4, features_v1_1 as F11
from preprocessing import extract_clean_patches as ecp
from preprocessing.colocated_utils import reproject_from_array
from preprocessing.data_utils import bicubic_upsample

S_FLOOR = 0.5
G = {}

def norm_product(name, a):
    if name == "ship_rsd": return np.log10(np.maximum(a, 1e-3))
    if name == "ship_count": return np.log10(a)
    if name == "ship_xtrack_frac": return a
    if name == "ship_beam_angle": return a / 90.0
    raise KeyError(name)

def load_products(row, g):
    """Contract v2.1 products on the lr.tif grid (asserted), with the R05 consumer-side override applied (NaN everywhere)."""
    if not C4.ship_usable(row): return None
    d = C4.products_dir(row); P = {}
    for p in R.PRODUCTS:
        with rasterio.open(os.path.join(d, f"{p}.tif")) as s:
            assert s.shape == g["lr_arr"].shape and s.transform == g["lr_tr"], f"{p}: products grid != lr.tif grid"
            P[p] = s.read(1).astype(np.float32)
    for (pid, prod), why in F11.OVERRIDES.items():
        if pid == str(row["pair_id"]): P[prod] = np.full_like(P[prod], np.nan); print(f"    override {pid}/{prod}: {why}", flush=True)
    return P

def window_inputs(g, P, r0, c0):
    """SHIP SIDE ONLY: never touches g['hr_arr'], g['hrv_arr'] or g['joint']."""
    lr, lr_valid = ecp.tile_ship_side(g, r0, c0, R.WIN)
    inp = ecp.build_inputs(lr, lr_valid, R.WIN, S_FLOOR)
    assert inp is not None, "window rejected on lr_valid: not a tiles_v1.1 window"
    cond = np.array([np.log2(g["lr_native"]), np.log2(g["hr_res"]), np.log2(inp["s_lr"]), np.log2(abs(inp["mu_lr"]))], np.float32)
    out = dict(x=inp["x"], lrv=inp["lr_valid_hr"].astype(np.uint8), mu_lr=inp["mu_lr"], s_lr=inp["s_lr"], n_lr_valid=inp["n_lr_valid"], cond=cond,
               bic=inp["lr_up_m"][R.MARGIN:R.MARGIN + R.TILE, R.MARGIN:R.MARGIN + R.TILE], prod=None, prodm=None)
    if P is not None:
        b = rasterio.windows.bounds(rasterio.windows.Window(c0, r0, R.WIN, R.WIN), g["transform"])
        prod = np.zeros((len(R.PRODUCTS), R.WIN, R.WIN), np.float32); prodm = np.zeros((len(R.PRODUCTS), R.WIN, R.WIN), np.uint8)
        for j, p in enumerate(R.PRODUCTS):
            a = reproject_from_array(P[p], g["lr_tr"], g["lr_crs"], b, g["lr_px"], g["dst_crs"], Resampling.average)
            avail = np.isfinite(a) & ((a > 0) if p == "ship_count" else True)
            if avail.any():
                v = norm_product(p, np.where(avail, a, 1.0)); fill = float(v[avail].mean())
                prod[j] = bicubic_upsample(np.where(avail, v, fill).astype(np.float32), R.WIN); prodm[j] = ecp.nearest_up(avail, R.WIN).astype(np.uint8)
        out["prod"], out["prodm"] = prod, prodm
    return out

def work(job):
    i, r0, c0 = job; o = window_inputs(G["g"], G["P"], r0, c0); o["i"] = i; return o

def process_pair(m, pid, tiles, blocks, workers, out_root):
    t0 = time.time(); row = C4.pair_row(m, pid); unit = C4.UNIT_OF[pid]
    T = tiles[tiles.pair == pid].reset_index(drop=True); n = len(T); assert n > 0, pid
    g = ecp.pair_setup(C4.DATA, row, R.WIN); P = load_products(row, g); usable = P is not None
    G.clear(); G.update(g=g, P=P)
    d = f"{out_root}/{pid}"; os.makedirs(d, exist_ok=True)
    X = np.lib.format.open_memmap(f"{d}/x.npy", "w+", np.float32, (n, R.WIN, R.WIN)); LV = np.lib.format.open_memmap(f"{d}/lrv.npy", "w+", np.uint8, (n, R.WIN, R.WIN))
    PR = np.lib.format.open_memmap(f"{d}/prod.npy", "w+", np.float32, (n, len(R.PRODUCTS), R.WIN, R.WIN)) if usable else None
    PM = np.lib.format.open_memmap(f"{d}/prodm.npy", "w+", np.uint8, (n, len(R.PRODUCTS), R.WIN, R.WIN)) if usable else None
    mu = np.zeros(n); s = np.zeros(n); nlv = np.zeros(n, int); cond = np.zeros((n, 4), np.float32); bic_max_diff = np.zeros(n); mask_frac = np.zeros((n, len(R.PRODUCTS)))
    with rasterio.open(f"{R.FIELDS}/{pid}/bicubic_m.tif") as src: BIC = src.read(1)   # R04 frozen mosaic (central 128 of every window)
    jobs = [(i, int(r.r0), int(r.c0)) for i, r in enumerate(T.itertuples())]
    with Pool(workers) as pool:
        for o in pool.imap_unordered(work, jobs, chunksize=4):
            i = o["i"]; X[i] = o["x"]; LV[i] = o["lrv"]; mu[i] = o["mu_lr"]; s[i] = o["s_lr"]; nlv[i] = o["n_lr_valid"]; cond[i] = o["cond"]
            tr0, tc0 = int(T.tr0[i]), int(T.tc0[i]); ref = BIC[tr0:tr0 + R.TILE, tc0:tc0 + R.TILE]; ok = np.isfinite(ref)
            bic_max_diff[i] = float(np.max(np.abs(ref[ok] - o["bic"][ok]))) if ok.any() else np.nan
            if usable: PR[i] = o["prod"]; PM[i] = o["prodm"]; mask_frac[i] = o["prodm"].reshape(len(R.PRODUCTS), -1).mean(1)
    X.flush(); LV.flush()
    if usable: PR.flush(); PM.flush()
    # labels: log10 A_fine per block, kept mask, in block_id order
    B = blocks[blocks.pair == pid].set_index(["tile_id", "block_id"])
    y = np.full((n, R.NB * R.NB), np.nan, np.float32); kept = np.zeros((n, R.NB * R.NB), bool)
    for i, tid in enumerate(T.tile_id):
        bb = B.loc[tid].sort_index(); assert list(bb.index) == list(range(16)), tid
        k = bb.kept.values.astype(bool); kept[i] = k; y[i, k] = np.log10(np.maximum(bb.A_fine.values[k], R.LOG10_FLOOR_M))
    np.save(f"{d}/y.npy", y); np.save(f"{d}/kept.npy", kept)
    meta = T[["tile_id", "unit", "pair", "r0", "c0", "tr0", "tc0", "lr_native", "hr_res", "k", "A_fine", "T_fine", "E_1m"]].copy()
    meta["mu_lr"] = mu; meta["s_lr"] = s; meta["n_lr_valid"] = nlv; meta["usable"] = usable
    for j in range(4): meta[f"cond{j}"] = cond[:, j]
    for j, p in enumerate(R.PRODUCTS): meta[f"maskfrac_{p}"] = mask_frac[:, j] if usable else np.nan
    meta.to_parquet(f"{d}/meta.parquet")
    chk = dict(pair=pid, unit=unit, n_windows=n, usable=usable, lr_px=int(g["lr_px"]), hr_res=float(g["hr_res"]), lr_native=float(g["lr_native"]),
               mu_lr_max_abs_diff=float(np.max(np.abs(mu - T.mu_lr.values))), s_lr_max_abs_diff=float(np.max(np.abs(s - T.s_lr.values))),
               bicubic_max_abs_diff=float(np.nanmax(bic_max_diff)), n_lr_valid_mismatch=int(np.sum(nlv != T.n_lr_valid.values)),
               kept_blocks=int(kept.sum()), product_mask_frac_mean={p: float(mask_frac[:, j].mean()) for j, p in enumerate(R.PRODUCTS)} if usable else None,
               product_windows_any={p: float((mask_frac[:, j] > 0).mean()) for j, p in enumerate(R.PRODUCTS)} if usable else None, seconds=time.time() - t0)
    print(f"  {pid} ({unit}): {n} windows, usable={usable}, lr_px={g['lr_px']}, |dmu|max={chk['mu_lr_max_abs_diff']:.2e} |ds|max={chk['s_lr_max_abs_diff']:.2e} |dbicubic|max={chk['bicubic_max_abs_diff']:.2e} ({chk['seconds']:.0f} s)", flush=True)
    G.clear(); del g
    return chk

def hb(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def leak_test(m, tiles, out_root, n=100, seed=0):
    """§0.2 rule 3: recompute n windows with HR / HR-validity / joint replaced by noise; inputs must be byte-identical
    to the stored arrays (which were computed from the true grids)."""
    rng = np.random.default_rng(seed); sample = tiles.sample(n=min(n, len(tiles)), random_state=seed); res = []
    for pid, sub in sample.groupby("pair"):
        row = C4.pair_row(m, pid); g = ecp.pair_setup(C4.DATA, row, R.WIN); P = load_products(row, g)
        g2 = dict(g); g2["hr_arr"] = np.where(np.isfinite(g["hr_arr"]), rng.standard_normal(g["hr_arr"].shape).astype(np.float32) * 100 - 2000, np.nan)
        g2["hrv_arr"] = rng.random(g["hrv_arr"].shape) > 0.3; g2["joint"] = (rng.random(g["joint"].shape) > 0.3).astype(np.uint8)
        d = f"{out_root}/{pid}"; meta = pd.read_parquet(f"{d}/meta.parquet"); X = np.load(f"{d}/x.npy", mmap_mode="r"); LV = np.load(f"{d}/lrv.npy", mmap_mode="r")
        PR = np.load(f"{d}/prod.npy", mmap_mode="r") if P is not None else None; PM = np.load(f"{d}/prodm.npy", mmap_mode="r") if P is not None else None
        pos = {t: i for i, t in enumerate(meta.tile_id)}
        for r in sub.itertuples():
            i = pos[r.tile_id]; o = window_inputs(g2, P, int(r.r0), int(r.c0))
            same = hb(o["x"]) == hb(X[i]) and hb(o["lrv"]) == hb(LV[i]) and hb(o["cond"]) == hb(meta[["cond0", "cond1", "cond2", "cond3"]].values[i].astype(np.float32))
            if P is not None: same = same and hb(o["prod"]) == hb(PR[i]) and hb(o["prodm"]) == hb(PM[i])
            res.append(dict(pair=pid, tile_id=r.tile_id, usable=P is not None, identical=bool(same)))
        del g, g2
    return res

def verify_fields():
    """R04 field mosaics (F_hr, F_res, bicubic_m, joint_v1 per pair) against the OAK SHA256SUMS written at preservation."""
    sums = f"{R.OAK_ARCHIVE}/R04/fields/SHA256SUMS"; out = []
    for line in open(sums):
        h, _, name = line.rstrip("\n").partition("  "); p = os.path.join(R.FIELDS, name[2:])
        if any(s in p for s in C4.SEALED): continue
        out.append(dict(file=name, ok=(R.sha256_file(p) == h)))
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--pairs", default=None); ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    ap.add_argument("--out", default=R.INPUTS); ap.add_argument("--leak", type=int, default=100); ap.add_argument("--skip_fields", action="store_true"); a = ap.parse_args()
    m = R.start("build_inputs_r06"); os.makedirs(a.out, exist_ok=True)
    v = C4.verify_unit_list(m); assert v["verified"], v
    tiles = pd.read_parquet(R.TILES); blocks = pd.read_parquet(R.BLOCKS)
    pairs = a.pairs.split(",") if a.pairs else C4.DEV_PAIRS + C4.EW0207["lu16"]; C4.assert_no_sealed(pairs)
    fields = None
    if not a.skip_fields:
        t0 = time.time(); fields = verify_fields(); nbad = sum(not f["ok"] for f in fields)
        print(f"R04 field mosaics vs OAK SHA256SUMS: {len(fields) - nbad}/{len(fields)} verified ({time.time()-t0:.0f} s)", flush=True)
        if nbad: raise SystemExit("STOP (directive §0.1): R04 field mosaics differ from the OAK archive")
    checks = [process_pair(m, pid, tiles, blocks, a.workers, a.out) for pid in pairs]
    summ = dict(pairs=checks, n_windows=int(sum(c["n_windows"] for c in checks)), fields_verified=(None if fields is None else len(fields)),
                product_norm={"ship_rsd": "log10(max(rsd, 1e-3 m))", "ship_count": "log10(count)", "ship_xtrack_frac": "as is", "ship_beam_angle": "deg / 90"},
                availability_per_unit={})
    for c in checks:
        u = c["unit"]; summ["availability_per_unit"].setdefault(u, dict(pairs=0, usable_pairs=0, windows=0, usable_windows=0))
        s = summ["availability_per_unit"][u]; s["pairs"] += 1; s["windows"] += c["n_windows"]
        if c["usable"]: s["usable_pairs"] += 1; s["usable_windows"] += c["n_windows"]; s[f"mask_frac_mean_{c['pair']}"] = c["product_mask_frac_mean"]
    json.dump(summ, open(f"{a.out}/inputs_summary.json", "w"), indent=1, default=float)
    worst = max(c["mu_lr_max_abs_diff"] for c in checks), max(c["s_lr_max_abs_diff"] for c in checks), max(c["bicubic_max_abs_diff"] for c in checks)
    print(f"consistency with frozen tiles_v1.1 / R04 bicubic mosaics: max |dmu_lr| {worst[0]:.2e}, max |ds_lr| {worst[1]:.2e}, max |dbicubic| {worst[2]:.2e}", flush=True)
    if a.leak:
        res = leak_test(m, tiles[tiles.pair.isin(pairs)], a.out, n=a.leak); n_ok = sum(r["identical"] for r in res)
        json.dump(dict(n=len(res), n_identical=n_ok, n_usable=sum(r["usable"] for r in res), rows=res), open(f"{a.out}/leak_test.json", "w"), indent=1)
        print(f"INPUT LEAK TEST: {n_ok}/{len(res)} windows byte-identical with HR, HR validity and joint mask replaced by noise", flush=True)
        if n_ok != len(res): raise SystemExit("INPUT LEAK TEST FAILED")
    print("done", flush=True)
