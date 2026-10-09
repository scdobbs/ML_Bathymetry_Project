"""R07 §1 — the pretraining window cache (§1.5): blocks used (§1.1), cell-size bins and the pre-registered sampler
(§1.2: bin uniform -> block uniform within the bin -> position uniform within the block; hr_res drawn by the R06
development sampler, unit uniform then pair uniform), the held-out cruises (§1.3: 5 % of corpus cruises stratified by
instrument family, default_rng(20261007)), windows built by the R06 path (§1.4, r07_windows), and the sub-cell
sounding-spread target (§2.2 item 2) with its training-set standardisation.
Cache layout (shards = groups of whole blocks; a modest number of large files):
  cache/<split>/shard_NNNN/{lr_flat.npy, lrv_flat.npy, prod_flat.npy, offsets.npy, lr_px.npy, hr_res.npy, cell.npy,
                            block.npy, r0.npy, c0.npy, bin.npy, usable.npy, n_valid.npy, rsd_tgt.npy, rsd_cov.npy, wid.npy}
  cache/cache_manifest.json (counts, bins, held-out cruises, rsd standardisation, ledger), cache/shards.parquet,
  cache/blocks_used.parquet, cache/heldout_cruises.json, cache/ledger.csv (every dropped window with its reason).
One unique window per training presentation: N_train = 60,000 steps x 64 (§2.3); N_val = 16,384 held-out windows."""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r07_common as R7, r04_common as C4, r07_windows as W
from preprocessing import extract_clean_patches as ecp

WIN = R7.WIN; MAX_POS_TRIES = 30
EDGES = np.log2(R7.CELL_MIN_M) + np.arange(R7.N_BINS + 1) * (np.log2(R7.CELL_MAX_M) - np.log2(R7.CELL_MIN_M)) / R7.N_BINS

def bin_of(cell): return np.clip(np.floor((np.log2(cell) - EDGES[0]) / (EDGES[1] - EDGES[0])).astype(int), 0, R7.N_BINS - 1)

def block_dir(row): return f"/vsitar/{R7.CORPUS}/blocks/{row.cruise}.tar/{row.tar_member_dir}"

def choose_heldout(blocks):
    """§1.3: 5 % of corpus cruises, stratified by instrument family (>= 1 per family), default_rng(20261007)."""
    rng = np.random.default_rng(R7.HOLDOUT_SEED); held = {}
    fam = blocks.groupby("instrument_family").cruise.unique()
    for f in sorted(fam.index):
        cr = sorted(fam[f]); k = max(1, int(round(R7.HOLDOUT_FRAC * len(cr))))
        held[f] = sorted(rng.choice(cr, k, replace=False).tolist())
    return held

def dev_hr_res_table():
    """pair -> hr_res of the development pairs (tiles_v1.1), and the unit -> pairs lists of units-v1 (R06 sampler domain)."""
    T = pd.read_parquet(R7.R6.TILES, columns=["unit", "pair", "hr_res"]); hr = T.groupby("pair").hr_res.first()
    units = sorted(C4.UNITS); pairs = {u: list(C4.UNITS[u]) for u in units}
    for u in units:
        for p in pairs[u]: assert p in hr.index, p
    return hr.to_dict(), units, pairs

def draw_windows(n, blocks_by_bin, hr_of, units, pairs, seed_tag):
    """Pre-registered sampler (§1.2 + §1.4 hr_res): returns (bin, block row index, hr_res) per window."""
    rng = np.random.default_rng([R7.PRE_SEED, seed_tag]); avail = np.array([k for k in range(R7.N_BINS) if len(blocks_by_bin[k])])
    b = avail[rng.integers(len(avail), size=n)]; blk = np.empty(n, np.int64)         # bins with no block in this split are skipped (reported)
    for k in avail:
        idx = np.flatnonzero(b == k); cand = blocks_by_bin[k]; blk[idx] = cand[rng.integers(len(cand), size=len(idx))]
    u = rng.integers(len(units), size=n); hr = np.empty(n, np.float64)
    for j in range(n):
        ps = pairs[units[u[j]]]; hr[j] = hr_of[ps[rng.integers(len(ps))]]
    return b, blk, hr

G = {}
def process_block(job):
    """One block: all its windows (positions drawn per window with default_rng([PRE_SEED, split_tag, wid]))."""
    bi, wids, hrs, split_tag = job; row = G["blocks"].loc[bi]; d = block_dir(row); out = []; ledger = []
    try:
        geoms = {}; P = None; P_loaded = False
        for wid, hr in zip(wids, hrs):
            hr = float(hr)
            if hr not in geoms: geoms[hr] = W.block_geom(f"{d}/grid.tif", float(row.cell_m), hr)
            g = geoms[hr]
            if not P_loaded: P = W.load_block_products(d, g, bool(row.products_usable)); P_loaded = True
            rng = np.random.default_rng([R7.PRE_SEED, split_tag, int(wid)]); ok = False
            for _ in range(MAX_POS_TRIES):
                r0 = int(rng.integers(0, g["H"] - WIN + 1)); c0 = int(rng.integers(0, g["W"] - WIN + 1))
                lr, lv = ecp.tile_ship_side(g, r0, c0, WIN); nv = int(lv.sum())
                if nv / lv.size >= R7.MIN_VALID_FRAC and nv >= 3 and nv - W.n_mask_of(nv) >= 2: ok = True; break
            if not ok: ledger.append(dict(wid=int(wid), block=int(bi), cruise=row.cruise, hr_res=hr, reason=f"no valid position in {MAX_POS_TRIES} tries (fill {row.fill:.3f})")); continue
            t = W.extract_lr_tile(g, P, r0, c0)
            if P is not None: tgt, cov = W.rsd_block_target(P["ship_rsd"], g["lr_tr"], g["transform"], r0, c0)
            else: tgt, cov = np.full((4, 4), np.nan, np.float32), np.zeros((4, 4), bool)
            prod_a = t["prod_a"] if t["prod_a"] is not None else np.full((len(R7.PRODUCTS), g["lr_px"], g["lr_px"]), np.nan, np.float32)
            out.append(dict(wid=int(wid), block=int(bi), lr=t["lr"].astype(np.float32).ravel(), lrv=t["lr_valid"].astype(np.uint8).ravel(), prod=prod_a.astype(np.float32).reshape(len(R7.PRODUCTS), -1),
                            lr_px=int(g["lr_px"]), hr_res=hr, cell=float(row.cell_m), r0=r0, c0=c0, bin=int(row.bin), usable=bool(P is not None), n_valid=nv, rsd_tgt=tgt.ravel(), rsd_cov=cov.ravel()))
    except Exception as e:
        for wid, hr in zip(wids, hrs): ledger.append(dict(wid=int(wid), block=int(bi), cruise=row.cruise, hr_res=float(hr), reason=f"block failed: {type(e).__name__}: {e}"))
        out = []
    return out, ledger

def write_shard(path, recs):
    os.makedirs(path, exist_ok=True); recs = sorted(recs, key=lambda r: r["wid"])
    sizes = np.array([r["lr"].size for r in recs], np.int64); off = np.concatenate([[0], np.cumsum(sizes)])
    np.save(f"{path}/lr_flat.npy", np.concatenate([r["lr"] for r in recs])); np.save(f"{path}/lrv_flat.npy", np.concatenate([r["lrv"] for r in recs]))
    np.save(f"{path}/prod_flat.npy", np.concatenate([r["prod"] for r in recs], axis=1)); np.save(f"{path}/offsets.npy", off)
    for k, dt in (("lr_px", np.int16), ("hr_res", np.float32), ("cell", np.float32), ("block", np.int32), ("r0", np.int32), ("c0", np.int32), ("bin", np.int8), ("usable", bool), ("n_valid", np.int32), ("wid", np.int64)):
        np.save(f"{path}/{k}.npy", np.array([r[k] for r in recs], dt))
    np.save(f"{path}/rsd_tgt.npy", np.stack([r["rsd_tgt"] for r in recs]).astype(np.float32)); np.save(f"{path}/rsd_cov.npy", np.stack([r["rsd_cov"] for r in recs]).astype(bool))
    return dict(n=len(recs), cells=int(off[-1]), bytes=sum(os.path.getsize(f"{path}/{f}") for f in os.listdir(path)))

def build_split(name, split_tag, n, blocks, blocks_by_bin, hr_of, units, pairs, workers, log):
    b, blk, hr = draw_windows(n, blocks_by_bin, hr_of, units, pairs, split_tag)
    order = np.argsort(blk, kind="stable"); blk_s, hr_s, wid_s = blk[order], hr[order], order
    ub, starts = np.unique(blk_s, return_index=True); ends = np.append(starts[1:], len(blk_s))
    jobs = [(int(bi), wid_s[s:e], hr_s[s:e], split_tag) for bi, s, e in zip(ub, starts, ends)]
    # shards = contiguous groups of whole blocks with about WINDOWS_PER_SHARD windows each
    shards = []; cur = []; cnt = 0
    for j in jobs:
        cur.append(j); cnt += len(j[1])
        if cnt >= R7.WINDOWS_PER_SHARD: shards.append(cur); cur = []; cnt = 0
    if cur: shards.append(cur)
    log(f"[{name}] {n} windows over {len(jobs)} blocks in {len(shards)} shards; bins: {np.bincount(b, minlength=R7.N_BINS).tolist()}")
    G.clear(); G["blocks"] = blocks; shard_rows = []; ledger = []; t0 = time.time(); done = 0
    with Pool(workers) as pool:
        for si, sh in enumerate(shards):
            recs = []
            for out, led in pool.imap_unordered(process_block, sh, chunksize=1): recs += out; ledger += led
            info = write_shard(f"{R7.CACHE}/{name}/shard_{si:04d}", recs); info.update(shard=si, split=name, n_blocks=len(sh)); shard_rows.append(info); done += info["n"]
            log(f"   [{name}] shard {si:04d}: {info['n']} windows from {len(sh)} blocks, {info['bytes']/1e6:.0f} MB; {done}/{n} done, {len(ledger)} dropped ({time.time()-t0:.0f} s)")
    return dict(n_drawn=n, n_built=done, shards=shard_rows, ledger=ledger, bins_drawn=np.bincount(b, minlength=R7.N_BINS).tolist())

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n_train", type=int, default=R7.N_TRAIN_WINDOWS); ap.add_argument("--n_val", type=int, default=R7.N_VAL_WINDOWS)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8))); ap.add_argument("--out", default=R7.CACHE); a = ap.parse_args()
    R7.CACHE = a.out; m = R7.start("build_cache_r07"); os.makedirs(R7.CACHE, exist_ok=True); t0 = time.time()
    logf = open(f"{R7.CACHE}/build_log.txt", "a")
    def log(s): print(s, flush=True); logf.write(s + "\n"); logf.flush()
    M = pd.read_parquet(R7.CORPUS_MANIFEST); M["bin"] = -1
    # §1.1 blocks used
    in_range = (M.cell_m >= R7.CELL_MIN_M) & (M.cell_m <= R7.CELL_MAX_M)
    counts = dict(corpus_blocks=len(M), corpus_cruises=int(M.cruise.nunique()), excluded_cell_below_10m=int((M.cell_m < R7.CELL_MIN_M).sum()), excluded_cell_above_256m=int((M.cell_m > R7.CELL_MAX_M).sum()),
                  reconstruction_task_blocks=int(in_range.sum()), spread_task_blocks=int((in_range & M.products_usable).sum()), cruises_in_range=int(M[in_range].cruise.nunique()))
    blocks = M[in_range].copy(); blocks["bin"] = bin_of(blocks.cell_m.values)
    # §1.3 held-out cruises
    held = choose_heldout(blocks); held_set = set(c for v in held.values() for c in v); blocks["heldout"] = blocks.cruise.isin(held_set)
    json.dump(dict(seed=R7.HOLDOUT_SEED, frac=R7.HOLDOUT_FRAC, stratified_by="instrument_family", by_family=held, cruises=sorted(held_set),
                   n_cruises=len(held_set), n_blocks=int(blocks.heldout.sum()), family_cruise_counts={f: int(n) for f, n in blocks.groupby("instrument_family").cruise.nunique().items()}), open(f"{R7.CACHE}/heldout_cruises.json", "w"), indent=1)
    blocks.drop(columns=[c for c in ("products_available", "qa_shift_argmin_cells", "grid_vsitar") if c in blocks]).to_parquet(f"{R7.CACHE}/blocks_used.parquet")
    per_bin = pd.DataFrame(dict(bin=range(R7.N_BINS), cell_lo_m=2 ** EDGES[:-1], cell_hi_m=2 ** EDGES[1:],
                                blocks=blocks.groupby("bin").size().reindex(range(R7.N_BINS), fill_value=0).values,
                                blocks_train=blocks[~blocks.heldout].groupby("bin").size().reindex(range(R7.N_BINS), fill_value=0).values,
                                blocks_heldout=blocks[blocks.heldout].groupby("bin").size().reindex(range(R7.N_BINS), fill_value=0).values,
                                blocks_usable=blocks[blocks.products_usable].groupby("bin").size().reindex(range(R7.N_BINS), fill_value=0).values,
                                cruises=blocks.groupby("bin").cruise.nunique().reindex(range(R7.N_BINS), fill_value=0).values))
    log(f"blocks: {counts}\nheld-out cruises: {len(held_set)} ({sum(len(v) for v in held.values())} by family {dict((f, len(v)) for f, v in held.items())}), {int(blocks.heldout.sum())} blocks\n{per_bin.to_string()}")
    assert (per_bin.blocks_train > 0).all(), "a bin has no training block"
    if not (per_bin.blocks_heldout > 0).all(): log(f"NOTE: bins without a held-out block (skipped in the held-out draw): {per_bin.bin[per_bin.blocks_heldout == 0].tolist()}")
    hr_of, units, pairs = dev_hr_res_table()
    tr = blocks[~blocks.heldout]; va = blocks[blocks.heldout]
    bb_tr = {k: tr.index[tr.bin == k].values for k in range(R7.N_BINS)}; bb_va = {k: va.index[va.bin == k].values for k in range(R7.N_BINS)}
    val = build_split("val", 1, a.n_val, blocks, bb_va, hr_of, units, pairs, a.workers, log)
    train = build_split("train", 0, a.n_train, blocks, bb_tr, hr_of, units, pairs, a.workers, log)
    # training-set standardisation of the spread target (§2.2 item 2), from the train shards
    vals = []
    for s in train["shards"]:
        p = f"{R7.CACHE}/train/shard_{s['shard']:04d}"; tg = np.load(f"{p}/rsd_tgt.npy"); cv = np.load(f"{p}/rsd_cov.npy"); vals.append(tg[cv])
    vals = np.concatenate(vals); rsd_stats = dict(mean=float(vals.mean()), sd=float(vals.std()), n_blocks=int(len(vals)), p01=float(np.percentile(vals, 1)), p99=float(np.percentile(vals, 99)))
    # effective share of training windows per bin (built windows) and per-bin usable share
    bins_built = np.zeros(R7.N_BINS, int); usable_built = np.zeros(R7.N_BINS, int); cov_blocks = 0
    for s in train["shards"]:
        p = f"{R7.CACHE}/train/shard_{s['shard']:04d}"; bn = np.load(f"{p}/bin.npy"); us = np.load(f"{p}/usable.npy")
        bins_built += np.bincount(bn, minlength=R7.N_BINS); usable_built += np.bincount(bn[us], minlength=R7.N_BINS); cov_blocks += int(np.load(f"{p}/rsd_cov.npy").sum())
    per_bin["train_windows_built"] = bins_built; per_bin["train_windows_share"] = bins_built / max(1, bins_built.sum()); per_bin["train_windows_usable"] = usable_built
    per_bin.to_csv(f"{R7.CACHE}/per_bin.csv", index=False)
    pd.DataFrame(val["shards"] + train["shards"]).to_parquet(f"{R7.CACHE}/shards.parquet")
    ledger = pd.DataFrame(val["ledger"] + train["ledger"], columns=["wid", "block", "cruise", "hr_res", "reason"]); ledger.to_csv(f"{R7.CACHE}/ledger.csv", index=False)
    man = dict(created=time.strftime("%Y-%m-%dT%H:%M:%S"), corpus_manifest_sha256=R7.CORPUS_MANIFEST_SHA256, counts=counts, bin_edges_m=(2 ** EDGES).tolist(), per_bin=per_bin.to_dict(orient="records"),
               heldout=dict(n_cruises=len(held_set), n_blocks=int(blocks.heldout.sum()), by_family={f: len(v) for f, v in held.items()}),
               train=dict(n_drawn=train["n_drawn"], n_built=train["n_built"], n_dropped=len(train["ledger"]), n_shards=len(train["shards"]), bins_drawn=train["bins_drawn"], bins_built=bins_built.tolist(), bytes=int(sum(s["bytes"] for s in train["shards"]))),
               val=dict(n_drawn=val["n_drawn"], n_built=val["n_built"], n_dropped=len(val["ledger"]), n_shards=len(val["shards"]), bins_drawn=val["bins_drawn"], bytes=int(sum(s["bytes"] for s in val["shards"]))),
               rsd_log10_standardisation=rsd_stats, rsd_covered_blocks_train=cov_blocks, sampler=dict(bins="8 equal-width log2 bins from log2 10 to log2 256; bin uniform, block uniform within bin, position uniform within block (integer px on the synthetic HR grid)",
               hr_res="R06 development sampler over the 16 units-v1 units: unit uniform, pair uniform; the pair's tiles_v1.1 hr_res", validity=f"window kept if >= {R7.MIN_VALID_FRAC:.0%} of its lr_px^2 LR cells are valid and >= 3 are valid (so >= 2 remain after masking {R7.MASK_FRAC:.0%})",
               position_tries=MAX_POS_TRIES, seeds=dict(pre_seed=R7.PRE_SEED, draw_train=[R7.PRE_SEED, 0], draw_val=[R7.PRE_SEED, 1], positions="default_rng([PRE_SEED, split_tag, wid])")),
               block_cover_min=R7.BLOCK_COVER_MIN, rsd_floor_m=R7.RSD_FLOOR_M, seconds=time.time() - t0, workers=a.workers, host=os.uname().nodename, slurm_job=os.environ.get("SLURM_JOB_ID"), code_hash=R7.code_hash())
    json.dump(man, open(f"{R7.CACHE}/cache_manifest.json", "w"), indent=1, default=float)
    log(f"rsd standardisation: {rsd_stats}\ntrain built {train['n_built']}/{train['n_drawn']} (dropped {len(train['ledger'])}), val {val['n_built']}/{val['n_drawn']} (dropped {len(val['ledger'])}); {time.time()-t0:.0f} s")
    open(f"{R7.CACHE}/DONE", "w").write(man["created"])
