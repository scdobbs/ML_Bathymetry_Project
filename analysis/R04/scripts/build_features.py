"""R04 §3.1 driver: features_v1 for every tile (centre of the 128 tile) and every kept block (centre of the 32 block), two
window sizes, plus the feature leak test (100 tiles + 100 blocks recomputed with the HR grid replaced by noise).
Usage: python build_features.py [--tiles_dir D] [--out F] [--pairs a,b] [--workers N] [--leak 100]"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r04_common as C, features_v1 as F, targets_v1 as T
from preprocessing import extract_clean_patches as ecp
CTX = {}

def work(job):
    tid, bid, cy, cx = job
    return dict(tile_id=tid, block_id=bid, **F.support_features(CTX["ctx"], cy, cx))

def jobs_for(tiles, blocks, pid):
    t = tiles[tiles.pair == pid]; b = blocks[(blocks.pair == pid) & blocks.kept]
    J = [(r.tile_id, -1, r.tr0 + T.TILE / 2.0, r.tc0 + T.TILE / 2.0) for r in t.itertuples()]
    J += [(r.tile_id, int(r.block_id), r.br0 + T.BLOCK / 2.0, r.bc0 + T.BLOCK / 2.0) for r in b.itertuples()]
    return J

def build_ctx(m, pid, g=None):
    row = C.pair_row(m, pid); g = g or ecp.pair_setup(C.DATA, row, T.WIN)
    return F.pair_context(g, row, C.products_dir(row), C.ship_usable(row)), row, g

def leak_test(m, tiles, blocks, n=100, seed=0):
    rng = np.random.default_rng(seed); res = []
    st = tiles.sample(n=min(n, len(tiles)), random_state=seed); sb = blocks[blocks.kept].sample(n=min(n, int(blocks.kept.sum())), random_state=seed)
    jobs = {}
    for r in st.itertuples(): jobs.setdefault(r.pair, []).append((r.tile_id, -1, r.tr0 + T.TILE / 2.0, r.tc0 + T.TILE / 2.0))
    for r in sb.itertuples(): jobs.setdefault(r.pair, []).append((r.tile_id, int(r.block_id), r.br0 + T.BLOCK / 2.0, r.bc0 + T.BLOCK / 2.0))
    for pid, J in jobs.items():
        ctx, row, g = build_ctx(m, pid)
        g2 = dict(g); g2["hr_arr"] = np.where(np.isfinite(g["hr_arr"]), rng.standard_normal(g["hr_arr"].shape).astype(np.float32) * 100 - 2000, np.nan)
        g2["hrv_arr"] = rng.random(g["hrv_arr"].shape) > 0.3; g2["joint"] = (rng.random(g["joint"].shape) > 0.3).astype(np.uint8)
        ctx2 = F.pair_context(g2, row, C.products_dir(row), C.ship_usable(row))
        for tid, bid, cy, cx in J:
            fa = F.support_features(ctx, cy, cx); fb = F.support_features(ctx2, cy, cx)
            res.append(dict(pair=pid, tile_id=tid, block_id=bid, identical=F.feature_hash(fa) == F.feature_hash(fb)))
        del ctx, ctx2, g, g2
    return res

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tiles_dir", default=f"{C.ROOT}/tiles"); ap.add_argument("--out", default=f"{C.ROOT}/features/features_v1.parquet")
    ap.add_argument("--pairs", default=None); ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4))); ap.add_argument("--leak", type=int, default=100); a = ap.parse_args()
    m = C.start("build_features"); tiles = pd.read_parquet(f"{a.tiles_dir}/tiles_v1.parquet"); blocks = pd.read_parquet(f"{a.tiles_dir}/blocks_v1.parquet")
    pairs = a.pairs.split(",") if a.pairs else list(tiles.pair.unique()); C.assert_no_sealed(pairs); os.makedirs(os.path.dirname(a.out), exist_ok=True)
    rows = []; avail = {}
    for pid in pairs:
        t0 = time.time(); ctx, row, g = build_ctx(m, pid); CTX["ctx"] = ctx; J = jobs_for(tiles, blocks, pid)
        with Pool(a.workers) as pool: out = pool.map(work, J, chunksize=16)
        meta = dict(unit=C.UNIT_OF[pid], pair=pid, lr_native=float(row["lr_native_res_m"]), k=int(row["max_recoverable_k"]), hr_res=float(g["hr_res"]), ship_usable=C.ship_usable(row))
        s256 = tiles[tiles.pair == pid].set_index("tile_id").s_lr
        for o in out: o.update(meta); o["s_lr_256"] = float(s256[o["tile_id"]]); o["depth"] = o["mu_lr_128"]; o["log_depth_over_lr"] = float(np.log10(abs(o["mu_lr_128"]) / meta["lr_native"])) if np.isfinite(o["mu_lr_128"]) and o["mu_lr_128"] != 0 else np.nan
        rows += out; avail[pid] = dict(unit=meta["unit"], usable=meta["ship_usable"], n_rows=len(out), n_tiles=int(sum(o["block_id"] < 0 for o in out)), n_blocks=int(sum(o["block_id"] >= 0 for o in out)),
                                  frac_rsd_defined_128=float(np.mean([np.isfinite(o["rsd_med_128"]) for o in out])) if out else 0.0, frac_spectral_defined_128=float(np.mean([np.isfinite(o["beta_128"]) for o in out])) if out else 0.0,
                                  frac_spectral_defined_224=float(np.mean([np.isfinite(o["beta_224"]) for o in out])) if out else 0.0, seconds=time.time() - t0)
        print(f"  {pid}: {len(out)} rows ({avail[pid]['n_tiles']} tiles, {avail[pid]['n_blocks']} blocks) usable={meta['ship_usable']} rsd@128 defined {avail[pid]['frac_rsd_defined_128']:.2f} spectral@128 {avail[pid]['frac_spectral_defined_128']:.2f} @224 {avail[pid]['frac_spectral_defined_224']:.2f} ({time.time()-t0:.0f} s)", flush=True)
        del CTX["ctx"], ctx, g
    df = pd.DataFrame(rows); df.to_parquet(a.out); json.dump(avail, open(a.out.replace(".parquet", "_availability.json"), "w"), indent=1)
    print("wrote", a.out, df.shape, flush=True)
    if a.leak:
        res = leak_test(m, tiles[tiles.pair.isin(pairs)], blocks[blocks.pair.isin(pairs)], n=a.leak); n_ok = sum(r["identical"] for r in res)
        json.dump(dict(n=len(res), n_identical=n_ok, n_tiles=sum(r["block_id"] < 0 for r in res), n_blocks=sum(r["block_id"] >= 0 for r in res), rows=res), open(a.out.replace(".parquet", "_leak.json"), "w"), indent=1)
        print(f"FEATURE LEAK TEST: {n_ok}/{len(res)} identical with HR replaced by noise", flush=True)
        if n_ok != len(res): raise SystemExit("FEATURE LEAK TEST FAILED")
