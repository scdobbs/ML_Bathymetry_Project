"""R14 §4.1-4.2 — s_lr and the features_v1 bank on the FROZEN tile set (tiles_v1.1 windows of each C1-set pair) at the native level and
at every coarsened level, through the frozen code unchanged:
  geometry  extract_clean_patches.pair_setup(DATA, row, 256) with the pair's manifest row (native) or its harness row (lr_native_res_m =
            lr_target, LR raster and validity paths pointed at the coarsened files; r14_common.harness_row);
  s_lr      extract_clean_patches.tile_ship_side (area-average onto lr_px = max(2, round(256 x hr_res / lr_native)) cells over the frozen
            window, validity nearest) + build_inputs (SD of the valid cells, floored at 0.5 m; None below 2 valid cells -> NaN);
  features  features_v1.pair_context (products from <dir of lr.tif>/ship_products_v2_1 where usable) + support_features at the tile centre,
            both windows, then the driver's META columns exactly as build_features.__main__ / phase4_build.build_pair write them
            (lr_native = the row's lr_native_res_m, k and hr_res frozen, s_lr_256 = the window s_lr of the same build, depth = mu_lr_128,
            log_depth_over_lr = log10(|mu_lr_128| / lr_native)).
Native level (§3.3 must-hold): s_lr against tiles_v1.1 and every bar_v1_1.ALL column against the frozen R04 feature table (tile rows).
Writes <PRED>/<pair>/{native,c<level>}.parquet and record.json."""
import json, math, os, sys, time
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r14_common as R
C4, R8 = R.C4, R.R8
if R.REPO not in sys.path: sys.path.insert(0, R.REPO)
from preprocessing import extract_clean_patches as ecp
import features_v1 as F, targets_v1 as T, bar_v1_1 as B
G = {}

def work(job):
    tid, r0, c0, tr0, tc0 = job; g, ctx = G["g"], G["ctx"]
    lr, lv = ecp.tile_ship_side(g, r0, c0, R.WIN); inp = ecp.build_inputs(lr, lv, R.WIN, R.S_FLOOR)
    s = dict(tile_id=tid, r0=r0, c0=c0, s_lr=(inp["s_lr"] if inp else np.nan), n_lr_valid=(inp["n_lr_valid"] if inp else int(lv.sum())), s_floor_hit=(bool(inp["s_floor_hit"]) if inp else None), mu_lr_256=(inp["mu_lr"] if inp else np.nan))
    return s, F.support_features(ctx, tr0 + T.TILE / 2.0, tc0 + T.TILE / 2.0)

def run_level(m, pid, row, tp, workers):
    g = ecp.pair_setup(C4.DATA, row, R.WIN); usable = C4.ship_usable(row); ctx = F.pair_context(g, row, C4.products_dir(row), usable)
    G.clear(); G.update(g=g, ctx=ctx); jobs = [(r.tile_id, int(r.r0), int(r.c0), int(r.tr0), int(r.tc0)) for r in tp.itertuples()]
    with Pool(workers) as pool: res = pool.map(work, jobs, chunksize=8)
    meta = dict(unit=C4.UNIT_OF[pid], pair=pid, lr_native=float(row["lr_native_res_m"]), k=int(row["max_recoverable_k"]), hr_res=float(g["hr_res"]), ship_usable=usable); rows = []
    for s, f in res:
        o = dict(f); o.update(meta); o["s_lr_256"] = float(s["s_lr"]); o["depth"] = o["mu_lr_128"]
        o["log_depth_over_lr"] = float(np.log10(abs(o["mu_lr_128"]) / meta["lr_native"])) if np.isfinite(o["mu_lr_128"]) and o["mu_lr_128"] != 0 else np.nan
        o.update(tile_id=s["tile_id"], block_id=-1, r0=s["r0"], c0=s["c0"], s_lr=s["s_lr"], n_lr_valid=s["n_lr_valid"], s_floor_hit=s["s_floor_hit"], mu_lr_256=s["mu_lr_256"], lr_px=g["lr_px"]); rows.append(o)
    info = dict(lr_px=g["lr_px"], lr_native=g["lr_native"], hr_res=g["hr_res"], cell_m=float(ctx["cell_m"]), usable=bool(ctx["usable"]), lr_shape=list(g["lr_arr"].shape), paths=dict(lr=g["paths"]["lr"], lr_valid=g["paths"]["lr_valid"]), products_dir=(C4.products_dir(row) if usable else None))
    G.clear(); del g, ctx
    return pd.DataFrame(rows), info

def native_check(df, tp, feats_frozen):
    """s_lr vs tiles_v1.1; bar_v1_1.ALL columns vs the frozen R04 feature table (tile rows), NaN pattern and max |diff| per column."""
    d = df.set_index("tile_id"); t = tp.set_index("tile_id"); fz = feats_frozen.set_index("tile_id").reindex(d.index)
    out = dict(n_tiles=int(len(d)), s_lr_max_abs_diff=float(np.nanmax(np.abs(d.s_lr.values - t.s_lr.reindex(d.index).values))), s_lr_identical=bool(np.array_equal(d.s_lr.values, t.s_lr.reindex(d.index).values)),
               n_lr_valid_identical=bool(np.array_equal(d.n_lr_valid.values, t.n_lr_valid.reindex(d.index).values)))
    cols = {}; worst = 0.0; nan_ok = True
    for c in B.ALL:
        a, b = d[c].values.astype(float), fz[c].values.astype(float); na, nb = np.isnan(a), np.isnan(b); same_nan = bool(np.array_equal(na, nb)); ok = ~na & ~nb
        mx = float(np.max(np.abs(a[ok] - b[ok]))) if ok.any() else 0.0; cols[c] = dict(max_abs_diff=mx, nan_pattern_identical=same_nan, n_nan=int(na.sum())); worst = max(worst, mx); nan_ok &= same_nan
    out.update(features_max_abs_diff=worst, features_nan_pattern_identical=nan_ok, features_within_1e6=bool(worst <= 1e-6), columns_above_1e9={c: v["max_abs_diff"] for c, v in cols.items() if v["max_abs_diff"] > 1e-9})
    return out

if __name__ == "__main__":
    t0 = time.time(); only = [a for a in sys.argv[1:] if not a.startswith("--")]; workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 4)); m, rec = R.start("predictors_r14.py")
    tiles = R8.load_tiles(); feats = R8.load_features(); feats = feats[feats.block_id < 0]; SI, PP, cmap, stats = R.load_r10b(); pairs = [p for p in R.c1_pairs(PP) if not only or p in only]
    cr = json.load(open(f"{R.COARSE}/coarse_record.json")); summary = {}
    for pid in pairs:
        t1 = time.time(); tp = tiles[tiles.pair == pid].reset_index(drop=True); out = f"{R.PRED}/{pid}"; os.makedirs(out, exist_ok=True); row = C4.pair_row(m, pid); rp = dict(pair=pid, unit=C4.UNIT_OF[pid], n_tiles=int(len(tp)), levels={})
        df, info = run_level(m, pid, row, tp, workers); df.to_parquet(f"{out}/native.parquet", index=False); chk = native_check(df, tp, feats[feats.pair == pid]); rp["native"] = dict(info=info, check=chk, sha256=R.sha256_file(f"{out}/native.parquet"))
        print(f"{pid:40s} native: lr_px {info['lr_px']} cell {info['cell_m']:.2f} m; s_lr identical {chk['s_lr_identical']} (max |d| {chk['s_lr_max_abs_diff']:.1e}); features max |d| {chk['features_max_abs_diff']:.2e}, NaN pattern {chk['features_nan_pattern_identical']} ({time.time()-t1:.0f} s)", flush=True)
        for c in R.LEVELS:
            lv = cr["pairs"][pid]["levels"].get(str(c))
            if lv is None: rp["levels"][str(c)] = dict(skipped=True, reason="lr_target <= lr_native"); continue
            t2 = time.time(); hrow = R.harness_row(m, pid, lv["lr_target"], R.coarse_dir(pid, c)); assert sorted(R.harness_changes(m, pid, hrow)) == ["harmonized_path_lr", "lr_native_res_m", "lr_valid"]
            for f_ in (hrow["harmonized_path_lr"], hrow["lr_valid"]): assert R.sha256_file(f_) == lv["files"][os.path.basename(f_)], f_      # the coarsened rasters as recorded
            df, info = run_level(m, pid, hrow, tp, workers); p = f"{out}/{R.tag(c)}.parquet"; df.to_parquet(p, index=False)
            und = int(np.isnan(df.s_lr.values).sum()); flo = int(np.nansum([1 if v else 0 for v in df.s_floor_hit.values]))
            rp["levels"][str(c)] = dict(skipped=False, lr_target=lv["lr_target"], cells_across=lv["cells_across"], info=info, n_tiles=int(len(df)), n_s_lr_undefined=und, n_s_lr_floor=flo, frac_undefined=und / len(df), frac_floor=flo / len(df), median_n_lr_valid=float(np.median(df.n_lr_valid.values)), sha256=R.sha256_file(p), seconds=time.time() - t2)
            print(f"  c={c:4}: lr_target {lv['lr_target']:7.2f} m lr_px {info['lr_px']} cell {info['cell_m']:.2f} m; s_lr undefined {und}/{len(df)} floor {flo} ({time.time()-t2:.0f} s)", flush=True)
        rp["seconds"] = time.time() - t1; R.dump(rp, f"{out}/record.json"); summary[pid] = rp
    R.dump(dict(pairs=summary, node=os.uname().nodename, workers=workers, seconds=time.time() - t0), f"{R.PRED}/predictors_record{'_' + '_'.join(only) if only else ''}.json"); print(f"done ({time.time()-t0:.0f} s)", flush=True)
