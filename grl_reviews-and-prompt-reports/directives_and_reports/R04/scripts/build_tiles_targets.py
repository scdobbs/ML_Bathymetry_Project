"""R04 §1.1-1.3 driver: windows (R02 extract_clean_patches logic, unchanged, at stride 128), tiles_v1, band-passed fields
(targets_v1) written as per-pair GeoTIFF mosaics, tile and block targets, R02 sigma_m means (native 7), window-grid check
against the R02/R03 patch indices, and the §1.1 consistency check against R03's A_fine at matched support.
Usage: python build_tiles_targets.py [--pairs a,b] [--workers N] [--skip_consistency]"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
import numpy as np, pandas as pd, rasterio
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r04_common as C, targets_v1 as T
from preprocessing import extract_clean_patches as ecp
from scipy.stats import pearsonr

FIELDS = f"{C.ROOT}/fields"; OUT = f"{C.ROOT}/tiles"
G = {}

def r02_mosaic_sigma(pid):
    """R02 mosaic sigma_m raster (HR grid) for a native-7 pair, or None."""
    u = C.NATIVE7_PAIRS.get(pid)
    if u is None: return None
    f = f"{C.R02}/mosaics/{u}/mosaic.npz" if u != "TN299" else f"{C.R02}/mosaics/{u}/mosaic_{pid}.npz"
    if not os.path.exists(f): return None
    M = np.load(f, allow_pickle=True); s = M["sigma_m"]; cov = M["covered"].astype(bool)
    return np.where(cov & np.isfinite(s), s, np.nan).astype(np.float32)

def work(rc):
    """One window: R02 rules + inputs (unchanged code path), then targets_v1 fields. Returns None if rejected."""
    r0, c0 = rc; g = G["g"]; win = T.WIN
    if g["joint"][r0:r0 + win, c0:c0 + win].mean() < 0.5: return dict(r0=r0, c0=c0, rej="joint")
    hr = g["hr_arr"][r0:r0 + win, c0:c0 + win].copy(); hr_valid = g["hrv_arr"][r0:r0 + win, c0:c0 + win].copy() & np.isfinite(hr)
    hr = np.where(hr_valid, hr, np.nan)
    if hr_valid.mean() < 0.5: return dict(r0=r0, c0=c0, rej="hr_valid")
    hr_std = float(np.nanstd(hr))
    if not np.isfinite(hr_std) or hr_std < 0.05: return dict(r0=r0, c0=c0, rej="flat")
    lr, lr_valid = ecp.tile_ship_side(g, r0, c0, win)
    inp = ecp.build_inputs(lr, lr_valid, win, 0.5)
    if inp is None: return dict(r0=r0, c0=c0, rej="lr_valid")
    tgt = ecp.build_target(hr, hr_valid, inp); joint = tgt["joint"].astype(bool)
    F_hr, F_res = T.window_fields(np.where(joint, hr, 0.0), joint, tgt["delta"], inp["lr_up_m"], G["H"], G["taper"])
    a3 = T.r03_afine(np.where(joint, hr, 0.0), joint, tgt["delta"], G["f_lr"], G["f_hi"])
    sl = slice(T.MARGIN, T.MARGIN + T.TILE)
    return dict(r0=r0, c0=c0, rej=None, mu_lr=inp["mu_lr"], s_lr=inp["s_lr"], s_floor_hit=inp["s_floor_hit"], delta=tgt["delta"], delta_ok=tgt["delta_ok"],
                hr_std_m=hr_std, frac_joint_window=float(joint.mean()), frac_hr_valid=float(hr_valid.mean()), n_lr_valid=inp["n_lr_valid"],
                A_fine_r03conv_256=a3, F_hr=F_hr, F_res=F_res, bic=inp["lr_up_m"][sl, sl].astype(np.float32), joint=joint[sl, sl])

def write_tif(path, arr, g, dtype, nodata):
    prof = dict(driver="GTiff", dtype=dtype, count=1, height=arr.shape[0], width=arr.shape[1], crs=g["dst_crs"], transform=g["transform"],
                nodata=nodata, compress="deflate", tiled=True, blockxsize=256, blockysize=256, BIGTIFF="IF_SAFER")
    with rasterio.open(path, "w", **prof) as d: d.write(arr, 1)

def process_pair(m, pid, workers):
    row = C.pair_row(m, pid); unit = C.UNIT_OF[pid]; t0 = time.time()
    g = ecp.pair_setup(C.DATA, row, T.WIN); H, W = g["H"], g["W"]; hr_res = g["hr_res"]
    f_lr, f_hi = C.band_edges(row, hr_res); k = int(row["max_recoverable_k"])
    G.clear(); G.update(g=g, H=T.raised_cosine_response(T.radial_freq(T.WIN), f_lr, f_hi), taper=T.tukey2d(), f_lr=f_lr, f_hi=f_hi)
    cand = [(r0, c0) for r0 in range(0, H - T.WIN + 1, T.STRIDE) for c0 in range(0, W - T.WIN + 1, T.STRIDE)]
    with Pool(workers) as pool: res = pool.map(work, cand, chunksize=8)
    acc = [r for r in res if r["rej"] is None]; rej = {}
    for r in res:
        if r["rej"]: rej[r["rej"]] = rej.get(r["rej"], 0) + 1
    print(f"  {pid}: {len(cand)} candidate windows, {len(acc)} accepted, rejected {rej}, hr_res={hr_res:.3f} f_lr={f_lr:.4f} f_hi={f_hi:.4f} k={k} ({time.time()-t0:.0f} s)", flush=True)
    # mosaics
    F_hr = np.full((H, W), np.nan, np.float32); F_res = np.full((H, W), np.nan, np.float32); bic = np.full((H, W), np.nan, np.float32); jm = np.zeros((H, W), np.uint8)
    sig = r02_mosaic_sigma(pid)
    if sig is not None: assert sig.shape == (H, W), (pid, sig.shape, (H, W))
    for r in acc:
        tr, tc = r["r0"] + T.MARGIN, r["c0"] + T.MARGIN; sl = (slice(tr, tr + T.TILE), slice(tc, tc + T.TILE))
        assert not jm[sl].any(), "tile written twice"
        F_hr[sl] = r["F_hr"]; F_res[sl] = r["F_res"]; bic[sl] = r["bic"]; jm[sl] = r["joint"].astype(np.uint8)
    pdir = f"{FIELDS}/{pid}"; os.makedirs(pdir, exist_ok=True)
    write_tif(f"{pdir}/F_hr.tif", F_hr, g, "float32", np.nan); write_tif(f"{pdir}/F_res.tif", F_res, g, "float32", np.nan)
    write_tif(f"{pdir}/bicubic_m.tif", bic, g, "float32", np.nan); write_tif(f"{pdir}/joint_v1.tif", jm, g, "uint8", 0)
    # tiles + blocks
    tiles, blocks = [], []
    for r in acc:
        r0, c0 = r["r0"], r["c0"]; tr, tc = r0 + T.MARGIN, c0 + T.MARGIN; sl = (slice(tr, tr + T.TILE), slice(tc, tc + T.TILE))
        jt = jm[sl].astype(bool); tt, bb = T.tile_and_blocks(F_hr[sl], F_res[sl], jt)
        tid = f"{pid}_r{r0:06d}_c{c0:06d}"
        wsl = (slice(r0, r0 + T.WIN), slice(c0, c0 + T.WIN)); jw = jm[wsl].astype(bool) & np.isfinite(F_hr[wsl])
        a256 = float(np.sqrt(np.mean(F_hr[wsl][jw] ** 2))) if jw.sum() else np.nan
        cov256 = float(jw.sum() / max(1, (jm[wsl] > 0).sum()))
        b = rasterio.windows.bounds(rasterio.windows.Window(tc, tr, T.TILE, T.TILE), g["transform"])
        tiles.append(dict(unit=unit, pair=pid, window_id=tid, tile_id=tid, r0=r0, c0=c0, tr0=tr, tc0=tc, size_px=T.TILE,
                          left=b[0], bottom=b[1], right=b[2], top=b[3], crs=str(g["dst_crs"]), hr_res=hr_res, lr_native=float(row["lr_native_res_m"]), k=k, f_lr=f_lr, f_hi=f_hi,
                          mu_lr=r["mu_lr"], s_lr=r["s_lr"], s_floor_hit=r["s_floor_hit"], delta=r["delta"], delta_ok=r["delta_ok"], hr_std_m=r["hr_std_m"],
                          frac_joint_window=r["frac_joint_window"], frac_hr_valid_window=r["frac_hr_valid"], n_lr_valid=r["n_lr_valid"],
                          A_fine_r03conv_256=r["A_fine_r03conv_256"], A_fine_Fhr_256=a256, Fhr_cov_256=cov256,
                          sigma_m_R02=float(np.nanmean(sig[sl][jt])) if (sig is not None and jt.any() and np.isfinite(sig[sl][jt]).any()) else np.nan, **tt))
        for s in bb:
            bsl = (slice(tr + s["br"], tr + s["br"] + T.BLOCK), slice(tc + s["bc"], tc + s["bc"] + T.BLOCK)); jb = jm[bsl].astype(bool)
            blocks.append(dict(unit=unit, pair=pid, tile_id=tid, br0=tr + s["br"], bc0=tc + s["bc"], size_px=T.BLOCK,
                               sigma_m_R02=float(np.nanmean(sig[bsl][jb])) if (sig is not None and jb.any() and np.isfinite(sig[bsl][jb]).any()) else np.nan, **s))
    # window-grid check against R02 / R03 patch indices (stride-128 subset)
    chk = dict(pair=pid, n_r04=len(acc), prior=None)
    for name, p in (("R02", f"{C.R02}/patches_clean/patch_index.parquet"), ("R03", f"{C.R03}/patches_clean_expanded/patch_index.parquet")):
        if not os.path.exists(p): continue
        pi = pd.read_parquet(p); pi = pi[pi.pair_id == pid]
        if len(pi) == 0: continue
        prior = {(int(a), int(b)) for a, b in zip(pi.r0, pi.c0) if a % T.STRIDE == 0 and b % T.STRIDE == 0}; mine = {(r["r0"], r["c0"]) for r in acc}
        chk.update(prior=name, n_prior_total=int(len(pi)), n_prior_stride128=len(prior), n_common=len(prior & mine), only_prior=sorted(prior - mine)[:10], only_r04=sorted(mine - prior)[:10],
                   identical=(prior == mine), n_only_prior=len(prior - mine), n_only_r04=len(mine - prior))
    del G["g"]; del g
    return pd.DataFrame(tiles), pd.DataFrame(blocks), dict(pair=pid, unit=unit, n_candidates=len(cand), n_windows=len(acc), rejected=rej, hr_res=hr_res, f_lr=f_lr, f_hi=f_hi, k=k, H=H, W=W, seconds=time.time() - t0), chk

def consistency_check(tiles):
    """§1.1: R03 A_fine (full 256 window, hard annulus, no taper) vs RMS of F_hr over the same 256 window (and vs my
    R03-convention recomputation, and vs the R04 128-tile A_fine), per unit: n, log10 Pearson r, median ratio."""
    out = {}; rows = []
    s3 = pd.read_parquet(f"{C.R03}/s3/s3_pertile.parquet")[["pid", "key", "A_fine"]].rename(columns={"pid": "pair", "A_fine": "A_fine_R03_s3"})
    s3["r0"] = s3.key.str.extract(r"_r(\d{6})_").astype(int); s3["c0"] = s3.key.str.extract(r"_c(\d{6})").astype(int)
    s4 = pd.read_parquet(f"{C.R03}/s4/features_all.parquet")[["pair_id", "r0", "c0", "A_fine"]].rename(columns={"pair_id": "pair", "A_fine": "A_fine_R03_s4"})
    t = tiles.merge(s3[["pair", "r0", "c0", "A_fine_R03_s3"]], on=["pair", "r0", "c0"], how="left").merge(s4, on=["pair", "r0", "c0"], how="left")
    t["A_fine_R03"] = t.A_fine_R03_s3.fillna(t.A_fine_R03_s4)
    def stats(a, b):
        ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
        if ok.sum() < 3: return dict(n=int(ok.sum()), r_log=np.nan, median_ratio=np.nan)
        return dict(n=int(ok.sum()), r_log=float(pearsonr(np.log10(a[ok]), np.log10(b[ok]))[0]), median_ratio=float(np.median(b[ok] / a[ok])))
    for u, d in t.groupby("unit"):
        d = d[d.A_fine_R03.notna()]
        if len(d) == 0: continue
        rec = dict(unit=u, r03_name=C.R03_UNIT_NAME.get(u, u), n_r03_rows=int(len(d)),
                   vs_Fhr256=stats(d.A_fine_R03.values, d.A_fine_Fhr_256.values), vs_r03conv_recomputed=stats(d.A_fine_R03.values, d.A_fine_r03conv_256.values),
                   vs_tile128=stats(d.A_fine_R03.values, d.A_fine.values), Fhr_cov_256_median=float(d.Fhr_cov_256.median()),
                   s3_vs_s4=stats(d.A_fine_R03_s3.values, d.A_fine_R03_s4.values) if d.A_fine_R03_s3.notna().any() and d.A_fine_R03_s4.notna().any() else None)
        rec["below_0p9"] = bool(np.isfinite(rec["vs_Fhr256"]["r_log"]) and rec["vs_Fhr256"]["r_log"] < 0.9); out[u] = rec
    # whole-corpus: R03-convention (my recomputation on ALL R04 windows) vs F_hr RMS on the same window
    for u, d in tiles.groupby("unit"):
        rows.append(dict(unit=u, **{f"all_{k}": v for k, v in stats(d.A_fine_r03conv_256.values, d.A_fine_Fhr_256.values).items()}))
    return out, pd.DataFrame(rows)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--pairs", default=None); ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    ap.add_argument("--skip_consistency", action="store_true"); ap.add_argument("--out", default=OUT); a = ap.parse_args()
    m = C.start("build_tiles_targets"); assert T.E1M_THRESHOLD_M == C.E1M_THRESHOLD_M
    v = C.verify_unit_list(m); print("unit-list verification:", json.dumps(v)); assert v["verified"], v
    pairs = a.pairs.split(",") if a.pairs else C.DEV_PAIRS + C.EW0207["lu16"]; C.assert_no_sealed(pairs)
    os.makedirs(a.out, exist_ok=True); os.makedirs(FIELDS, exist_ok=True)
    TL, BL, SUM, CHK = [], [], [], []
    for pid in pairs:
        t, b, s, c = process_pair(m, pid, a.workers); TL.append(t); BL.append(b); SUM.append(s); CHK.append(c)
        print(f"    grid check: {c}", flush=True)
    tiles = pd.concat(TL, ignore_index=True); blocks = pd.concat(BL, ignore_index=True)
    tiles.to_parquet(f"{a.out}/tiles_v1.parquet"); blocks.to_parquet(f"{a.out}/blocks_v1.parquet")
    json.dump(dict(pairs=SUM, grid_checks=CHK, units_version=C.UNITS_VERSION, e1m_threshold_m=T.E1M_THRESHOLD_M), open(f"{a.out}/tiles_summary.json", "w"), indent=1, default=str)
    print(f"tiles_v1: {len(tiles)} tiles, {len(blocks)} blocks ({int(blocks.kept.sum())} kept) over {tiles.pair.nunique()} pairs", flush=True)
    if not a.skip_consistency:
        cc, allrows = consistency_check(tiles)
        json.dump(cc, open(f"{a.out}/consistency_check.json", "w"), indent=1, default=float); allrows.to_csv(f"{a.out}/consistency_all_windows.csv", index=False)
        for u, r in cc.items(): print(f"  consistency {u} ({r['r03_name']}): n={r['n_r03_rows']} r_log(F_hr256)={r['vs_Fhr256']['r_log']:.3f} ratio={r['vs_Fhr256']['median_ratio']:.3f} | r_log(recomputed R03 conv)={r['vs_r03conv_recomputed']['r_log']:.4f} ratio={r['vs_r03conv_recomputed']['median_ratio']:.4f} | r_log(tile128)={r['vs_tile128']['r_log']:.3f} {'BELOW 0.9' if r['below_0p9'] else ''}", flush=True)
    print("done", flush=True)
