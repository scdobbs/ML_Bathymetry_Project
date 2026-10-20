"""R14 §5.2 — the target rebuilt for the coarser cell (secondary analysis), through the frozen target functions unchanged, on the
frozen tile set, HR grid, joint mask and native ship side:
  band      lower edge f_lr' = hr_res / lr_target; upper edge FROZEN (r04_common.band_edges: min(k f_lr, 0.45) at the native lr_native);
            response targets_v1.raised_cosine_response(radial_freq(256), f_lr', f_hi), Tukey taper as frozen;
  F_hr      per frozen window exactly as build_tiles_targets.work: HR window, hr_valid, tile_ship_side (NATIVE ship grid) -> build_inputs ->
            build_target (joint, delta) -> targets_v1.window_fields; the central 128 x 128 crop mosaicked; A_fine = targets_v1.summary over
            the joint crop (as tile_and_blocks does);
  interior  targets_v1_int on the mosaic: Jm = joint_v1 & isfinite(F_hr) (identical at every level: F_hr is NaN exactly outside the joint crop),
            distance map once per pair, D1' = targets_v1_int.d_px(lr_target, hr_res), interior tile (>= 95 %) and A_fine_int (>= 2,048 px).
Native level (check): the native band reproduces the frozen tiles_v1.1 A_fine, the frozen F_hr.tif / joint_v1.tif (OAK-verified) and the
R09 interior flag / A_fine_int at D1 as carried by the R10b scoring inputs. The joint mask is NOT recomputed with the coarsened validity
(only the band and D1 change, as §5.2 lists). Writes <TGT>/<pair>/targets.parquet and record.json."""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np, pandas as pd, rasterio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r14_common as R
C4, R8 = R.C4, R.R8
if R.REPO not in sys.path: sys.path.insert(0, R.REPO)
from preprocessing import extract_clean_patches as ecp
import targets_v1 as T, targets_v1_int as TI
G = {}

def work(job):
    tid, r0, c0 = job; g = G["g"]; win = R.WIN
    hr = g["hr_arr"][r0:r0 + win, c0:c0 + win].copy(); hr_valid = g["hrv_arr"][r0:r0 + win, c0:c0 + win].copy() & np.isfinite(hr); hr = np.where(hr_valid, hr, np.nan)
    lr, lr_valid = ecp.tile_ship_side(g, r0, c0, win); inp = ecp.build_inputs(lr, lr_valid, win, R.S_FLOOR); assert inp is not None, tid
    tgt = ecp.build_target(hr, hr_valid, inp); joint = tgt["joint"].astype(bool); hr0 = np.where(joint, hr, 0.0); sl = slice(T.MARGIN, T.MARGIN + T.TILE)
    crops = {key: T.window_fields(hr0, joint, tgt["delta"], inp["lr_up_m"], H, G["taper"])[0] for key, H in G["H"].items()}
    return tid, r0, c0, joint[sl, sl], crops

def run_pair(m, pid, tp, SIp, workers, plan):
    row = C4.pair_row(m, pid); g = ecp.pair_setup(C4.DATA, row, R.WIN); H, W, hr_res = g["H"], g["W"], g["hr_res"]; f_lr, f_hi = C4.band_edges(row, hr_res); fr = T.radial_freq(R.WIN)
    bands = {"native": (f_lr, f_hi)} | {R.tag(c): (hr_res / lt, f_hi) for c, lt in plan.items()}; Hs = {k: T.raised_cosine_response(fr, a, b) for k, (a, b) in bands.items()}
    G.clear(); G.update(g=g, H=Hs, taper=T.tukey2d()); jobs = [(r.tile_id, int(r.r0), int(r.c0)) for r in tp.itertuples()]
    mos = {k: np.full((H, W), np.nan, np.float32) for k in Hs}; jm = np.zeros((H, W), np.uint8); t = tp.set_index("tile_id")
    with Pool(workers) as pool:
        for tid, r0, c0, jc, crops in pool.imap(work, jobs, chunksize=4):
            tr, tc = r0 + T.MARGIN, c0 + T.MARGIN; sl = (slice(tr, tr + T.TILE), slice(tc, tc + T.TILE)); assert not jm[sl].any(), "tile written twice"; jm[sl] = jc.astype(np.uint8)
            for k, F_ in crops.items(): mos[k][sl] = F_
    G.clear(); del g
    # frozen fields (OAK-verified): the interior distance map is the frozen one (phase4_build / R09: Jm = joint_v1 & isfinite(F_hr) over the FULL
    # tiles_v1 mosaic, which also holds the crops of the tiles the v1.1 selection dropped); the R14 mosaics cover the v1.1 tiles only and are
    # checked crop by crop against the frozen fields (joint crop identical; F_hr crop identical at the native band).
    for name in ("F_hr.tif", "joint_v1.tif"): R8.oak_verify("R04/fields", f"{pid}/{name}", f"{R8.R04_FIELDS}/{pid}/{name}")
    with rasterio.open(f"{R8.R04_FIELDS}/{pid}/F_hr.tif") as s: Ffz = s.read(1)
    with rasterio.open(f"{R8.R04_FIELDS}/{pid}/joint_v1.tif") as s: Jfz = s.read(1)
    Jm = (Jfz == 1) & np.isfinite(Ffz); dist = TI.distance_map(Jm)
    nan_same = all(np.array_equal(np.isnan(mos[k]), np.isnan(mos["native"])) for k in mos); jc_same = fz_nan_same = True; fz_max = 0.0; n_fz_cmp = 0
    rows = []; D1s = {"native": TI.d_px(float(row["lr_native_res_m"]), hr_res)} | {R.tag(c): TI.d_px(lt, hr_res) for c, lt in plan.items()}
    for r in tp.itertuples():
        sl = (slice(r.tr0, r.tr0 + TI.TILE), slice(r.tc0, r.tc0 + TI.TILE)); j, d = Jm[sl], dist[sl]; jt = jm[sl].astype(bool); o = dict(tile_id=r.tile_id, pair=pid, r0=int(r.r0), c0=int(r.c0), n_joint=int(jt.sum()))
        jc_same &= bool(np.array_equal(jm[sl], Jfz[sl])); a, b = mos["native"][sl], Ffz[sl]; fz_nan_same &= bool(np.array_equal(np.isnan(a), np.isnan(b))); ok = np.isfinite(a) & np.isfinite(b)
        if ok.any(): fz_max = max(fz_max, float(np.max(np.abs(a[ok] - b[ok])))); n_fz_cmp += int(ok.sum())
        o["frozen_joint_crop_identical"] = bool(np.array_equal(jm[sl], Jfz[sl]))
        for k in mos:
            f = mos[k][sl]; s = T.summary(f, f, jt); i1, f1 = TI.interior_tile(j, d, D1s[k]); a1, n1 = TI.afine_int(f, j, d, D1s[k])
            o.update({f"A_fine_{k}": s["A_fine"], f"frac_ge_D1_{k}": f1, f"interior_{k}": i1, f"A_fine_int_{k}": a1, f"n_int_px_{k}": n1})
        rows.append(o)
    df = pd.DataFrame(rows); d = df.set_index("tile_id"); s = SIp.set_index("tile_id").reindex(d.index)
    def eqnan(a, b): a, b = np.asarray(a, float), np.asarray(b, float); return bool(np.array_equal(np.isnan(a), np.isnan(b)) and np.array_equal(a[~np.isnan(a)], b[~np.isnan(b)]))
    chk = dict(joint_identical=bool(jc_same), F_hr_nan_pattern_identical=bool(fz_nan_same), F_hr_max_abs_diff=fz_max, n_F_hr_px_compared=n_fz_cmp, distance_map_source="frozen R04 fields (joint_v1.tif & isfinite(F_hr.tif), OAK-verified), as phase4_build / R09 targets_v1_int")
    chk.update(A_fine_identical=eqnan(d.A_fine_native.values, t.A_fine.reindex(d.index).values), A_fine_max_abs_diff=float(np.nanmax(np.abs(d.A_fine_native.values - t.A_fine.reindex(d.index).values))),
               interior_D1_identical=bool(np.array_equal(d.interior_native.values.astype(bool), s.interior.values.astype(bool))), A_fine_int_D1_identical=eqnan(d.A_fine_int_native.values, s.A_fine_int.values),
               A_fine_int_D1_max_abs_diff=float(np.nanmax(np.abs(d.A_fine_int_native.values - s.A_fine_int.values))) if np.isfinite(s.A_fine_int.values).any() else 0.0, F_hr_nan_pattern_identical_across_levels=bool(nan_same), D1_native_equals_r09=None)
    rec = dict(pair=pid, unit=C4.UNIT_OF[pid], hr_res=hr_res, lr_native=float(row["lr_native_res_m"]), k=int(row["max_recoverable_k"]), f_lr_native=f_lr, f_hi_frozen=f_hi, bands={k: dict(f_lr=a, f_hi=b, D1_px=D1s[k]) for k, (a, b) in bands.items()}, lr_target={R.tag(c): lt for c, lt in plan.items()},
               n_tiles=int(len(df)), check=chk, views={k: dict(n_interior=int(df[f"interior_{k}"].sum()), n_fallback=int(np.isfinite(df[f"A_fine_int_{k}"]).sum()), n_A_fine_defined=int(np.isfinite(df[f"A_fine_{k}"]).sum())) for k in mos})
    return df, rec

if __name__ == "__main__":
    t0 = time.time(); only = [a for a in sys.argv[1:] if not a.startswith("--")]; workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 4)); m, rec0 = R.start("targets_r14.py")
    tiles = R8.load_tiles(); SI, PP, cmap, stats = R.load_r10b(); pairs = [p for p in R.c1_pairs(PP) if not only or p in only]; summary = {}
    for pid in pairs:
        t1 = time.time(); tp = tiles[tiles.pair == pid].reset_index(drop=True); g = R.pair_geometry(tiles, pid); plan, skipped = R.level_plan(g["hr_res"], g["lr_native"]); out = f"{R.TGT}/{pid}"; os.makedirs(out, exist_ok=True)
        df, rec = run_pair(m, pid, tp, SI[SI.pair == pid], workers, plan); df.to_parquet(f"{out}/targets.parquet", index=False); rec["sha256"] = R.sha256_file(f"{out}/targets.parquet"); rec["skipped"] = {str(c): v for c, v in skipped.items()}; rec["seconds"] = time.time() - t1
        R.dump(rec, f"{out}/record.json"); summary[pid] = rec; c = rec["check"]
        print(f"{pid:40s} native check: joint {c['joint_identical']} F_hr max|d| {c['F_hr_max_abs_diff']:.1e} A_fine identical {c['A_fine_identical']} interior_D1 {c['interior_D1_identical']} A_fine_int {c['A_fine_int_D1_identical']}; views " + "; ".join(f"{k}: int {v['n_interior']} fb {v['n_fallback']}" for k, v in rec["views"].items()) + f" ({time.time()-t1:.0f} s)", flush=True)
    R.dump(dict(pairs=summary, node=os.uname().nodename, workers=workers, seconds=time.time() - t0), f"{R.TGT}/targets_record{'_' + '_'.join(only) if only else ''}.json"); print(f"done ({time.time()-t0:.0f} s)", flush=True)
