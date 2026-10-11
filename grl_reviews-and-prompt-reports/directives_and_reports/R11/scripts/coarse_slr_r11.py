"""R11 Part B — a development check at the sealed geometry (ship data only; nothing derived from the AUV grid; the fine-band target is
NOT recomputed). For each pair of the ten real-relief units: block-average its harmonized lr.tif (valid cells only) by the integer factor
f = max(2, round(125 / lr_native)) so that a cell is about 125 m; recompute s_lr on every tiles_v1.1 256-px window through the FROZEN
path unchanged — extract_clean_patches.tile_ship_side (area-average onto lr_px = max(2, round(256 x hr_res / lr_native')) cells over the
window, validity nearest) and build_inputs (SD of the valid cells, floored at 0.5 m; None below 2 valid cells -> NaN) — with lr_native'
= f x lr_native; then score the coarsened s_lr against the frozen A_fine through phase4_score.score_pair (eval_v1_1 rho, B4 block
bootstrap, B_min = 8, each pair its own unit) on the R10b scoring inputs (OAK-verified), beside the original s_lr of R10b; the C1 count
over the C1-set pairs of the ten and the pooled row (phase4_score.pooled_row, imported unchanged). B1: cells across the window for the
three confirmatory pairs from the ACQ-R02 records as tabulated in acq_r06_rehearsal/records/sealed_records.json (records only).
Writes <root>/coarse_slr/{tiles_coarse.parquet, per_pair.csv, coarse_slr.json}."""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np, pandas as pd, rasterio
from rasterio import Affine
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "phase4", "scripts"))
import phase4_common as P4, phase4_score as PS, eval_v1_1 as E
if P4.REPO not in sys.path: sys.path.insert(0, P4.REPO)
from preprocessing import extract_clean_patches as ecp
from preprocessing.colocated_utils import resolve_pair_paths, load_band
R8, RB, C4 = P4.R8, P4.R9.RB, P4.C4; ROOT = P4.ROOT["rehearsal"]; OUT = f"{ROOT}/coarse_slr"; OAK10B = f"{R8.OAK_ARCHIVE}/R10b/rehearsal"; WIN = 256; TARGET_CELL = 125.0
def block_mean_nan(a, f):
    H, W = a.shape; Hp, Wp = -(-H // f) * f, -(-W // f) * f; p = np.full((Hp, Wp), np.nan, np.float32); p[:H, :W] = a; b = p.reshape(Hp // f, f, Wp // f, f)
    with np.errstate(invalid="ignore"): s = np.nansum(b, axis=(1, 3)); n = np.isfinite(b).sum(axis=(1, 3))
    return np.where(n > 0, s / np.maximum(n, 1), np.nan).astype(np.float32), n
def setup(row):
    """Ship-side geometry only: the HR header (window transform / CRS; no HR data), lr.tif, lr_valid.tif."""
    paths = resolve_pair_paths(C4.DATA, row)
    with rasterio.open(paths["hr"]) as s: transform, dst_crs, hr_res = s.transform, s.crs, abs(s.transform.a)
    with rasterio.open(paths["lr"]) as s: lr_arr, lr_tr, lr_crs = load_band(s)
    with rasterio.open(paths["lr_valid"]) as s: lrv_raw = s.read(1).astype(np.float32); lrv_tr, lrv_crs = s.transform, s.crs
    return dict(transform=transform, dst_crs=dst_crs, hr_res=hr_res, lr_arr=lr_arr, lr_tr=lr_tr, lr_crs=lr_crs, lrv_raw=lrv_raw, lrv_tr=lrv_tr, lrv_crs=lrv_crs, posting=abs(lr_tr.a))
def s_lr_window(g, r0, c0):
    lr, lv = ecp.tile_ship_side(g, r0, c0, WIN); inp = ecp.build_inputs(lr, lv, WIN, 0.5)
    return (inp["s_lr"], inp["n_lr_valid"], inp["s_floor_hit"]) if inp else (np.nan, int(lv.sum()), None)
G = {}                                                            # per-pair geometry shared with the forked workers (as the frozen builders do)
def work_c(rc): return s_lr_window(G["gc"], rc[0], rc[1])
def work_n(rc): return s_lr_window(G["gn"], rc[0], rc[1])
if __name__ == "__main__":
    t0 = time.time(); pairs = [p for u in RB.TEN for p in C4.UNITS[u]]; m, root, rec = P4.start("coarse_slr_r11.py", "rehearsal", pairs); os.makedirs(OUT, exist_ok=True); workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
    tiles = R8.load_tiles(); R8.oak_verify("R10b/rehearsal", "score_inputs/inputs_D1.parquet", f"{OAK10B}/score_inputs/inputs_D1.parquet"); R8.oak_verify("R10b/rehearsal", "scores/per_pair_D1.csv", f"{OAK10B}/scores/per_pair_D1.csv"); R8.oak_verify("R10b/rehearsal", "score_inputs/calls_rehearsal.csv", f"{OAK10B}/score_inputs/calls_rehearsal.csv")
    SI = pd.read_parquet(f"{OAK10B}/score_inputs/inputs_D1.parquet"); PP = pd.read_csv(f"{OAK10B}/scores/per_pair_D1.csv").set_index("pair"); calls = pd.read_csv(f"{OAK10B}/score_inputs/calls_rehearsal.csv").fillna(""); cmap = {r["pair"]: r for r in calls.to_dict("records")}; dev_min = P4.dev_min_cells()
    # B1: records only (ACQ-R02 harmonize hr_res_m and grid cell_m as tabulated by ACQ-R06 records.py on OAK, verified against its SHA256SUMS)
    sr_path = f"{C4.DATA}/acq_r06_rehearsal/records/sealed_records.json"; sums = {l.split(maxsplit=1)[1].strip().lstrip("./"): l.split()[0] for l in open(f"{C4.DATA}/acq_r06_rehearsal/SHA256SUMS")}; assert P4.sha256_file(sr_path) == sums["records/sealed_records.json"]
    b1 = {}
    for r in json.load(open(sr_path))["sealed_pairs"]:
        if r["set"] != "confirmatory": continue
        hr, cell = r["harmonize_record"]["hr_res_m"], r["grid_record"]["cell_m"]; b1[r["pair_id"]] = dict(unit=r["unit"], hr_res_m=hr, ship_cell_m=cell, cells_across=256 * hr / cell, source="ACQ-R02 harmonize/<pair>.json hr_res_m and grid/<cruise>.json cell_m via acq_r06_rehearsal/records/sealed_records.json", note="Phase 4 uses lr_native_res_m from the supplement (C2a rule: max(footprint, posting)) and may differ")
    rows, per_tile = [], []
    for pid in pairs:
        t1 = time.time(); row = C4.pair_row(m, pid); g = setup(row); lr_native = float(row["lr_native_res_m"]); hr_res = g["hr_res"]; f = max(2, int(round(TARGET_CELL / lr_native))); tp = tiles[tiles.pair == pid]
        coarse, ncount = block_mean_nan(g["lr_arr"], f); gc = dict(g, lr_arr=coarse, lr_tr=g["lr_tr"] * Affine.scale(f), lr_px=max(2, int(round(WIN * hr_res / (f * lr_native)))))
        gn = dict(g, lr_px=max(2, int(round(WIN * hr_res / lr_native))))      # the native path, as a check that this setup reproduces the frozen s_lr
        G.clear(); G.update(gc=gc, gn=gn); rc = [(int(r.r0), int(r.c0)) for r in tp.itertuples()]
        with Pool(workers) as pool: chk_raw = pool.map(work_n, rc[:40], chunksize=4); res = pool.map(work_c, rc, chunksize=16)
        chk = [abs(a[0] - s) for a, s in zip(chk_raw, tp.s_lr.values[:40])]
        for r, (s, n, fl) in zip(tp.itertuples(), res): per_tile.append(dict(tile_id=r.tile_id, pair=pid, unit=r.unit, r0=int(r.r0), c0=int(r.c0), s_lr=float(r.s_lr), s_lr_coarse=s, n_coarse_cells_valid=int(n), floor_hit_coarse=fl, f=f))
        T = pd.DataFrame(per_tile[-len(tp):]).set_index("tile_id")
        X = SI[SI.pair == pid].copy(); X["s_lr_orig"] = X.s_lr.values; X["s_lr"] = T.s_lr_coarse.reindex(X.tile_id).values
        r = PS.score_pair(X, cmap.get(pid, {}), P4.B_MIN_DEFAULT, list(P4.CANDIDATES), None, dev_min); view = r.get("_view"); o = PP.loc[pid]
        scored_defined = int(np.isfinite(view["pred"]).sum()) if view else int(np.isfinite(X.s_lr.values[(X.interior.values.astype(bool))]).sum())
        rows.append(dict(pair=pid, unit=r["unit"], f=f, lr_native=lr_native, posting=g["posting"], coarse_cell_m=f * g["posting"], coarse_cell_m_from_lr_native=f * lr_native, hr_res=hr_res, cells_across_orig=256 * hr_res / lr_native, cells_across_coarse=256 * hr_res / (f * g["posting"]), lr_px_coarse=gc["lr_px"], lr_px_native=gn["lr_px"],
                         native_check_max_abs_diff=float(max(chk)), n_tiles=int(len(tp)), n_tiles_coarse_defined=int(np.isfinite(T.s_lr_coarse).sum()), frac_tiles_defined=float(np.isfinite(T.s_lr_coarse).mean()), median_valid_coarse_cells=float(T.n_coarse_cells_valid.median()),
                         view=r["view"], view_r10b=o["view"], n_scored=r["n_scored"], n_scored_defined=scored_defined, rho_coarse=r["rho_s_lr"], ci_lo_coarse=r["ci_lo_s_lr"], ci_hi_coarse=r["ci_hi_s_lr"], ci_estimable_coarse=r["ci_estimable"], lower_gt0_coarse=bool(r["c1_condition"]), boot_undefined_coarse=r["boot_undefined"],
                         rho_orig=o["rho_s_lr"], ci_lo_orig=o["ci_lo_s_lr"], ci_hi_orig=o["ci_hi_s_lr"], lower_gt0_orig=bool(o["c1_condition"]), status_r10b=o["status"], in_c1_set=bool(str(o["status"]).startswith("in C1")), rho_coarse_all=r["rho_all_s_lr"], seconds=time.time() - t1, _view=view))
        print(f"{pid:45s} f={f} cell {f*g['posting']:6.1f} m cells {256*hr_res/(f*g['posting']):.2f} lr_px {gc['lr_px']} defined {rows[-1]['frac_tiles_defined']:.2f} view {r['view']} rho {r['rho_s_lr']:+.3f} [{r['ci_lo_s_lr']:+.3f},{r['ci_hi_s_lr']:+.3f}] (orig {o['rho_s_lr']:+.3f} [{o['ci_lo_s_lr']:+.3f},{o['ci_hi_s_lr']:+.3f}]) native check {max(chk):.1e} ({time.time()-t1:.0f} s)", flush=True)
    PR = pd.DataFrame([{k: v for k, v in r.items() if k != "_view"} for r in rows]); PR.to_csv(f"{OUT}/per_pair.csv", index=False); pd.DataFrame(per_tile).to_parquet(f"{OUT}/tiles_coarse.parquet", index=False)
    c1 = [r for r in rows if r["in_c1_set"]]; units = [r["_view"] for r in c1 if r["_view"]]; pooled, _ = PS.pooled_row(units)
    # the original pooled row recomputed here from the same inputs (for the side-by-side; R10b's own value is in its triples.json whole pool)
    orig_units = []
    for r in c1:
        X = SI[SI.pair == r["pair"]]; X = X[np.isfinite(X.A_fine.values)]; vi = X[X.interior.values.astype(bool)]; vf = X[np.isfinite(X.A_fine_int.values)]
        tab, tgt = (vi, "A_fine") if r["view"] == "interior" else (vf, "A_fine_int"); orig_units.append(dict(pair=r["pair"], pred=tab.s_lr.values.astype(float), target=tab[tgt].values.astype(float), blk=PS.blocks_of(tab)))
    pooled_orig, _ = PS.pooled_row(orig_units)
    summ = dict(B1_confirmatory_cells_across=b1, dev_min_cells=dev_min, target_cell_m=TARGET_CELL, n_pairs=len(rows), c1_set=dict(pairs=[r["pair"] for r in c1], n=len(c1), n_pass_coarse=int(sum(r["lower_gt0_coarse"] for r in c1)), n_pass_orig=int(sum(r["lower_gt0_orig"] for r in c1)), n_ci_estimable_coarse=int(sum(r["ci_estimable_coarse"] for r in c1))),
                pooled_coarse={k: v for k, v in pooled.items() if k != "per_unit_rho"} | dict(per_unit_rho=pooled["per_unit_rho"]) if pooled else None, pooled_orig_recomputed={k: v for k, v in pooled_orig.items()} if pooled_orig else None,
                reading="The fine-band target (A_fine at D1, frozen) is not recomputed for the coarser cell: this check measures the predictor s_lr at sealed-like coarseness only.", frozen_path="extract_clean_patches.tile_ship_side + build_inputs (unchanged; lr_native' = f x lr_native)", cells_across_coarse_definition="256 x hr_res / (f x lr.tif posting)", seconds=time.time() - t0, node=os.uname().nodename)
    json.dump(summ, open(f"{OUT}/coarse_slr.json", "w"), indent=1, default=float)
    print(json.dumps(summ["c1_set"]), "pooled coarse", pooled and (pooled["mean_rho"], pooled["ci95"]), "pooled orig", pooled_orig and (pooled_orig["mean_rho"], pooled_orig["ci95"]), f"({time.time()-t0:.0f} s)")
