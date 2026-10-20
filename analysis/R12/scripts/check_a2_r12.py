"""R12 §A2 must-hold checks on the k = 32 harness builds and predictions (development data; counts and fractions only, no
target or prediction VALUE is printed or written):
  1. build complete: tiles_v1 / tiles_v1.1 / blocks, F_hr / F_res / bicubic_m / joint_v1, targets_v1 with A_fine finite on every
     tile with joint-valid px, tiles_int (targets_v1_int: A_fine_int at D1, interior flag at D1 = expected), the feature table,
     the P-V1 input arrays (x, lrv finite; meta rows = tiles);
  2. the band's upper frequency f_hi = k hr_res / lr_native below the clip 0.45 and recorded (tiles_v1 f_hi, pair_stats r04_summary);
     no 'warning' / 'Traceback' / 'Error' line in the build and prediction logs (PYTHONWARNINGS=always);
  3. s_lr finite on >= 99 % of tiles_v1.1 tiles, lr_px = 2 (R06 inputs record; the formula max(2, round(256 hr_res / lr_native)));
  4. GBT_ship and P-V1 (and s_lr) finite on every predicted tile;
  5. fraction of finite F_hr cells (all cells; joint-valid cells) against the unmodified frozen R04 build of the same pair.
Also informational: tile / block / interior counts against the frozen tiles_v1.1 of the same pair (the window set does not depend on k)."""
import glob, json, os, re, sys
import numpy as np, pandas as pd, rasterio
import r12_common as R12
P4 = R12.P4; R8 = P4.R8; ROOT = R12.ROOT

def finite_frac(path):
    with rasterio.open(path) as s: a = s.read(1)
    return a, float(np.isfinite(a).mean()), a.shape

def scan_log(path):
    if not path or not os.path.exists(path): return dict(path=path, found=False)
    L = open(path, errors="replace").read().splitlines(); hits = [l for l in L if re.search(r"warning|traceback|error", l, re.I)]
    return dict(path=path, found=True, n_lines=len(L), n_hits=len(hits), hits=sorted(set(h.strip()[:200] for h in hits))[:40])

if __name__ == "__main__":
    logs = sys.argv[1:]; frozen_tiles = R8.load_tiles(); out = dict(root=ROOT, logs={os.path.basename(p): scan_log(p) for p in logs}, pairs={})
    harness = json.load(open(f"{ROOT}/harness/harness_record.json"))
    for pid, h in R12.HARNESS_ROWS.items():
        b = f"{ROOT}/build/{pid}"; st = json.load(open(f"{b}/pair_stats.json")); r = dict(label=h["label"], status=st["status"])
        t1 = pd.read_parquet(f"{b}/tiles_v1.parquet"); t = pd.read_parquet(f"{b}/tiles_v1_1.parquet"); bl = pd.read_parquet(f"{b}/blocks_v1_1.parquet"); ti = pd.read_parquet(f"{b}/tiles_int.parquet"); fe = pd.read_parquet(f"{b}/features.parquet")
        ft = frozen_tiles[frozen_tiles.pair == pid]
        # 1. build complete
        r["item1"] = dict(n_tiles_v1=int(len(t1)), n_tiles_v1_1=int(len(t)), n_blocks=int(len(bl)), n_blocks_kept=int(bl.kept.sum()), frozen_n_tiles_v1_1=int(len(ft)), tile_ids_equal_frozen=bool(set(t.tile_id) == set(ft.tile_id)),
                          fields={f: os.path.exists(f"{ROOT}/fields/{pid}/{f}.tif") for f in ("F_hr", "F_res", "bicubic_m", "joint_v1")},
                          A_fine_finite_on_tiles_with_joint_px=dict(n_tiles_with_joint_px=int((t.n_joint > 0).sum()), n_A_fine_finite=int(np.isfinite(t.A_fine.values[t.n_joint.values > 0]).sum())),
                          A_fine_finite_all_tiles=int(np.isfinite(t.A_fine.values).sum()), tiles_int_rows=int(len(ti)), D1_px=int(st["D1_px"]), D1_expected=R12.EXPECTED_D1[pid], D1_ok=bool(int(st["D1_px"]) == R12.EXPECTED_D1[pid]),
                          interior_D1=dict(n_tiles=int(ti.interior_D1.sum()), n_blocks=int(st["views"]["interior"]["n_blocks"])), fallback_D1=dict(n_tiles_A_fine_int_defined=int(ti.A_fine_int_D1.notna().sum()), n_blocks=int(st["views"]["fallback"]["n_blocks"])),
                          view_B8=st["views"]["view_B8"], n_feature_rows=int(len(fe)), n_feature_tile_rows=int((fe.block_id < 0).sum()), feature_meta=dict(k=sorted(set(fe.k.astype(int))), lr_native=sorted(set(np.round(fe.lr_native.astype(float), 2))), log_depth_over_lr_finite_frac=float(np.isfinite(fe.log_depth_over_lr.astype(float)).mean())),
                          product_features=st["product_features"], contract_columns_frac_nan=st["contract_columns_frac_nan"])
        ip = f"{ROOT}/inputs/{pid}"; X = np.load(f"{ip}/x.npy", mmap_mode="r"); LV = np.load(f"{ip}/lrv.npy", mmap_mode="r"); me = pd.read_parquet(f"{ip}/meta.parquet")
        r["item1"]["pv1_inputs"] = dict(x_shape=list(X.shape), x_finite=bool(np.isfinite(np.asarray(X)).all()), lrv_shape=list(LV.shape), meta_rows=int(len(me)), meta_rows_equal_tiles=bool(len(me) == len(t)), cond_finite=bool(np.isfinite(me[["cond0", "cond1", "cond2", "cond3"]].values).all()), r06_lr_px=int(st["r06_inputs"]["lr_px"]))
        r["item1"]["passed"] = bool(all(r["item1"]["fields"].values()) and r["item1"]["A_fine_finite_on_tiles_with_joint_px"]["n_A_fine_finite"] == r["item1"]["A_fine_finite_on_tiles_with_joint_px"]["n_tiles_with_joint_px"] and r["item1"]["D1_ok"] and len(ti) == len(t) and len(fe) > 0 and r["item1"]["pv1_inputs"]["x_finite"] and r["item1"]["pv1_inputs"]["meta_rows_equal_tiles"] and r["item1"]["pv1_inputs"]["cond_finite"])
        # 2. band edge below the clip; recorded; no warnings
        hr = float(st["hr_res"]); lr = float(st["lr_native"]); k = int(st["k"]); fhi_formula = k * hr / lr; fhi_rec = sorted(set(np.round(t1.f_hi.astype(float), 10)))
        r["item2"] = dict(k=k, hr_res=hr, lr_native=lr, f_lr=hr / lr, f_hi_formula=fhi_formula, f_hi_recorded_tiles_v1=fhi_rec, f_hi_recorded_r04_summary=st["r04_summary"].get("f_hi"), clip=R12.F_HI_CLIP, below_clip=bool(fhi_formula < R12.F_HI_CLIP), equals_record=bool(len(fhi_rec) == 1 and abs(fhi_rec[0] - min(fhi_formula, R12.F_HI_CLIP)) < 1e-9))
        r["item2"]["passed"] = bool(r["item2"]["below_clip"] and r["item2"]["equals_record"])
        # 3. s_lr defined, lr_px = 2
        lr_px_formula = max(2, int(round(256 * hr / lr))); n_def = int(np.isfinite(t.s_lr.values).sum())
        r["item3"] = dict(n_tiles=int(len(t)), n_s_lr_finite=n_def, frac=n_def / max(1, len(t)), lr_px_formula=lr_px_formula, lr_px_r06=int(st["r06_inputs"]["lr_px"]), n_lr_valid_min=int(t.n_lr_valid.min()), n_lr_valid_max=int(t.n_lr_valid.max()))
        r["item3"]["passed"] = bool(r["item3"]["frac"] >= 0.99 and lr_px_formula == 2 and r["item3"]["lr_px_r06"] == 2)
        # 4. predictions finite
        pp = f"{ROOT}/predictions/{pid}.parquet"
        if os.path.exists(pp):
            P = pd.read_parquet(pp); r["item4"] = dict(n_pred_tiles=int(len(P)), n_tiles_equal=bool(len(P) == len(t)), **{f"{c}_finite": int(np.isfinite(P[c].values.astype(float)).sum()) for c in ("s_lr", "GBT_ship", "P-V1")}, **{f"{c}_positive": int((P[c].values.astype(float) > 0).sum()) for c in ("GBT_ship", "P-V1")})
            r["item4"]["passed"] = bool(r["item4"]["n_tiles_equal"] and all(r["item4"][f"{c}_finite"] == len(P) for c in ("s_lr", "GBT_ship", "P-V1")))
        else: r["item4"] = dict(passed=False, note=f"{pp} missing")
        # 5. F_hr finite fraction vs the frozen build
        Fh, ff, sh = finite_frac(f"{ROOT}/fields/{pid}/F_hr.tif"); Fz, fz, sz = finite_frac(f"{R12.FROZEN_FIELDS}/{pid}/F_hr.tif")
        with rasterio.open(f"{ROOT}/fields/{pid}/joint_v1.tif") as s: Jh = s.read(1) == 1
        with rasterio.open(f"{R12.FROZEN_FIELDS}/{pid}/joint_v1.tif") as s: Jz = s.read(1) == 1
        r["item5"] = dict(shape_harness=list(sh), shape_frozen=list(sz), same_shape=bool(sh == sz), finite_frac_all_cells=dict(harness=ff, frozen=fz), finite_frac_joint_valid_cells=dict(harness=float(np.isfinite(Fh[Jh]).mean()), frozen=float(np.isfinite(Fz[Jz]).mean())),
                          joint_valid_cells=dict(harness=int(Jh.sum()), frozen=int(Jz.sum())), joint_mask_identical=bool(sh == sz and np.array_equal(Jh, Jz)), finite_mask_identical=bool(sh == sz and np.array_equal(np.isfinite(Fh), np.isfinite(Fz))))
        um = f"{ROOT}/unmodified/fields/{pid}/F_hr.tif"
        if os.path.exists(um):
            Fu, fu, su = finite_frac(um); ut = pd.read_parquet(f"{ROOT}/unmodified/build/{pid}/tiles_v1_1.parquet")
            r["item5"]["unmodified_build_this_environment"] = dict(shape=list(su), finite_frac_all_cells=fu, finite_mask_identical_to_harness=bool(su == sh and np.array_equal(np.isfinite(Fu), np.isfinite(Fh))), finite_mask_identical_to_frozen=bool(su == sz and np.array_equal(np.isfinite(Fu), np.isfinite(Fz))), n_tiles_v1_1=int(len(ut)), tile_ids_equal_harness=bool(set(ut.tile_id) == set(t.tile_id))); del Fu
        r["item5"]["joint_cells_frozen_not_harness"] = int((Jz & ~Jh).sum()) if sh == sz else None; r["item5"]["joint_cells_harness_not_frozen"] = int((Jh & ~Jz).sum()) if sh == sz else None
        r["item5"]["criterion"] = "not NaN-dominated: >= 99 % of the harness build's own joint-valid cells finite, and the all-cell finite fraction within 0.05 of the unmodified (frozen) build"
        r["item5"]["passed"] = bool(r["item5"]["same_shape"] and r["item5"]["finite_frac_joint_valid_cells"]["harness"] >= 0.99 and ff >= fz - 0.05)
        del Fh, Fz, Jh, Jz
        r["all_passed"] = bool(all(r[f"item{i}"]["passed"] for i in range(1, 6))); out["pairs"][pid] = r
    hl = {k: v for k, v in out["logs"].items() if "harness" in k}; out["logs_clean"] = bool(hl and all(v["found"] and v["n_hits"] == 0 for v in hl.values())) if logs else None   # must-hold 2: the harness build and prediction logs
    out["gate"] = dict(all_items_passed=bool(all(r["all_passed"] for r in out["pairs"].values())), logs_clean=out["logs_clean"], harness_sha256=harness["sha256"])
    os.makedirs(f"{ROOT}/checks", exist_ok=True); json.dump(out, open(f"{ROOT}/checks/a2_checks.json", "w"), indent=1, default=str); print(json.dumps(out, indent=1, default=str))
