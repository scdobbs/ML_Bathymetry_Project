"""contract_v2_1_validate — CNN-repo (consumer-side) validator for INTERFACE_CONTRACT_v2.1 (directive R04 §6.2).
Independent of the acquisition repo's validator. Checks, per pair: products.json schema (§5), frame_offset rules,
grid identity of every raster with lr.tif (CRS, transform, shape), dtype/nodata, NaN and count conditions (§3), value
ranges, QA gate reproduction (§4), and the ship_products_usable gate (§6: usable only if every `available` flag is true
AND QA flags are empty; the manifest column must agree).
Usage:  python contract_v2_1_validate.py --manifest pairs_v2_1.parquet [--pairs a,b] --out DIR
        python contract_v2_1_validate.py --test            (unit tests on synthetic conforming / non-conforming directories)"""
from __future__ import annotations
import argparse, csv, json, os, sys, tempfile
import numpy as np, pandas as pd, rasterio
from rasterio.transform import from_origin

TOP_KEYS = {"contract_version", "pair_id", "lr_tif_path", "available", "unavailable_reason", "frame_offset", "qa_vs_lr_tif"}
AVAIL = ("ship_count", "ship_sd", "ship_rsd", "ship_xtrack_frac", "ship_beam_angle")
RASTERS = AVAIL + ("ship_mean_regrid", "ship_median_regrid")
FO_KEYS = {"applied", "dx_m", "dy_m", "basis", "evidence"}
QA_KEYS = {"n_common_cells", "median_offset_m", "shift_argmin_cells", "flags", "sigma0_m", "sigma_argmin_m", "gain_m"}

def validate_pair(pair_dir, pid, usable_manifest=None):
    """pair_dir = directory containing lr.tif (from the manifest). Returns dict(errors=[...], info={...})."""
    d = os.path.join(pair_dir, "ship_products_v2_1"); E = []; info = dict(pair_id=pid, dir=d)
    pj = os.path.join(d, "products.json")
    if not os.path.exists(pj): return dict(errors=["products.json missing"], info=info)
    p = json.load(open(pj))
    miss = TOP_KEYS - set(p)
    if miss: E.append(f"missing keys {sorted(miss)}")
    if p.get("contract_version") != "2.1": E.append(f"contract_version {p.get('contract_version')!r} != '2.1'")
    if p.get("pair_id") != pid: E.append("pair_id mismatch")
    fo = p.get("frame_offset") or {}
    if FO_KEYS - set(fo): E.append(f"frame_offset keys missing {sorted(FO_KEYS - set(fo))}")
    if fo.get("basis") not in ("none", "grid_header_registration"): E.append("frame_offset.basis invalid")
    if fo.get("applied") and fo.get("basis") != "grid_header_registration": E.append("frame_offset applied without grid_header_registration basis")
    if not fo.get("applied") and (fo.get("dx_m") or fo.get("dy_m")): E.append("frame_offset not applied but dx/dy non-zero")
    av = p.get("available") or {}; all_av = all(bool(av.get(k)) for k in AVAIL); info["available"] = all_av
    tifs = {r: os.path.join(d, f"{r}.tif") for r in RASTERS}
    if not all_av:
        if any(os.path.exists(t) for t in tifs.values()): E.append("unavailable pair but rasters present")
        if not p.get("unavailable_reason"): E.append("unavailable without unavailable_reason")
        info["flags"] = None; info["usable_rule"] = False
    else:
        qa = p.get("qa_vs_lr_tif") or {}
        if QA_KEYS - set(qa): E.append(f"qa keys missing {sorted(QA_KEYS - set(qa))}")
        flags = list(qa.get("flags") or []); info["flags"] = flags; info["usable_rule"] = (len(flags) == 0)
        # gate reproduction (§4) from the recorded numbers
        try:
            exp = []
            if qa["n_common_cells"] < 500: exp.append("insufficient_overlap")
            if "vertical_offset" in flags: exp.append("vertical_offset")   # needs median s_lr (recorded upstream); accepted as recorded
            am = qa["shift_argmin_cells"]
            if max(abs(am[0]), abs(am[1])) >= 0.5 and qa["gain_m"] >= max(0.10, 0.05 * qa["sigma0_m"]) - 1e-9: exp.append("registration_shift")
            if sorted(exp) != sorted(flags): E.append(f"flags {flags} do not reproduce the v2.1 gate {exp}")
            if abs((qa["sigma0_m"] - qa["sigma_argmin_m"]) - qa["gain_m"]) > 2e-3: E.append("gain_m != sigma0_m - sigma_argmin_m")
            sp = os.path.join(d, qa.get("shift_surface_file") or "qa_shift_surface.csv")
            if os.path.exists(sp):
                rows = list(csv.DictReader(open(sp)))
                if len(rows) != 81: E.append(f"shift surface has {len(rows)} rows, expected 81")
            else: E.append("shift surface csv missing")
        except (KeyError, TypeError, IndexError) as e: E.append(f"qa block unreadable: {e!r}")
        # rasters
        lrp = os.path.join(pair_dir, "lr.tif")
        with rasterio.open(lrp) as lr: lcrs, ltr, lshape = lr.crs, lr.transform, lr.shape; lr_arr = lr.read(1, masked=True).filled(np.nan)
        arrs = {}
        for r, t in tifs.items():
            if not os.path.exists(t): E.append(f"{r}.tif missing"); continue
            with rasterio.open(t) as ds:
                if ds.count != 1: E.append(f"{r} has {ds.count} bands")
                if ds.dtypes[0] != "float32": E.append(f"{r} dtype {ds.dtypes[0]} != float32")
                if ds.nodata is not None and not np.isnan(ds.nodata): E.append(f"{r} nodata {ds.nodata} != NaN")
                if ds.crs != lcrs: E.append(f"{r} CRS != lr.tif")
                if ds.transform != ltr: E.append(f"{r} transform != lr.tif")
                if ds.shape != lshape: E.append(f"{r} shape {ds.shape} != lr.tif {lshape}")
                if ds.shape == lshape: arrs[r] = ds.read(1)
        if len(arrs) == len(RASTERS):
            n = arrs["ship_count"]; has = np.isfinite(n)
            if (n[has] < 1).any(): E.append("ship_count < 1 where finite")
            if has.any() and (n[has] != np.round(n[has])).any(): E.append("ship_count non-integer")
            for r in ("ship_xtrack_frac", "ship_beam_angle", "ship_mean_regrid", "ship_median_regrid"):
                if (np.isfinite(arrs[r]) & ~has).any(): E.append(f"{r} finite where count is NaN")
            for r in ("ship_mean_regrid", "ship_median_regrid", "ship_xtrack_frac", "ship_beam_angle"):
                if (has & ~np.isfinite(arrs[r])).any(): E.append(f"{r} NaN where count >= 1")
            if (np.isfinite(arrs["ship_sd"]) & ~(has & (n >= 2))).any(): E.append("ship_sd finite where count < 2")
            if ((has & (n >= 2)) & ~np.isfinite(arrs["ship_sd"])).any(): E.append("ship_sd NaN where count >= 2")
            if (np.isfinite(arrs["ship_rsd"]) & ~(has & (n >= 3))).any(): E.append("ship_rsd finite where count < 3")
            if ((has & (n >= 3)) & ~np.isfinite(arrs["ship_rsd"])).any(): E.append("ship_rsd NaN where count >= 3")
            x = arrs["ship_xtrack_frac"][np.isfinite(arrs["ship_xtrack_frac"])]
            if x.size and (x.min() < 0 or x.max() > 1.0001): E.append("ship_xtrack_frac outside [0, 1]")
            b = arrs["ship_beam_angle"][np.isfinite(arrs["ship_beam_angle"])]
            if b.size and (b.min() < 0 or b.max() > 90): E.append("ship_beam_angle outside [0, 90]")
            for r in ("ship_sd", "ship_rsd"):
                v = arrs[r][np.isfinite(arrs[r])]
                if v.size and (v < 0).any(): E.append(f"{r} negative")
            both = np.isfinite(arrs["ship_median_regrid"]) & np.isfinite(lr_arr)
            info["n_common_with_lr"] = int(both.sum())
            if both.any() and np.sign(np.nanmedian(arrs["ship_median_regrid"][both])) != np.sign(np.nanmedian(lr_arr[both])): E.append("ship_median_regrid sign differs from lr.tif")
            info["n_cells_with_soundings"] = int(has.sum())
    # §6 usable gate vs manifest
    if usable_manifest is not None:
        um = bool(usable_manifest is True or usable_manifest == 1 or str(usable_manifest).lower() == "true"); info["usable_manifest"] = um
        if um and not info["usable_rule"]: E.append("manifest ship_products_usable=True but products unavailable or flagged")
    return dict(errors=E, info=info)

# ---------------- unit tests ----------------
def _write_synthetic(root, pid, conforming=True, variant=None, unavailable=False):
    pdir = os.path.join(root, pid); d = os.path.join(pdir, "ship_products_v2_1"); os.makedirs(d, exist_ok=True)
    n = 12; tr = from_origin(500000.0, 4000000.0, 25.0, 25.0); crs = "EPSG:32610"
    prof = dict(driver="GTiff", dtype="float32", count=1, height=n, width=n, crs=crs, transform=tr, nodata=np.nan)
    rng = np.random.default_rng(1); lr = (-1500 + rng.normal(0, 5, (n, n))).astype(np.float32)
    with rasterio.open(os.path.join(pdir, "lr.tif"), "w", **prof) as ds: ds.write(lr, 1)
    pj = dict(contract_version="2.1", pair_id=pid, lr_tif_path=os.path.join(pdir, "lr.tif"), available={k: (not unavailable) for k in AVAIL},
              unavailable_reason=("no swath" if unavailable else None), frame_offset=dict(applied=False, dx_m=0.0, dy_m=0.0, basis="none", evidence="synthetic"),
              qa_vs_lr_tif=dict(n_common_cells=600, median_offset_m=0.05, shift_argmin_cells=[0.0, 0.25], flags=[], sigma0_m=1.0, sigma_argmin_m=0.9, gain_m=0.1, shift_surface_file="qa_shift_surface.csv"))
    if not unavailable:
        cnt = rng.integers(0, 6, (n, n)).astype(np.float32); cnt[cnt == 0] = np.nan
        sd = np.where(cnt >= 2, np.abs(rng.normal(1, 0.3, (n, n))), np.nan).astype(np.float32); rsd = np.where(cnt >= 3, sd * 0.9, np.nan).astype(np.float32)
        xt = np.where(np.isfinite(cnt), rng.random((n, n)), np.nan).astype(np.float32); ba = np.where(np.isfinite(cnt), rng.random((n, n)) * 60, np.nan).astype(np.float32)
        mean = np.where(np.isfinite(cnt), lr + 0.1, np.nan).astype(np.float32); med = mean.copy()
        if variant == "sd_where_count1": sd = np.where(np.isfinite(cnt), np.abs(rng.normal(1, 0.3, (n, n))), np.nan).astype(np.float32)
        if variant == "rsd_missing": rsd[:] = np.nan
        if variant == "xtrack_range": xt = xt * 3
        arrs = dict(ship_count=cnt, ship_sd=sd, ship_rsd=rsd, ship_xtrack_frac=xt, ship_beam_angle=ba, ship_mean_regrid=mean, ship_median_regrid=med)
        for r, a in arrs.items():
            pr = dict(prof)
            if variant == "grid_mismatch" and r == "ship_sd": pr["transform"] = from_origin(500010.0, 4000000.0, 25.0, 25.0)
            if variant == "dtype" and r == "ship_count": pr["dtype"] = "float64"; a = a.astype(np.float64)
            with rasterio.open(os.path.join(d, f"{r}.tif"), "w", **pr) as ds: ds.write(a, 1)
        with open(os.path.join(d, "qa_shift_surface.csv"), "w") as f:
            f.write("dx_cells,dy_cells,robust_sigma_m\n")
            for dx in np.arange(-1, 1.01, 0.25):
                for dy in np.arange(-1, 1.01, 0.25): f.write(f"{dx},{dy},{1.0 if (dx, dy) != (0.0, 0.25) else 0.9}\n")
        if variant == "flags_gate": pj["qa_vs_lr_tif"]["flags"] = ["registration_shift"]       # argmin 0.25 cell -> gate says no flag
        if variant == "frame_offset": pj["frame_offset"] = dict(applied=True, dx_m=12.5, dy_m=0.0, basis="none", evidence="fitted")
        if variant == "missing_key": del pj["frame_offset"]
    else:
        if variant == "unavailable_with_raster":
            with rasterio.open(os.path.join(d, "ship_count.tif"), "w", **prof) as ds: ds.write(lr, 1)
    json.dump(pj, open(os.path.join(d, "products.json"), "w"), indent=1)
    return pdir

def run_tests():
    with tempfile.TemporaryDirectory() as root:
        ok = validate_pair(_write_synthetic(root, "SYN_OK"), "SYN_OK", usable_manifest=True); assert not ok["errors"], ok["errors"]
        un = validate_pair(_write_synthetic(root, "SYN_UNAVAIL", unavailable=True), "SYN_UNAVAIL", usable_manifest=False); assert not un["errors"], un["errors"]
        bad = {}
        for v in ("grid_mismatch", "sd_where_count1", "rsd_missing", "xtrack_range", "dtype", "flags_gate", "frame_offset", "missing_key"):
            r = validate_pair(_write_synthetic(root, f"SYN_{v}", variant=v), f"SYN_{v}", usable_manifest=True); bad[v] = r["errors"]; assert r["errors"], f"non-conforming variant {v} passed"
        r = validate_pair(_write_synthetic(root, "SYN_UA2", unavailable=True, variant="unavailable_with_raster"), "SYN_UA2", usable_manifest=False); assert r["errors"]; bad["unavailable_with_raster"] = r["errors"]
        r = validate_pair(_write_synthetic(root, "SYN_OK2"), "SYN_OK2", usable_manifest=True); r2 = validate_pair(_write_synthetic(root, "SYN_FL"), "SYN_FL", usable_manifest=True)
        # usable gate: manifest says usable but products flagged
        pj = os.path.join(root, "SYN_FL", "ship_products_v2_1", "products.json"); p = json.load(open(pj)); p["qa_vs_lr_tif"]["flags"] = ["vertical_offset"]; json.dump(p, open(pj, "w"))
        r3 = validate_pair(os.path.join(root, "SYN_FL"), "SYN_FL", usable_manifest=True); assert any("usable" in e for e in r3["errors"]), r3["errors"]; bad["usable_gate"] = r3["errors"]
    print("UNIT TESTS PASS: conforming (available) 0 errors; conforming (unavailable) 0 errors; non-conforming variants each rejected:")
    for k, v in bad.items(): print(f"   {k}: {v}")
    return bad

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--manifest"); ap.add_argument("--pairs", default=None); ap.add_argument("--out", default="."); ap.add_argument("--test", action="store_true"); a = ap.parse_args()
    if a.test: run_tests(); sys.exit(0)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import r04_common as C
    m = C.start("contract_v2_1_validate") if a.manifest is None else pd.read_parquet(a.manifest)
    pids = a.pairs.split(",") if a.pairs else C.DEV_PAIRS + C.EW0207["lu16"]; C.assert_no_sealed(pids); out = []
    for pid in pids:
        row = C.pair_row(m, pid); r = validate_pair(os.path.join(C.DATA, os.path.dirname(str(row["harmonized_path_lr"]))), pid, row["ship_products_usable"])
        r["info"].update(unit=C.UNIT_OF.get(pid), usable_manifest_raw=str(row["ship_products_usable"]), reason=str(row["ship_products_usable_reason"])); out.append(r)
        print(f"  {pid:48s} available={r['info'].get('available')} flags={r['info'].get('flags')} usable(manifest)={r['info'].get('usable_manifest')} errors={r['errors']}", flush=True)
    os.makedirs(a.out, exist_ok=True); json.dump(out, open(f"{a.out}/contract_v2_1_validation.json", "w"), indent=1, default=str)
    md = ["| pair | unit | available | v2.1 flags | usable (manifest) | usable (rule) | errors |", "|---|---|---|---|---|---|---|"]
    for r in out:
        i = r["info"]; md.append(f"| {i['pair_id']} | {i['unit']} | {i.get('available')} | {','.join(i.get('flags') or []) if i.get('flags') is not None else '—'} | {i.get('usable_manifest')} | {i.get('usable_rule')} | {'; '.join(r['errors']) or 'none'} |")
    open(f"{a.out}/contract_v2_1_validation.md", "w").write("\n".join(md) + "\n")
    bad = [r["info"]["pair_id"] for r in out if r["errors"]]; print(f"{len(out)} pairs validated, {len(bad)} with errors: {bad}")
    sys.exit(1 if bad else 0)
