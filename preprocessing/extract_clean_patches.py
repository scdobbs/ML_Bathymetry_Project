"""R02 §1 — CLEAN patch set: nothing derived from the AUV (HR) grid reaches the model inputs.

New module (R02 §1.1). Does NOT touch the code path that produced `patches_native_phase2`
(`extract_colocated_patches.process_pair_native`); it copies that function's tile enumeration and
rejection rules verbatim so the tile footprints (pair, r0, c0) are identical, and replaces the
normalization:

    inputs  (ship side only)             labels (AUV side, stored, never an input)
    ------------------------------       ------------------------------------------------
    lr_raw_m  LR on the tile's LR grid   delta   = median(HR - lr_up_m) over joint-valid px
    mu_lr     = mean(lr_raw_m | lr_valid)  y     = (HR - delta - mu_lr) / s_lr  (HR-valid px)
    s_lr      = max(std(lr_raw_m | lr_valid), s_floor)
    lr_up_m   = bicubic(where(lr_valid, lr_raw_m, mu_lr))
    x         = (lr_up_m - mu_lr) / s_lr             -> input channel 1
    lr_valid_hr (nearest-upsampled lr_valid)          -> input channel 2
    scale     = [log2(lr_native), log2(hr_res)]       (manifest grid spacings)

Tile selection keeps the original rules (joint-valid >= min_valid, HR-valid >= min_valid,
HR std >= flat_threshold_m). The last two use the AUV grid for SELECTION only; they are label-side
choices identical to the submitted pipeline and are recorded per tile.

`build_inputs()` takes only ship-side arrays, so the leak test (R02 §1.3) can regenerate the inputs
with the HR grid replaced by noise and compare bytes.
"""
from __future__ import annotations

import argparse, hashlib, json, math, os, time
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling

from preprocessing.colocated_utils import (load_manifest, load_k_map, resolve_pair_paths,
                                           compute_joint_valid, load_band, reproject_from_array)
from preprocessing.data_utils import bicubic_upsample

NATIVE_PAIRS = ["NA090__MGDS_31212", "NA080__MGDS_31290", "TN399__MGDS_30373", "RR1506__MGDS_29779",
                "AT37-13__MGDS_31199", "EW0207__MGDS_32556", "FK181031__MGDS_24618",
                "TN299__MGDS_27339", "TN299__MGDS_31253"]


def nearest_up(mask_lr: np.ndarray, hr_px: int) -> np.ndarray:
    """Nearest-neighbour upsample of an (L,L) mask to (hr_px,hr_px); same index rule as
    torch F.interpolate(mode='nearest') used by the submitted loader (floor(i*L/H))."""
    L = mask_lr.shape[0]
    ci = (np.arange(hr_px) * L) // hr_px
    return mask_lr[ci][:, ci]


def build_inputs(lr_raw_m: np.ndarray, lr_valid: np.ndarray, hr_px: int, s_floor: float):
    """SHIP-SIDE ONLY. Returns dict(x, lr_valid_hr, lr_up_m, mu_lr, s_lr, s_floor_hit, n_lr_valid).
    No HR array is accepted by this function (leak test relies on this)."""
    lv = lr_valid.astype(bool)
    n = int(lv.sum())
    if n < 2:
        return None
    vals = lr_raw_m[lv].astype(np.float64)
    mu = float(vals.mean())
    sd = float(vals.std())
    s = max(sd, s_floor)
    lr_filled = np.where(lv, lr_raw_m, mu).astype(np.float32)
    lr_up_m = bicubic_upsample(lr_filled, hr_px).astype(np.float32)
    x = ((lr_up_m - mu) / s).astype(np.float32)
    return dict(x=x, lr_valid_hr=nearest_up(lv, hr_px), lr_up_m=lr_up_m, mu_lr=mu, s_lr=float(s),
                s_floor_hit=bool(sd < s_floor), n_lr_valid=n)


def build_target(hr_m: np.ndarray, hr_valid: np.ndarray, inp: dict):
    """AUV SIDE (labels only). y = (HR - delta - mu_lr)/s_lr on HR-valid px, 0 elsewhere."""
    hv = hr_valid.astype(bool)
    joint = hv & inp["lr_valid_hr"]
    if joint.sum() >= 64:
        delta = float(np.median((hr_m - inp["lr_up_m"])[joint]))
        delta_ok = True
    else:
        delta, delta_ok = 0.0, False
    y = np.where(hv, (hr_m - delta - inp["mu_lr"]) / inp["s_lr"], 0.0).astype(np.float32)
    return dict(y=y, delta=delta, delta_ok=delta_ok, joint=joint,
                hr_mean_m=float(np.nanmean(hr_m[hv])) if hv.any() else np.nan,
                hr_std_m=float(np.nanstd(hr_m[hv])) if hv.any() else np.nan)


def pair_setup(data_root, row, hr_px):
    """Geometry identical to process_pair_native (extract_colocated_patches.py:240-262)."""
    paths = resolve_pair_paths(data_root, row)
    lr_native = float(row["lr_native_res_m"])
    joint, _ = compute_joint_valid(paths)
    with rasterio.open(paths["hr"]) as hr_src:
        H, W = hr_src.height, hr_src.width
        hr_res = abs(hr_src.transform.a)
        transform = hr_src.transform
        dst_crs = hr_src.crs
    lr_px = max(2, int(round(hr_px * hr_res / lr_native)))
    scale_vec = np.array([math.log2(lr_native), math.log2(hr_res)], dtype=np.float32)
    with rasterio.open(paths["hr"]) as hr_src, rasterio.open(paths["lr"]) as lr_src, \
         rasterio.open(paths["hr_valid"]) as hrv_src, rasterio.open(paths["lr_valid"]) as lrv_src:
        hr_arr, _, _ = load_band(hr_src)
        hrv_arr = (hrv_src.read(1) == 1)
        lr_arr, lr_tr, lr_crs = load_band(lr_src)
        lrv_raw = lrv_src.read(1).astype(np.float32)
        lrv_tr, lrv_crs = lrv_src.transform, lrv_src.crs
    return dict(paths=paths, lr_native=lr_native, joint=joint, H=H, W=W, hr_res=hr_res, transform=transform,
                dst_crs=dst_crs, lr_px=lr_px, scale_vec=scale_vec, hr_arr=hr_arr, hrv_arr=hrv_arr,
                lr_arr=lr_arr, lr_tr=lr_tr, lr_crs=lr_crs, lrv_raw=lrv_raw, lrv_tr=lrv_tr, lrv_crs=lrv_crs,
                semantic=paths.get("lr_valid_semantic") or "")


def tile_ship_side(g, r0, c0, win):
    """LR tile over the exact HR window footprint: area-average onto lr_px cells; validity nearest."""
    b = rasterio.windows.bounds(rasterio.windows.Window(c0, r0, win, win), g["transform"])
    lr = reproject_from_array(g["lr_arr"], g["lr_tr"], g["lr_crs"], b, g["lr_px"], g["dst_crs"], Resampling.average)
    lr_valid = reproject_from_array(g["lrv_raw"], g["lrv_tr"], g["lrv_crs"], b, g["lr_px"], g["dst_crs"],
                                    Resampling.nearest) == 1
    lr_valid = lr_valid & np.isfinite(lr)
    lr = np.where(lr_valid, lr, np.nan).astype(np.float32)
    return lr, lr_valid


def process_pair_clean(data_root, row, k, split, out_dir, hr_px, stride_frac, min_valid, flat_threshold_m,
                       s_floor, train_eligible=True, only_windows=None):
    g = pair_setup(data_root, row, hr_px)
    H, W, win = g["H"], g["W"], int(hr_px)
    if win < 8 or win > min(H, W):
        return (row["pair_id"], 0, 0, f"win={win} out of range", [])
    step = max(1, int(round(win * stride_frac)))
    split_dir = os.path.join(out_dir, split); os.makedirs(split_dir, exist_ok=True)
    terrain = str(row["terrain_class"]) if "terrain_class" in row else ""
    leakage_unit = str(row["leakage_unit"]) if "leakage_unit" in row else ""
    rows_out = []; saved = rejected = 0; rej = dict(joint=0, hr_valid=0, flat=0, lr_valid=0)
    windows = only_windows if only_windows is not None else \
        [(r0, c0) for r0 in range(0, H - win + 1, step) for c0 in range(0, W - win + 1, step)]
    for r0, c0 in windows:
        if g["joint"][r0:r0 + win, c0:c0 + win].mean() < min_valid:
            rejected += 1; rej["joint"] += 1; continue
        hr = g["hr_arr"][r0:r0 + win, c0:c0 + win].copy()
        hr_valid = g["hrv_arr"][r0:r0 + win, c0:c0 + win].copy() & np.isfinite(hr)
        hr = np.where(hr_valid, hr, np.nan)
        if hr_valid.mean() < min_valid:
            rejected += 1; rej["hr_valid"] += 1; continue
        hr_std = float(np.nanstd(hr))
        if not np.isfinite(hr_std) or hr_std < flat_threshold_m:       # label-side selection (unchanged rule)
            rejected += 1; rej["flat"] += 1; continue
        lr, lr_valid = tile_ship_side(g, r0, c0, win)
        inp = build_inputs(lr, lr_valid, hr_px, s_floor)
        if inp is None:
            rejected += 1; rej["lr_valid"] += 1; continue
        tgt = build_target(hr, hr_valid, inp)
        fname = f"{row['pair_id']}_r{r0:06d}_c{c0:06d}.npz"
        stats = np.array([inp["mu_lr"], inp["s_lr"], tgt["delta"], tgt["hr_mean_m"], tgt["hr_std_m"],
                          float(k), g["lr_native"], g["hr_res"], float(inp["s_floor_hit"])], dtype=np.float32)
        np.savez_compressed(
            os.path.join(split_dir, fname),
            x=inp["x"][np.newaxis], lr_valid_hr=inp["lr_valid_hr"][np.newaxis].astype(np.uint8),
            y=tgt["y"][np.newaxis], hr_valid=hr_valid[np.newaxis].astype(np.uint8),
            lr_raw_m=np.where(lr_valid, lr, np.nan)[np.newaxis].astype(np.float32),
            lr_valid=lr_valid[np.newaxis].astype(np.uint8),
            stats=stats, scale=g["scale_vec"], train_eligible=np.uint8(1 if train_eligible else 0),
            lr_valid_semantic=np.array(g["semantic"]), pair_id=np.array(row["pair_id"]),
            terrain_class=np.array(terrain), leakage_unit=np.array(leakage_unit), split=np.array(split),
            stats_doc=np.array("[mu_lr, s_lr, delta, hr_mean_m, hr_std_m, k, lr_native, hr_res, s_floor_hit]"))
        rows_out.append(dict(file=fname, split=split, pair_id=row["pair_id"], r0=int(r0), c0=int(c0), k=int(k),
                             leakage_unit=leakage_unit, mu_lr=inp["mu_lr"], s_lr=inp["s_lr"], s_floor_hit=inp["s_floor_hit"],
                             delta=tgt["delta"], delta_ok=tgt["delta_ok"], hr_std_m=tgt["hr_std_m"],
                             n_lr_valid=inp["n_lr_valid"], frac_hr_valid=float(hr_valid.mean()),
                             frac_joint=float(tgt["joint"].mean()), max_abs_x=float(np.abs(inp["x"]).max()),
                             max_abs_y=float(np.abs(tgt["y"][hr_valid]).max())))
        saved += 1
    return (row["pair_id"], saved, rejected, f"CLEAN k={k} lr_px={g['lr_px']} win={win} step={step} rej={rej} [{split}]", rows_out)


def tile_inputs_full_path(g, r0, c0, win, hr_px, s_floor):
    """The complete per-tile path as run in process_pair_clean, returning only the model inputs.
    Used by the leak test with g['hr_arr'] swapped for noise."""
    hr = g["hr_arr"][r0:r0 + win, c0:c0 + win].copy()
    hr_valid = g["hrv_arr"][r0:r0 + win, c0:c0 + win].copy() & np.isfinite(hr)
    hr = np.where(hr_valid, hr, np.nan)
    lr, lr_valid = tile_ship_side(g, r0, c0, win)
    inp = build_inputs(lr, lr_valid, hr_px, s_floor)
    tgt = build_target(hr, hr_valid, inp)          # computed so the label path also runs, then discarded
    return inp["x"], inp["lr_valid_hr"], g["scale_vec"], tgt["delta"]


def leak_test(data_root, pairs_df, kmap, index_df, n_tiles, hr_px, s_floor, seed=0):
    """R02 §1.3: run the full tile path twice, once with the true HR grid and once with the HR grid
    replaced by random noise of the same shape and validity; the model inputs (x, lr_valid_hr, scale)
    must be byte-identical. Also checks the stored npz inputs equal the regenerated ones."""
    rng = np.random.default_rng(seed)
    sample = index_df.sample(n=min(n_tiles, len(index_df)), random_state=seed)
    hb = lambda arr: hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()
    results = []
    for pid, sub in sample.groupby("pair_id"):
        row = pairs_df[pairs_df.pair_id == pid].iloc[0]
        g = pair_setup(data_root, row, hr_px)
        g_noise = dict(g)
        noise = rng.standard_normal(g["hr_arr"].shape).astype(np.float32) * 100.0 - 2000.0
        g_noise["hr_arr"] = np.where(np.isfinite(g["hr_arr"]), noise, np.nan)
        for _, t in sub.iterrows():
            r0, c0, win = int(t.r0), int(t.c0), hr_px
            xa, va, sa, da = tile_inputs_full_path(g, r0, c0, win, hr_px, s_floor)
            xb, vb, sb, db = tile_inputs_full_path(g_noise, r0, c0, win, hr_px, s_floor)
            same = (hb(xa) == hb(xb)) and (hb(va) == hb(vb)) and (hb(sa) == hb(sb))
            z = np.load(os.path.join(index_df.attrs["out_dir"], t.split, t.file), allow_pickle=True)
            stored_same = (hb(z["x"][0]) == hb(xa)) and (hb(z["lr_valid_hr"][0]) == hb(va)) and (hb(z["scale"]) == hb(sa))
            results.append(dict(pair_id=pid, r0=r0, c0=c0, inputs_identical_true_vs_noise=bool(same),
                                stored_equals_regenerated=bool(stored_same),
                                delta_true=float(da), delta_noise=float(db)))   # labels DO change: proves noise reached the label path
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", default="/oak/stanford/groups/hilley/auv_ship_colocated_bathy")
    ap.add_argument("--split_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--pairs", default=",".join(NATIVE_PAIRS))
    ap.add_argument("--hr_px", type=int, default=256)
    ap.add_argument("--train_stride_frac", type=float, default=0.25)
    ap.add_argument("--eval_stride_frac", type=float, default=0.5)
    ap.add_argument("--min_valid", type=float, default=0.50)
    ap.add_argument("--flat_threshold_m", type=float, default=0.05)
    ap.add_argument("--s_floor", type=float, default=0.5)
    ap.add_argument("--leak_test_tiles", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    pairs = load_manifest(args.data_root); kmap = load_k_map(args.data_root)
    split_df = pd.read_parquet(args.split_path).set_index("pair_id")["split"].to_dict()
    want = [p for p in args.pairs.split(",") if p]
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time(); summary = []; index = []

    from concurrent.futures import ProcessPoolExecutor
    jobs = []
    for _, row in pairs[pairs.pair_id.isin(want)].iterrows():
        pid = row["pair_id"]; k = kmap.get(pid); split = split_df.get(pid, "train")
        stride = args.train_stride_frac if split == "train" else args.eval_stride_frac
        jobs.append((row, int(k) if k is not None else 2, split, stride, k is not None))
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as ex:
        futs = [ex.submit(process_pair_clean, args.data_root, row, k, split, args.out_dir, args.hr_px, stride,
                          args.min_valid, args.flat_threshold_m, args.s_floor, te) for row, k, split, stride, te in jobs]
        for f in futs:
            pid, saved, rejected, note, rows_out = f.result()
            print(f"  {pid:24s} saved={saved:6d} rejected={rejected:6d} {note}", flush=True)
            summary.append(dict(pair_id=pid, saved=saved, rejected=rejected, note=note)); index += rows_out
    idx = pd.DataFrame(index); idx.to_parquet(os.path.join(args.out_dir, "patch_index.parquet"))
    pd.DataFrame(summary).to_csv(os.path.join(args.out_dir, "pair_summary.csv"), index=False)
    print(f"patches written: {len(idx)} in {time.time()-t0:.0f}s", flush=True)

    if args.leak_test_tiles > 0:
        idx.attrs["out_dir"] = args.out_dir
        res = leak_test(args.data_root, pairs, kmap, idx, args.leak_test_tiles, args.hr_px, args.s_floor)
        n_ok = sum(r["inputs_identical_true_vs_noise"] and r["stored_equals_regenerated"] for r in res)
        json.dump(dict(n=len(res), n_pass=n_ok, tiles=res), open(os.path.join(args.out_dir, "leak_test.json"), "w"), indent=1)
        print(f"LEAK TEST: {n_ok}/{len(res)} tiles byte-identical (true HR vs noise HR; stored vs regenerated)", flush=True)
        if n_ok != len(res):
            raise SystemExit("LEAK TEST FAILED — escalation")


if __name__ == "__main__":
    main()
