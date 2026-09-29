"""Slicer for the co-located AUV/ship dataset (rewrite of the synthetic-LR slicer).

This is the §6/§9 rewrite mandated by TRAINING_PHASE_handoff_brief_2026-06-26.md.
It is driven entirely by `manifest/pairs.parquet`; it does NOT synthesize LR.

Geometry (the recommended design — see report 15 for the alternatives that are
Steve's/steering's call):
  - Per pair, the SR factor is the recorded `max_recoverable_k` (sweep json).
    Pairs with k=None (no recoverable SR signal) are skipped by default.
  - target_res = lr_native_res_m / k.   <-- per-pair absolute target resolution.
  - HR target patch is a FIXED `--hr_px` (default 256) grid AT target_res.
  - LR input patch is `hr_px // k` cells AT lr_native (the real ship multibeam,
    read by geo-window over the SAME tile footprint — never synthesized).
  - The tile footprint is T_m = hr_px * target_res; we step the HR-native grid
    in windows of that size and resample both rasters onto the tile.
  - The network is told the absolute scale via `scale = log2(target_res_m)`
    (cfg.scale_conditioning); per-pair k therefore varies the LR patch size and
    the residual/consistency pooling factor, while HR shape stays fixed.

Validity (§6, §8): a tile is kept iff its joint_valid fraction >= --min_valid.
joint_valid comes from the shipped mask when present, else is derived from
nodata (colocated_utils.compute_joint_valid). HR is read NaN-masked; the loss
must respect the saved hr_valid mask (written into each npz).

Split (§7): whole leakage units, from build_split.py's split_assignment.parquet.

Output .npz per patch (superset of the legacy schema; legacy keys preserved so
existing loaders keep working, new keys are additive):
    lr_raw   float32 (1, lr_px, lr_px)   real LR, per-patch z-scored
    lr_up    float32 (1, hr_px, hr_px)   bicubic(lr_raw)->hr_px, z-scored
    hr       float32 (1, hr_px, hr_px)   HR@target_res, z-scored
    hr_valid uint8   (1, hr_px, hr_px)   1 where HR is real (loss mask)
    lr_valid uint8   (1, lr_px, lr_px)   1 where LR is real
    stats    float32 (5,)  [mean, std, k, target_res_m, lr_native_res_m]
    scale    float32 ()    log2(target_res_m)   (model conditioning, brief §4)

Usage:
    python -m preprocessing.extract_colocated_patches \
        --data_root /oak/stanford/groups/hilley/auv_ship_colocated_bathy \
        --split_path /scratch/.../manifest/split_assignment.parquet \
        --out_dir   /scratch/.../patches_colocated_hr256 \
        --hr_px 256 --stride_frac 0.5 --min_valid 0.50
"""

import argparse
import math
import os

import numpy as np
import pandas as pd
import rasterio

from preprocessing.colocated_utils import (
    load_manifest, load_k_map, resolve_pair_paths, compute_joint_valid,
    read_raster_for_bounds, load_band, reproject_from_array,
)
from preprocessing.data_utils import normalize_patch, bicubic_upsample
from rasterio.enums import Resampling


def pair_hr_geometry(data_root, row, k):
    """Cheap geometry probe for a pair: (target_res, hr_res, H, W, max_hr_px).

    max_hr_px is the largest HR patch size whose model footprint
    (hr_px * target_res) still fits inside the AUV survey extent, i.e. the
    largest hr_px with win = hr_px*target_res/hr_res <= min(H, W). Pairs whose
    max_hr_px < the 128 floor cannot supply any patch at any allowed size
    (the report-17 EW9801/AR26 zero-patch case).
    """
    paths = resolve_pair_paths(data_root, row)
    lr_native = float(row["lr_native_res_m"])
    target_res = lr_native / k
    with rasterio.open(paths["hr"]) as hr_src:
        H, W = hr_src.height, hr_src.width
        hr_res = abs(hr_src.transform.a)
    max_hr_px = int(math.floor(min(H, W) * hr_res / target_res))
    return dict(target_res=target_res, hr_res=hr_res, H=H, W=W, max_hr_px=max_hr_px)


def plan_hr_px(data_root, pairs, kmap, unrecoverable_k=2,
               candidates=(256, 192, 128), floor_px=128):
    """Choose hr_px PER PAIR (directive 21 §3).

    Rule: hr_px = largest candidate in {256,192,128} with hr_px <= the pair's
    max_hr_px, floored at floor_px. The k-bucketed sampler keys on the shape
    tuple (k, hr_px) — NOT k alone — so different hr_px within a k still collate
    in their own shape bucket. This recovers footprint-starved pairs as their
    own buckets (EW9801 -> 128, KIWI10RR -> 192) at no cost to the 256 pairs.
    A pair whose max_hr_px < floor_px is too small for the 6-level U-Net
    bottleneck and is flagged (fits=False -> 0 patches).

    Returns (hr_px_by_pair: dict, plan_df: per-pair geometry DataFrame).
    """
    import pandas as pd
    cand = sorted(candidates, reverse=True)
    rows = []
    for _, row in pairs.iterrows():
        pid = row["pair_id"]
        k = kmap.get(pid)
        k_eff = int(k) if k is not None else int(unrecoverable_k)
        try:
            g = pair_hr_geometry(data_root, row, k_eff)
            mx, tr = g["max_hr_px"], g["target_res"]
        except Exception as e:
            rows.append(dict(pair_id=pid, k=k_eff, target_res=float("nan"),
                             max_hr_px=0, hr_px=floor_px, fits=False, note=f"geom-error {e}"))
            continue
        chosen = next((c for c in cand if c <= mx), floor_px)
        rows.append(dict(pair_id=pid, k=k_eff, target_res=tr, max_hr_px=mx,
                         hr_px=int(chosen), fits=bool(mx >= floor_px), note=""))
    plan_df = pd.DataFrame(rows)
    # keep bucket_hr_px == hr_px for back-compat with the figures/report code
    plan_df["bucket_hr_px"] = plan_df["hr_px"]
    hr_px_by_pair = dict(zip(plan_df["pair_id"], plan_df["hr_px"]))
    return hr_px_by_pair, plan_df


def process_pair(data_root, row, k, split, out_dir, hr_px, stride_frac,
                 min_valid, flat_threshold_m, train_eligible=True):
    paths = resolve_pair_paths(data_root, row)
    lr_native = float(row["lr_native_res_m"])
    target_res = lr_native / k
    lr_px = max(1, int(round(hr_px / k)))
    scale_val = math.log2(target_res)
    terrain = str(row["terrain_class"]) if "terrain_class" in row else ""
    leakage_unit = str(row["leakage_unit"]) if "leakage_unit" in row else ""

    joint, _ = compute_joint_valid(paths)            # uint8 [H, W] at HR grid
    with rasterio.open(paths["hr"]) as hr_src:
        H, W = hr_src.height, hr_src.width
        hr_res = abs(hr_src.transform.a)
        transform = hr_src.transform
        dst_crs = hr_src.crs
    # HR-native pixels spanned by one tile footprint (T_m = hr_px * target_res).
    win = int(round(hr_px * target_res / hr_res))
    if win < 8 or win > min(H, W):
        return (row["pair_id"], 0, 0, f"win={win} out of range (H={H},W={W})", [])
    step = max(1, int(round(win * stride_frac)))

    split_dir = os.path.join(out_dir, split)
    os.makedirs(split_dir, exist_ok=True)

    semantic = paths.get("lr_valid_semantic") or ""
    patch_rows = []   # one row per saved patch for the per-split index
    saved = rejected = 0
    with rasterio.open(paths["hr"]) as hr_src, rasterio.open(paths["lr"]) as lr_src, \
         rasterio.open(paths["hr_valid"]) as hrv_src, \
         rasterio.open(paths["lr_valid"]) as lrv_src:
        for r0 in range(0, H - win + 1, step):
            for c0 in range(0, W - win + 1, step):
                jv = joint[r0:r0 + win, c0:c0 + win]
                if jv.mean() < min_valid:
                    rejected += 1
                    continue
                # Geographic bounds of this tile (left, bottom, right, top).
                b = rasterio.windows.bounds(
                    rasterio.windows.Window(c0, r0, win, win), transform)
                hr = read_raster_for_bounds(hr_src, b, hr_px, dst_crs,
                                            Resampling.average)
                lr = read_raster_for_bounds(lr_src, b, lr_px, dst_crs,
                                            Resampling.average)
                # Per-patch validity from the CANONICAL masks (not finite-check):
                # for coverage_geotiff pairs the lr raster carries interpolated
                # fill that lr_valid excludes — finite() would wrongly keep it.
                # Nearest resampling preserves the categorical 1=valid encoding.
                hr_valid = read_raster_for_bounds(
                    hrv_src, b, hr_px, dst_crs, Resampling.nearest) == 1
                lr_valid = read_raster_for_bounds(
                    lrv_src, b, lr_px, dst_crs, Resampling.nearest) == 1
                # Tighten the canonical masks to also require FINITE data. The
                # coverage_geotiff anchor masks are survey-coverage polygons, so
                # lr_valid can mark a cell valid where the LR raster is actually
                # nodata/NaN (a gap inside the polygon) — that NaN would survive
                # the mean-fill (where(valid, nan, mean) == nan) and propagate
                # into the model as NaN. Intersecting with isfinite excludes
                # both polygon-invalid AND nodata-inside-polygon cells; it does
                # NOT reintroduce E1 (we keep the canonical mask, only AND it).
                hr_valid = hr_valid & np.isfinite(hr)
                lr_valid = lr_valid & np.isfinite(lr)
                # Mask elevation by validity so invalid/fill cells never enter
                # normalization stats or the bicubic upsample.
                hr = np.where(hr_valid, hr, np.nan)
                lr = np.where(lr_valid, lr, np.nan)
                if hr_valid.mean() < min_valid:
                    rejected += 1
                    continue
                hr_std = float(np.nanstd(hr))
                if not np.isfinite(hr_std) or hr_std < flat_threshold_m:
                    rejected += 1
                    continue

                # Per-patch z-score on HR stats; LR shares the HR mean/std.
                mean = float(np.nanmean(hr))
                std = hr_std if hr_std > 1e-6 else 1.0
                hr_filled = np.where(hr_valid, hr, mean)
                lr_filled = np.where(lr_valid, lr, mean)
                hr_norm = (hr_filled - mean) / std
                lr_norm = (lr_filled - mean) / std
                lr_up = bicubic_upsample(lr_filled, hr_px)
                lr_up_norm = (lr_up - mean) / std

                fname = f"{row['pair_id']}_r{r0:06d}_c{c0:06d}.npz"
                np.savez_compressed(
                    os.path.join(split_dir, fname),
                    lr_raw=lr_norm[np.newaxis].astype(np.float32),
                    lr_up=lr_up_norm[np.newaxis].astype(np.float32),
                    hr=hr_norm[np.newaxis].astype(np.float32),
                    hr_valid=hr_valid[np.newaxis].astype(np.uint8),
                    lr_valid=lr_valid[np.newaxis].astype(np.uint8),
                    stats=np.array([mean, std, float(k), target_res, lr_native],
                                   dtype=np.float32),
                    scale=np.float32(scale_val),
                    train_eligible=np.uint8(1 if train_eligible else 0),
                    lr_valid_semantic=np.array(semantic),
                    pair_id=np.array(row["pair_id"]),
                    terrain_class=np.array(terrain),
                    leakage_unit=np.array(leakage_unit),
                )
                patch_rows.append(dict(
                    file=fname, split=split, pair_id=row["pair_id"],
                    k=int(k), train_eligible=int(bool(train_eligible)),
                    terrain_class=terrain, leakage_unit=leakage_unit,
                    hr_px=int(hr_px), lr_px=int(lr_px)))
                saved += 1
    return (row["pair_id"], saved, rejected,
            f"k={k} target_res={target_res:.1f}m hr_px={hr_px} lr_px={lr_px} "
            f"win={win} step={step} [{split}]", patch_rows)


def process_pair_native(data_root, row, k, split, out_dir, hr_px, stride_frac,
                        min_valid, flat_threshold_m, context_factor=4.0,
                        lr_context_px=64, train_eligible=True):
    """NATIVE-target geometry (directive 25 §2). HR target = `hr_px` px at the
    HARMONIZED native resolution (NO downsample to target_res), so the fine
    relief is in the supervision signal. Two LR windows:
      - lr_raw   : LR over the EXACT tile footprint at lr_native res (small px,
                   `hr_px·hr_res/lr_native`) — residual baseline + consistency.
      - lr_context: LR over a `context_factor`× wider footprint, resampled to a
                   FIXED `lr_context_px` for EVERY pair — the cross-attention K/V
                   (fixes deep/high-k 4-px starvation; collatable corpus-wide).
    scale = [log2(lr_native), log2(hr_res)]  (encodes both the SR gap and the
    absolute output resolution — directive §3.1)."""
    paths = resolve_pair_paths(data_root, row)
    lr_native = float(row["lr_native_res_m"])
    terrain = str(row["terrain_class"]) if "terrain_class" in row else ""
    leakage_unit = str(row["leakage_unit"]) if "leakage_unit" in row else ""

    joint, _ = compute_joint_valid(paths)
    with rasterio.open(paths["hr"]) as hr_src:
        H, W = hr_src.height, hr_src.width
        hr_res = abs(hr_src.transform.a)
        transform = hr_src.transform
        dst_crs = hr_src.crs
    win = int(hr_px)                                   # native HR px, no target_res
    lr_px = max(2, int(round(hr_px * hr_res / lr_native)))
    scale_vec = np.array([math.log2(lr_native), math.log2(hr_res)], dtype=np.float32)
    if win < 8 or win > min(H, W):
        return (row["pair_id"], 0, 0, f"win={win} out of range (H={H},W={W})", [])
    step = max(1, int(round(win * stride_frac)))
    split_dir = os.path.join(out_dir, split)
    os.makedirs(split_dir, exist_ok=True)
    semantic = paths.get("lr_valid_semantic") or ""
    patch_rows = []
    saved = rejected = 0
    # Preload every band ONCE (the per-tile full-raster reads were the 10h
    # timeout). Under native geometry the HR tile is a DIRECT 256-px slice of the
    # HR grid (already at native res) — no reproject. LR reprojects from the small
    # preloaded LR array per tile.
    with rasterio.open(paths["hr"]) as hr_src, rasterio.open(paths["lr"]) as lr_src, \
         rasterio.open(paths["hr_valid"]) as hrv_src, \
         rasterio.open(paths["lr_valid"]) as lrv_src:
        hr_arr, _, _ = load_band(hr_src)
        hrv_arr = (hrv_src.read(1) == 1)
        lr_arr, lr_tr, lr_crs = load_band(lr_src)
        lrv_raw = lrv_src.read(1).astype(np.float32)
        lrv_tr, lrv_crs = lrv_src.transform, lrv_src.crs
    for r0 in range(0, H - win + 1, step):
        for c0 in range(0, W - win + 1, step):
            if joint[r0:r0 + win, c0:c0 + win].mean() < min_valid:
                rejected += 1
                continue
            b = rasterio.windows.bounds(
                rasterio.windows.Window(c0, r0, win, win), transform)
            left, bottom, right, top = b
            cx, cy = 0.5 * (left + right), 0.5 * (bottom + top)
            hw = 0.5 * context_factor * (right - left)
            b_ctx = (cx - hw, cy - hw, cx + hw, cy + hw)

            # HR + HR-valid: direct slice of the native grid (win == hr_px).
            hr = hr_arr[r0:r0 + win, c0:c0 + win].copy()
            hr_valid = hrv_arr[r0:r0 + win, c0:c0 + win].copy()
            lr = reproject_from_array(lr_arr, lr_tr, lr_crs, b, lr_px, dst_crs,
                                      Resampling.average)
            lr_ctx = reproject_from_array(lr_arr, lr_tr, lr_crs, b_ctx, lr_context_px,
                                          dst_crs, Resampling.average)
            lr_valid = reproject_from_array(lrv_raw, lrv_tr, lrv_crs, b, lr_px,
                                            dst_crs, Resampling.nearest) == 1
            hr_valid = hr_valid & np.isfinite(hr)
            lr_valid = lr_valid & np.isfinite(lr)
            hr = np.where(hr_valid, hr, np.nan)
            lr = np.where(lr_valid, lr, np.nan)
            if hr_valid.mean() < min_valid:
                rejected += 1
                continue
            hr_std = float(np.nanstd(hr))
            if not np.isfinite(hr_std) or hr_std < flat_threshold_m:
                rejected += 1
                continue
            mean = float(np.nanmean(hr))
            std = hr_std if hr_std > 1e-6 else 1.0
            hr_norm = (np.where(hr_valid, hr, mean) - mean) / std
            lr_filled = np.where(lr_valid, lr, mean)
            lr_norm = (lr_filled - mean) / std
            lr_up_norm = (bicubic_upsample(lr_filled, hr_px) - mean) / std
            # context: fill non-finite (outside coverage) with the patch mean.
            lr_ctx = np.where(np.isfinite(lr_ctx), lr_ctx, mean)
            lr_ctx_norm = (lr_ctx - mean) / std

            fname = f"{row['pair_id']}_r{r0:06d}_c{c0:06d}.npz"
            np.savez_compressed(
                os.path.join(split_dir, fname),
                lr_raw=lr_norm[np.newaxis].astype(np.float32),
                lr_up=lr_up_norm[np.newaxis].astype(np.float32),
                lr_context=lr_ctx_norm[np.newaxis].astype(np.float32),
                hr=hr_norm[np.newaxis].astype(np.float32),
                hr_valid=hr_valid[np.newaxis].astype(np.uint8),
                lr_valid=lr_valid[np.newaxis].astype(np.uint8),
                stats=np.array([mean, std, float(k), lr_native / k, lr_native, hr_res],
                               dtype=np.float32),
                scale=scale_vec,
                train_eligible=np.uint8(1 if train_eligible else 0),
                lr_valid_semantic=np.array(semantic),
                pair_id=np.array(row["pair_id"]),
                terrain_class=np.array(terrain),
                leakage_unit=np.array(leakage_unit),
            )
            patch_rows.append(dict(
                file=fname, split=split, pair_id=row["pair_id"], k=int(k),
                train_eligible=int(bool(train_eligible)), terrain_class=terrain,
                leakage_unit=leakage_unit, hr_px=int(hr_px), lr_px=int(lr_px)))
            saved += 1
    return (row["pair_id"], saved, rejected,
            f"NATIVE k={k} hr_res={hr_res:.2f}m hr_px={hr_px} lr_px={lr_px} "
            f"ctx={lr_context_px}@{context_factor:.0f}x win={win} step={step} [{split}]",
            patch_rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root",
                    default="/oak/stanford/groups/hilley/auv_ship_colocated_bathy")
    ap.add_argument("--split_path", required=True,
                    help="split_assignment.parquet from preprocessing.build_split")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--hr_px", type=int, default=0,
                    help="Force a single hr_px for ALL buckets (legacy). 0 (default) "
                         "auto-plans one hr_px per k-bucket via plan_hr_px (directive "
                         "19 §3.1): fine buckets k in {8,16} keep 256; coarser buckets "
                         "drop to the largest of {256,192,128} <= bucket median fit, "
                         "floored at 128.")
    ap.add_argument("--train_stride_frac", type=float, default=0.25,
                    help="Train tiles: 75%% overlap (directive 19 §3.2).")
    ap.add_argument("--eval_stride_frac", type=float, default=0.5,
                    help="Val/test tiles: 50%% overlap — calibration must not be "
                         "measured on artificially correlated tiles (directive 19 §3.2).")
    ap.add_argument("--stride_frac", type=float, default=0.0,
                    help="Override BOTH train and eval stride_frac with a single value "
                         "(0 = use the split-specific train/eval values above).")
    ap.add_argument("--min_valid", type=float, default=0.50)
    ap.add_argument("--flat_threshold_m", type=float, default=0.05)
    ap.add_argument("--unrecoverable_k", type=int, default=2,
                    help="Geometry k used to emit k=None pairs as eval-only "
                         "(data-preservation invariant, directive 16 §2.1). They "
                         "are tagged train_eligible=0 and excluded from the loss, "
                         "never dropped.")
    ap.add_argument("--only_pairs", type=str, default="",
                    help="Comma-separated pair_ids to process (smoke test).")
    ap.add_argument("--native", action="store_true",
                    help="NATIVE-target geometry (directive 25): hr_px px at native "
                         "resolution + dual LR windows. Forces hr_px=256, ignores plan_hr_px.")
    ap.add_argument("--native_hr_px", type=int, default=256)
    ap.add_argument("--context_factor", type=float, default=4.0,
                    help="lr_context footprint = this × the HR-target footprint.")
    ap.add_argument("--lr_context_px", type=int, default=64,
                    help="Fixed lr_context token grid for every pair (collatable).")
    args = ap.parse_args()

    pairs = load_manifest(args.data_root)
    kmap = load_k_map(args.data_root)
    split_df = pd.read_parquet(args.split_path).set_index("pair_id")["split"].to_dict()

    only = set(p for p in args.only_pairs.split(",") if p) if args.only_pairs else None

    # ---- Per-pair hr_px plan (directive 21 §3) ----
    if args.native:
        hr_px_by_pair = {pid: args.native_hr_px for pid in pairs["pair_id"]}
        plan_df = None
        print(f"NATIVE geometry: hr_px={args.native_hr_px} @ native res; "
              f"lr_context={args.lr_context_px}px @ {args.context_factor:.0f}× footprint.")
    elif args.hr_px and args.hr_px > 0:
        # Legacy: a single forced hr_px for every pair.
        hr_px_by_pair = {pid: args.hr_px for pid in pairs["pair_id"]}
        plan_df = None
        print(f"hr_px FORCED to {args.hr_px} for all pairs.")
    else:
        hr_px_by_pair, plan_df = plan_hr_px(args.data_root, pairs, kmap,
                                            unrecoverable_k=args.unrecoverable_k)
        print("Per-pair hr_px plan (directive 21 §3) — shape buckets (k, hr_px):")
        shp = plan_df.groupby(["k", "hr_px"]).size().reset_index(name="pairs")
        for _, r in shp.sort_values(["k", "hr_px"]).iterrows():
            print(f"  bucket (k={int(r['k'])}, hr_px={int(r['hr_px'])}): {int(r['pairs'])} pairs")
        bad = plan_df[~plan_df["fits"]]
        if len(bad):
            print("  PATHOLOGICAL (max_hr_px < 128 floor — 0 patches, flagged):")
            for _, r in bad.iterrows():
                print(f"    {r['pair_id']:50s} k={int(r['k'])} max_hr_px={int(r['max_hr_px'])}")
        # explicit recovery confirmation
        for pid in ("EW9801__MGDS_31425", "KIWI10RR__MGDS_24499"):
            r = plan_df[plan_df["pair_id"] == pid]
            if len(r):
                print(f"  RECOVERY: {pid} -> hr_px={int(r['hr_px'].iloc[0])} "
                      f"(max_hr_px={int(r['max_hr_px'].iloc[0])})")

    totals = {"train": 0, "val": 0, "test": 0}
    index = []   # one row per pair — proves no pair is silently absent (§2.1)
    patch_index = []   # one row per saved patch — drives the k-bucketed sampler
    errors = []
    for _, row in pairs.iterrows():
        pid = row["pair_id"]
        if only is not None and pid not in only:
            continue
        k = kmap.get(pid)
        # Data-preservation invariant (directive 16 §2.1): k=None pairs are
        # emitted as eval-only at a defined geometry, NEVER skipped.
        train_eligible = k is not None
        k_eff = int(k) if k is not None else int(args.unrecoverable_k)
        split = split_df.get(pid, "train")
        hr_px = int(hr_px_by_pair[pid])
        # Train tiles get denser overlap than val/test (directive 19 §3.2).
        if args.stride_frac and args.stride_frac > 0:
            stride_frac = args.stride_frac
        else:
            stride_frac = args.train_stride_frac if split == "train" else args.eval_stride_frac
        try:
            if args.native:
                _, saved, rejected, info, prows = process_pair_native(
                    args.data_root, row, k_eff, split, args.out_dir,
                    hr_px, stride_frac, args.min_valid, args.flat_threshold_m,
                    context_factor=args.context_factor, lr_context_px=args.lr_context_px,
                    train_eligible=train_eligible)
            else:
                _, saved, rejected, info, prows = process_pair(
                    args.data_root, row, k_eff, split, args.out_dir,
                    hr_px, stride_frac, args.min_valid, args.flat_threshold_m,
                    train_eligible=train_eligible)
        except Exception as e:
            errors.append((pid, f"ERROR {e}"))
            print(f"  ERROR {pid}: {e}", flush=True)
            continue
        totals[split] = totals.get(split, 0) + (saved if train_eligible else 0)
        tag = "train" if train_eligible else "eval_only"
        index.append({"pair_id": pid, "split": split, "tag": tag,
                      "k": k_eff, "hr_px": hr_px, "train_eligible": int(train_eligible),
                      "n_patches": saved})
        patch_index.extend(prows)
        print(f"  {pid:50s} saved={saved:5d} rej={rejected:6d} [{tag}]  {info}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    idx_df = pd.DataFrame(index)
    idx_path = os.path.join(args.out_dir, "emitted_index.parquet")
    idx_df.to_parquet(idx_path, index=False)
    if plan_df is not None:
        plan_df.to_parquet(os.path.join(args.out_dir, "hr_px_plan.parquet"), index=False)

    # Per-split patch index (drives the k-bucketed survey-balanced sampler).
    pidx_df = pd.DataFrame(patch_index)
    if len(pidx_df):
        for split in pidx_df["split"].unique():
            sub = pidx_df[pidx_df["split"] == split]
            sub.to_parquet(os.path.join(args.out_dir, split, "patch_index.parquet"),
                           index=False)

    print(f"\nSaved (train-eligible): train={totals['train']:,}  "
          f"val={totals['val']:,}  test={totals['test']:,}")
    print(f"Emitted index: {idx_path}  ({len(idx_df)} pairs; "
          f"{int((idx_df['tag']=='eval_only').sum())} eval_only)")
    if errors:
        print(f"ERRORS ({len(errors)}):")
        for pid, why in errors:
            print(f"  {pid}: {why}")
    print(f"Output: {args.out_dir}")


if __name__ == "__main__":
    main()
