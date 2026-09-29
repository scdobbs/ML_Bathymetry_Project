"""Shared helpers for ingesting the co-located AUV/ship dataset.

Everything here is driven by `manifest/pairs.parquet`. The cardinal rules
(TRAINING_PHASE_handoff_brief_2026-06-26.md §2, §8):
  - LR is a real, independent ship-multibeam raster on disk. NEVER synthesize
    it from HR. Resampling is for CRS/grid alignment only.
  - Open HR with masked reads; -9999 / NaN fill must become invalid, never read
    as elevation.
  - Use the provided hr_valid/lr_valid/joint_valid masks when present. They are
    MISSING for the 21 validated-anchor pairs (18 cal_dig + ccz + discol + tag);
    for those, derive validity from nodata + finite checks here. (Escalated in
    report 15 — generating canonical masks for the anchors is the cleaner fix.)

HR and LR are separate grids (often slightly offset, always different
resolution), so LR is always read into the HR window by geo-referenced
resampling, never by pixel indexing.
"""

import json
import os

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import Window, from_bounds


# ---------------------------------------------------------------------------
# Manifest / path resolution
# ---------------------------------------------------------------------------

def load_manifest(data_root: str):
    import pandas as pd
    return pd.read_parquet(os.path.join(data_root, "manifest", "pairs.parquet"))


def load_k_map(data_root: str) -> dict:
    """pair_id -> max_recoverable_k (may be None for unrecoverable pairs)."""
    sweep = json.load(open(os.path.join(data_root, "manifest", "stage_c2c_sweep.json")))
    return {p["pair_id"]: p.get("max_recoverable_k") for p in sweep["pairs"]}


def resolve_pair_paths(data_root: str, row) -> dict:
    """Resolve a manifest row to absolute paths.

    As of the 2026-06-29 OAK refresh the masks are first-class, checksummed
    manifest columns (`hr_valid`, `lr_valid`, `joint_valid`, all root-relative)
    and exist for all 39 pairs. We resolve from those columns when present and
    fall back to filename-next-to-hr only for older manifests. `lr_valid_semantic`
    distinguishes `coverage_geotiff` (21 anchors — validity is a survey-coverage
    polygon) from `real_sounding` (18 reaudit pairs — per-sounding validity).
    """
    def col_path(col, fallback_name):
        val = row[col] if (col in row and isinstance(row[col], str) and row[col]) else None
        if val:
            p = os.path.join(data_root, val)
            return p if os.path.exists(p) else None
        p = os.path.join(os.path.dirname(os.path.join(data_root, row["harmonized_path_hr"])),
                         fallback_name)
        return p if os.path.exists(p) else None

    hr = os.path.join(data_root, row["harmonized_path_hr"])
    lr = os.path.join(data_root, row["harmonized_path_lr"])
    jv = col_path("joint_valid", "joint_valid.tif")
    return {
        "hr": hr,
        "lr": lr,
        "hr_valid": col_path("hr_valid", "hr_valid.tif"),
        "lr_valid": col_path("lr_valid", "lr_valid.tif"),
        "joint_valid": jv,
        "has_masks": jv is not None,
        "lr_valid_semantic": row["lr_valid_semantic"] if "lr_valid_semantic" in row else None,
        "checksum_joint_valid": row["checksum_joint_valid"] if "checksum_joint_valid" in row else None,
    }


def verify_mask_checksums(data_root: str, row) -> dict:
    """Verify the on-disk masks against the manifest checksums (anchor protection,
    brief §8 / directive 16 §2.4). Returns {mask_name: bool|None}; None if the
    manifest has no checksum for that mask."""
    import hashlib
    paths = resolve_pair_paths(data_root, row)
    out = {}
    for name, col in (("hr_valid", "checksum_hr_valid"),
                      ("lr_valid", "checksum_lr_valid"),
                      ("joint_valid", "checksum_joint_valid")):
        expected = row[col] if (col in row and isinstance(row[col], str) and row[col]) else None
        p = paths.get(name)
        if expected is None or p is None:
            out[name] = None
            continue
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        out[name] = (h.hexdigest() == expected)
    return out


# ---------------------------------------------------------------------------
# Validity
# ---------------------------------------------------------------------------

def _finite_valid(arr: np.ndarray, nodata) -> np.ndarray:
    v = np.isfinite(arr)
    if nodata is not None and np.isfinite(nodata):
        v &= (arr != nodata)
    return v


def compute_joint_valid(paths: dict):
    """Return (joint_valid uint8 [H,W], profile) at the HR grid.

    If joint_valid.tif exists, read it (encoded 1=valid). Otherwise derive:
        hr_valid  = finite(hr) & hr != nodata        (HR grid, native)
        lr_valid@HR = finite(lr) resampled (nearest) onto HR grid
        joint     = hr_valid & lr_valid@HR
    """
    with rasterio.open(paths["hr"]) as hr_src:
        H, W = hr_src.height, hr_src.width
        hr_profile = hr_src.profile
        hr_transform = hr_src.transform
        hr_crs = hr_src.crs

        if paths["joint_valid"] is not None:
            with rasterio.open(paths["joint_valid"]) as jv:
                jvarr = jv.read(1)
                jv_nodata = jv.nodata
            # masks encode valid=1; nodata (255) -> invalid
            joint = (jvarr == 1).astype(np.uint8)
            return joint, hr_profile

        # Derive. HR validity from the elevation band.
        hr = hr_src.read(1).astype(np.float32)
        hr_valid = _finite_valid(hr, hr_src.nodata)

    # LR validity resampled onto the HR grid (nearest — validity is categorical).
    with rasterio.open(paths["lr"]) as lr_src:
        lr = lr_src.read(1).astype(np.float32)
        lr_valid_native = _finite_valid(lr, lr_src.nodata).astype(np.uint8)
        lr_valid_hr = np.zeros((H, W), dtype=np.uint8)
        reproject(
            source=lr_valid_native,
            destination=lr_valid_hr,
            src_transform=lr_src.transform, src_crs=lr_src.crs,
            dst_transform=hr_transform, dst_crs=hr_crs,
            resampling=Resampling.nearest,
        )
    joint = (hr_valid & (lr_valid_hr == 1)).astype(np.uint8)
    return joint, hr_profile


# ---------------------------------------------------------------------------
# Tiling
# ---------------------------------------------------------------------------

def iter_tiles(H: int, W: int, tile_px: int, stride: int):
    """Yield (row0, col0) top-left corners for tile_px windows over [H, W]."""
    for r in range(0, H - tile_px + 1, stride):
        for c in range(0, W - tile_px + 1, stride):
            yield r, c


def tile_bounds(transform, row0: int, col0: int, tile_px: int):
    """Geo bounds (minx, miny, maxx, maxy) of a tile window in the raster CRS."""
    win = Window(col0, row0, tile_px, tile_px)
    b = rasterio.windows.bounds(win, transform)
    return b  # (left, bottom, right, top)


# ---------------------------------------------------------------------------
# Reading an HR tile + co-registered LR
# ---------------------------------------------------------------------------

def read_hr_tile(hr_src, row0, col0, tile_px):
    """Read an HR window as float32 with nodata -> NaN."""
    win = Window(col0, row0, tile_px, tile_px)
    hr = hr_src.read(1, window=win, boundless=True, fill_value=np.nan).astype(np.float32)
    nod = hr_src.nodata
    if nod is not None and np.isfinite(nod):
        hr = np.where(hr == nod, np.nan, hr)
    return hr


def read_raster_for_bounds(src, bounds, out_px, dst_crs, resampling=Resampling.average):
    """Resample any raster band onto an (out_px, out_px) grid covering `bounds`
    (left, bottom, right, top) in dst_crs. nodata -> NaN. Used to co-register
    HR (at target_res) and LR (at lr_native) onto the SAME geographic tile,
    guaranteeing pixel-for-pixel alignment regardless of the source grids.
    """
    left, bottom, right, top = bounds
    dst_transform = rasterio.transform.from_bounds(left, bottom, right, top, out_px, out_px)
    dst = np.full((out_px, out_px), np.nan, dtype=np.float32)
    s = src.read(1).astype(np.float32)
    nod = src.nodata
    if nod is not None and np.isfinite(nod):
        s = np.where(s == nod, np.nan, s)
    reproject(
        source=s, destination=dst,
        src_transform=src.transform, src_crs=src.crs,
        dst_transform=dst_transform, dst_crs=dst_crs,
        resampling=resampling, src_nodata=np.nan, dst_nodata=np.nan,
    )
    return dst


def load_band(src):
    """Read a raster band ONCE into memory as float32 with nodata→NaN, plus its
    transform/crs. Amortizes the read across all tiles of a pair (the per-tile
    read_raster_for_bounds re-reads the whole raster every call — fatal at ~90k
    tiles). Returns (arr, transform, crs)."""
    arr = src.read(1).astype(np.float32)
    nod = src.nodata
    if nod is not None and np.isfinite(nod):
        arr = np.where(arr == nod, np.nan, arr)
    return arr, src.transform, src.crs


def reproject_from_array(arr, src_transform, src_crs, bounds, out_px, dst_crs,
                         resampling=Resampling.average):
    """Reproject a preloaded array onto an (out_px, out_px) grid over `bounds`
    (in-memory; no disk read). Same result as read_raster_for_bounds."""
    left, bottom, right, top = bounds
    dst_transform = rasterio.transform.from_bounds(left, bottom, right, top, out_px, out_px)
    dst = np.full((out_px, out_px), np.nan, dtype=np.float32)
    reproject(source=arr, destination=dst, src_transform=src_transform, src_crs=src_crs,
              dst_transform=dst_transform, dst_crs=dst_crs, resampling=resampling,
              src_nodata=np.nan, dst_nodata=np.nan)
    return dst


def read_lr_for_window(lr_src, hr_src, row0, col0, tile_px, lr_out_px):
    """Read the LR raster over an HR tile's geographic footprint, resampled to
    a (lr_out_px, lr_out_px) grid. LR and HR share target_crs, so this is a
    pure resample onto a regular grid covering the tile bounds.

    Returns float32 with nodata -> NaN.
    """
    bounds = tile_bounds(hr_src.transform, row0, col0, tile_px)  # left,bottom,right,top
    left, bottom, right, top = bounds
    # Destination transform: lr_out_px cells spanning the tile bounds.
    dst_transform = rasterio.transform.from_bounds(
        left, bottom, right, top, lr_out_px, lr_out_px)
    dst = np.full((lr_out_px, lr_out_px), np.nan, dtype=np.float32)
    src = lr_src.read(1).astype(np.float32)
    src_nod = lr_src.nodata
    if src_nod is not None and np.isfinite(src_nod):
        src = np.where(src == src_nod, np.nan, src)
    reproject(
        source=src,
        destination=dst,
        src_transform=lr_src.transform, src_crs=lr_src.crs,
        dst_transform=dst_transform, dst_crs=hr_src.crs,
        resampling=Resampling.average,
        src_nodata=np.nan, dst_nodata=np.nan,
    )
    return dst
