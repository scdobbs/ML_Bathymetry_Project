#!/bin/bash
#SBATCH --job-name=bathy_convert_mc
#SBATCH --partition=serc
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/home/users/scdobbs/GAN_Bathymetry_Project/logs/convert_mc_%j.out
#SBATCH --error=/home/users/scdobbs/GAN_Bathymetry_Project/logs/convert_mc_%j.err

set -euo pipefail

module load python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312
source ~/bathy_diffusion_env/bin/activate

SRC_DIR="/scratch/users/scdobbs/MontereyCanyon_CCE"
OUT_DIR="/scratch/users/scdobbs/more_data_2026/high-res"
WORK_DIR="/scratch/users/scdobbs/monterey_tmp"

mkdir -p "$WORK_DIR"

echo "=== Converting Monterey Canyon .grd.gz → 2m GeoTIFF ==="
echo "Started: $(date)"

python3 - <<'PYEOF'
import os, glob, shutil
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling
import subprocess

SRC_DIR  = "/scratch/users/scdobbs/MontereyCanyon_CCE"
OUT_DIR  = "/scratch/users/scdobbs/more_data_2026/high-res"
WORK_DIR = "/scratch/users/scdobbs/monterey_tmp"

# Only convert raw bathymetry surveys, not difference maps
src_files = sorted(glob.glob(os.path.join(SRC_DIR, "Topo1m_*.grd.gz")))
print(f"Found {len(src_files)} Topo1m files to convert")

# CRS for Monterey Canyon CCE surveys (UTM Zone 10N, WGS84)
# Confirmed from bounds: easting ~597000-620000, northing ~4060000-4085000
TARGET_CRS = CRS.from_epsg(32610)   # WGS84 UTM Zone 10N
TARGET_RES = 2.0                     # metres

converted = 0
for gz_path in src_files:
    basename = os.path.basename(gz_path)
    stem = basename.replace('.grd.gz', '')
    out_name = stem.replace('Topo1m_', 'Bathymetry_2m_MC_') + '.tif'
    out_path = os.path.join(OUT_DIR, out_name)

    if os.path.exists(out_path):
        print(f"  [skip] already exists: {out_name}")
        converted += 1
        continue

    # Gunzip to temp location
    tmp_grd = os.path.join(WORK_DIR, stem + '.grd')
    print(f"  Extracting {basename} ...")
    result = subprocess.run(['gunzip', '-c', gz_path], stdout=open(tmp_grd, 'wb'),
                            check=True)

    # Open .grd (GDAL reads GMT NetCDF grids natively)
    with rasterio.open(tmp_grd) as src:
        src_transform = src.transform
        src_width  = src.width
        src_height = src.height
        src_bounds = src.bounds

        # Read data
        data = src.read(1).astype(np.float32)
        nodata = src.nodata

        # Assign CRS (was None in the file)
        # Compute output transform at 2m in the same CRS
        dst_transform, dst_width, dst_height = calculate_default_transform(
            TARGET_CRS, TARGET_CRS,
            src_width, src_height,
            left=src_bounds.left, bottom=src_bounds.bottom,
            right=src_bounds.right, top=src_bounds.top,
            resolution=TARGET_RES
        )

        dst_meta = {
            'driver': 'GTiff',
            'dtype': 'float32',
            'width': dst_width,
            'height': dst_height,
            'count': 1,
            'crs': TARGET_CRS,
            'transform': dst_transform,
            'nodata': np.nan,
            'compress': 'lzw',
        }

        print(f"  {stem}: {src_width}x{src_height} @ 1m  →  {dst_width}x{dst_height} @ 2m")

        with rasterio.open(out_path, 'w', **dst_meta) as dst:
            # Source profile with CRS assigned
            src_crs_meta = dict(
                crs=TARGET_CRS,
                transform=src_transform,
            )
            # Replace nodata sentinel with NaN before resampling
            if nodata is not None and np.isfinite(float(nodata)):
                data[data == nodata] = np.nan

            reproject(
                source=data,
                destination=rasterio.band(dst, 1),
                src_transform=src_transform,
                src_crs=TARGET_CRS,
                dst_transform=dst_transform,
                dst_crs=TARGET_CRS,
                resampling=Resampling.average,
                src_nodata=np.nan,
                dst_nodata=np.nan,
            )

    os.remove(tmp_grd)
    print(f"  Saved: {out_name}")
    converted += 1

print(f"\nDone: {converted}/{len(src_files)} files converted to {OUT_DIR}")
PYEOF

echo ""
echo "=== Conversion complete: $(date) ==="
echo "Output files:"
ls /scratch/users/scdobbs/more_data_2026/high-res/Bathymetry_2m_MC_*.tif 2>/dev/null || echo "  (none found)"
