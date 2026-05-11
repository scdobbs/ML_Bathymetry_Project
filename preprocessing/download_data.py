"""
Download supplementary training data for the bathymetry SR project.

Downloads:
  1. NOAA Coastal Relief Model (CRM) Vol. 7 — central California coast
     ~90m resolution, used as LR input for new HR surveys outside the
     existing Morro Bay 32m grid footprint. Resampled to 32m on save.

  2. USGS DS781 high-resolution (2m) bathymetry GeoTIFFs via ScienceBase:
     - Offshore Morro Bay  (P9HEZNRO)
     - + any additional blocks listed in DS781_ITEMS below

Usage:
    python preprocessing/download_data.py \
        --out_dir data_for_model/supplementary/ \
        --crm \
        --ds781

    # Download only CRM:
    python preprocessing/download_data.py --out_dir data_for_model/supplementary/ --crm

    # Download only DS781 HR surveys:
    python preprocessing/download_data.py --out_dir data_for_model/supplementary/ --ds781
"""

import os
import sys
import argparse
import requests
import numpy as np
from tqdm import tqdm
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS

# ---------------------------------------------------------------------------
# DS781 ScienceBase item DOIs — add more blocks here as needed
# Navigate to https://pubs.usgs.gov/ds/781/ to find other map areas
# ---------------------------------------------------------------------------
DS781_ITEMS = {
    'OffshoreMorroBay':       '10.5066/P9HEZNRO',
    # Add more blocks here, e.g.:
    # 'OffshoreCambria':      '10.5066/XXXXX',
    # 'OffshorePointConception': '10.5066/XXXXX',
    # 'MontereyCanyon':       '10.5066/XXXXX',
}

# ---------------------------------------------------------------------------
# NOAA CRM bounding box for central California (covers Morro Bay to Monterey)
# The CRM ERDDAP endpoint at NCEI provides subsetted NetCDF/GeoTIFF downloads
# Lat: 34.0–37.5N, Lon: -122.5–-119.5W
# ---------------------------------------------------------------------------
CRM_BBOX = {
    'lat_min': 34.0, 'lat_max': 37.5,
    'lon_min': -122.5, 'lon_max': -119.5,
}

# NCEI ERDDAP endpoint for the US Coastal Relief Model
# Dataset ID 'usgsCeSrtm30v6' is the CRM at ~90m (3 arc-second)
# Request GeoTIFF directly to avoid NetCDF4/HDF5 dependency
CRM_ERDDAP_URL = (
    "https://coastwatch.pfeg.noaa.gov/erddap/griddap/usgsCeSrtm30v6.geotif"
    "?topo[({lat_max}):1:({lat_min})]"
    "[({lon_min}):1:({lon_max})]"
)

SCIENCEBASE_API = "https://www.sciencebase.gov/catalog/item"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def download_file(url, dest_path, desc=None):
    """Stream-download a file with a progress bar."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    total = int(r.headers.get('content-length', 0))
    with open(dest_path, 'wb') as f, tqdm(
        desc=desc or os.path.basename(dest_path),
        total=total, unit='B', unit_scale=True, unit_divisor=1024,
    ) as bar:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))


def sciencebase_get_files(doi):
    """
    Query ScienceBase REST API to get file download links for a DOI.
    Returns list of (filename, url) tuples for .tif/.zip files.
    """
    # Resolve DOI to ScienceBase item ID
    resp = requests.get(
        SCIENCEBASE_API,
        params={'doi': doi, 'format': 'json'},
        timeout=30
    )
    resp.raise_for_status()
    data = resp.json()
    items = data.get('items', [])
    if not items:
        print(f"  [warn] No ScienceBase item found for DOI {doi}")
        return []

    item_id = items[0]['id']
    print(f"  ScienceBase item ID: {item_id}")

    # Get file list
    files_resp = requests.get(
        f"{SCIENCEBASE_API}/{item_id}",
        params={'format': 'json'},
        timeout=30
    )
    files_resp.raise_for_status()
    item_data = files_resp.json()

    results = []
    for f in item_data.get('files', []):
        name = f.get('name', '')
        url = f.get('downloadUri', f.get('url', ''))
        # Only grab bathymetry GeoTIFFs and zips (skip backscatter, habitat, etc.)
        if url and any(name.lower().endswith(ext) for ext in ['.zip', '.tif']):
            if any(kw in name.lower() for kw in ['bathy', 'bath', 'depth', 'dem']):
                results.append((name, url))
    return results


def geotiff_warp_to_utm32m(src_tif, out_tif, target_res_m=32.0):
    """
    Warp a geographic (WGS84) GeoTIFF to UTM Zone 10N at target_res_m resolution.
    Used to reproject the CRM GeoTIFF downloaded from ERDDAP.
    """
    from rasterio.warp import calculate_default_transform, reproject, Resampling as WarpResampling
    from rasterio.crs import CRS

    dst_crs = CRS.from_epsg(32610)  # UTM Zone 10N

    print(f"  Warping to UTM Zone 10N @ {target_res_m}m ...")
    with rasterio.open(src_tif) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds,
            resolution=target_res_m
        )
        kwargs = src.meta.copy()
        kwargs.update({'crs': dst_crs, 'transform': transform,
                       'width': width, 'height': height,
                       'nodata': np.nan, 'dtype': 'float32'})
        with rasterio.open(out_tif, 'w', **kwargs) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=WarpResampling.bilinear
            )
    print(f"  Saved: {out_tif}")


# ---------------------------------------------------------------------------
# Download tasks
# ---------------------------------------------------------------------------

def download_crm(out_dir):
    """Download NOAA CRM for central California and save as 32m UTM GeoTIFF."""
    print("\n=== NOAA Coastal Relief Model (central California) ===")
    crm_dir = os.path.join(out_dir, 'crm')
    raw_tif  = os.path.join(crm_dir, 'crm_central_ca_geo.tif')
    tif_path = os.path.join(crm_dir, 'crm_central_ca_32m_utm.tif')
    os.makedirs(crm_dir, exist_ok=True)

    if os.path.exists(tif_path):
        print(f"  Already exists: {tif_path}")
        return tif_path

    url = CRM_ERDDAP_URL.format(**CRM_BBOX)
    print(f"  Downloading CRM subset from NCEI ERDDAP ...")
    print(f"  URL: {url[:80]}...")

    try:
        download_file(url, raw_tif, desc='CRM GeoTIFF')
        geotiff_warp_to_utm32m(raw_tif, tif_path, target_res_m=32.0)
        os.remove(raw_tif)
        return tif_path
    except Exception as e:
        print(f"\n  [ERROR] CRM download failed: {e}")
        print(f"\n  Manual download instructions:")
        print(f"  1. Go to https://www.ncei.noaa.gov/maps/bathymetry/")
        print(f"  2. Navigate to central California (34-37.5N, 120-122.5W)")
        print(f"  3. Select 'Coastal Relief Model' layer and use the download tool")
        print(f"  4. Save as GeoTIFF and run with --lr_dir pointing to the output")
        return None


def download_ds781(out_dir):
    """Download USGS DS781 2m HR bathymetry via ScienceBase API."""
    print("\n=== USGS DS781 High-Resolution Bathymetry ===")
    hr_dir = os.path.join(out_dir, 'ds781_hr')
    os.makedirs(hr_dir, exist_ok=True)

    downloaded = []
    for block_name, doi in DS781_ITEMS.items():
        print(f"\n  Block: {block_name} (DOI: {doi})")
        try:
            file_links = sciencebase_get_files(doi)
            if not file_links:
                print(f"  [warn] No bathymetry files found via API for {block_name}")
                print(f"  Manual download: https://doi.org/{doi}")
                continue

            for fname, url in file_links:
                dest = os.path.join(hr_dir, block_name, fname)
                if os.path.exists(dest):
                    print(f"  Already exists: {fname}")
                    downloaded.append(dest)
                    continue
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                print(f"  Downloading: {fname}")
                download_file(url, dest, desc=fname)
                downloaded.append(dest)

        except Exception as e:
            print(f"  [ERROR] Failed for {block_name}: {e}")
            print(f"  Manual download: https://doi.org/{doi}")

    return downloaded


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Download supplementary bathymetry data')
    parser.add_argument('--out_dir', required=True,
                        help='Output directory for downloaded data')
    parser.add_argument('--crm', action='store_true',
                        help='Download NOAA Coastal Relief Model (LR data)')
    parser.add_argument('--ds781', action='store_true',
                        help='Download USGS DS781 HR bathymetry surveys')
    args = parser.parse_args()

    if not args.crm and not args.ds781:
        print("Specify at least one of --crm or --ds781")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    crm_tif = None
    if args.crm:
        crm_tif = download_crm(args.out_dir)

    if args.ds781:
        hr_files = download_ds781(args.out_dir)
        print(f"\nDownloaded {len(hr_files)} DS781 files to {args.out_dir}/ds781_hr/")

    print("\n=== Next steps ===")
    if crm_tif:
        print(f"CRM LR grid: {crm_tif}")
    print(f"Run preprocessing with multiple LR grids:")
    print(f"  python preprocessing/extract_paired_patches.py \\")
    print(f"    --lr_dir {args.out_dir}/crm/ \\")
    print(f"    --lr_dir_extra data_for_model/ \\")
    print(f"    --hr_dir {args.out_dir}/ds781_hr/ \\")
    print(f"    --out_dir /scratch/users/$USER/bathy_patches_v2/ \\")
    print(f"    --patch_size 64 --stride 32")


if __name__ == '__main__':
    main()
