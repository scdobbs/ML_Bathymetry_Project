#!/bin/bash
#SBATCH --job-name=bathy_preproc_v3
#SBATCH --partition=serc
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/users/scdobbs/GAN_Bathymetry_Project/logs/preproc_v3_%j.out
#SBATCH --error=/home/users/scdobbs/GAN_Bathymetry_Project/logs/preproc_v3_%j.err

set -euo pipefail

module load python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312
source ~/bathy_diffusion_env/bin/activate

PROJ_DIR="/home/users/scdobbs/GAN_Bathymetry_Project"
NEW_DATA="/scratch/users/scdobbs/more_data_2026"
ORIG_HR="${PROJ_DIR}/data_for_model"
OUT_DIR="/scratch/users/scdobbs/bathy_diffusion/patches_v3"

mkdir -p "$OUT_DIR"
mkdir -p "${PROJ_DIR}/logs"

cd "$PROJ_DIR"

echo "=== Preprocessing v3: all data combined ==="
echo "Started: $(date)"
echo ""
echo "LR grids:"
echo "  1. ${ORIG_HR}/MorroBay_all_2020_interp_32m_converted.tif  (Morro Bay)"
echo "  2. ${NEW_DATA}/low_res/norcal-to-triple-junction-32m.tif  (NorCal)"
echo "  3. ${NEW_DATA}/low_res/Santa_Barbara_32m-clipped2.tif     (Santa Barbara)"
echo ""
echo "HR directories:"
echo "  1. ${ORIG_HR}  (original ~18 surveys)"
echo "  2. ${NEW_DATA}/high-res  (29 new surveys)"
echo ""

python3 preprocessing/extract_paired_patches.py \
    --lr_dir /scratch/users/scdobbs/lr_grids_v3 \
    --hr_dir \
        "${ORIG_HR}" \
        "${NEW_DATA}/high-res" \
    --out_dir "$OUT_DIR" \
    --patch_size 64 \
    --stride 32 \
    --nodata_threshold 0.1 \
    --flat_threshold 0.05

echo ""
echo "=== Done: $(date) ==="
echo ""
echo "Patch counts:"
for split in train val test; do
    count=$(find "$OUT_DIR/$split" -name "*.npz" 2>/dev/null | wc -l)
    echo "  $split: $count"
done
echo "  total: $(find "$OUT_DIR" -name "*.npz" 2>/dev/null | wc -l)"
