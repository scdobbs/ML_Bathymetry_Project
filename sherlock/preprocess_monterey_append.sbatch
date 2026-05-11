#!/bin/bash
#SBATCH --job-name=bathy_preproc_mc
#SBATCH --partition=serc
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/users/scdobbs/GAN_Bathymetry_Project/logs/preproc_mc_%j.out
#SBATCH --error=/home/users/scdobbs/GAN_Bathymetry_Project/logs/preproc_mc_%j.err
# Run after both preprocess_v3 and convert_monterey_grd complete

set -euo pipefail

module load python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312
source ~/bathy_diffusion_env/bin/activate

PROJ_DIR="/home/users/scdobbs/GAN_Bathymetry_Project"
NEW_DATA="/scratch/users/scdobbs/more_data_2026"
ORIG_HR="${PROJ_DIR}/data_for_model"
OUT_DIR="/scratch/users/scdobbs/bathy_diffusion/patches_v3"

cd "$PROJ_DIR"

echo "=== Appending Monterey Canyon patches to patches_v3 ==="
echo "Started: $(date)"
echo ""
echo "HR directory: ${NEW_DATA}/high-res  (only MC files will be new)"

# Count existing patches before
before=$(find "$OUT_DIR" -name "*.npz" 2>/dev/null | wc -l)
echo "Patches before: $before"
echo ""

python3 preprocessing/extract_paired_patches.py \
    --lr_dir \
        "${ORIG_HR}" \
        "${NEW_DATA}/low_res" \
    --hr_dir \
        "${NEW_DATA}/high-res" \
    --out_dir "$OUT_DIR" \
    --patch_size 64 \
    --stride 32 \
    --nodata_threshold 0.1 \
    --flat_threshold 0.05

# Note: existing .npz files are not overwritten (unique filenames per survey+position)
# Only the new Bathymetry_2m_MC_* files will produce new patches

echo ""
echo "=== Done: $(date) ==="
after=$(find "$OUT_DIR" -name "*.npz" 2>/dev/null | wc -l)
new=$((after - before))
echo "Patches after:  $after  (+$new new from Monterey Canyon)"
for split in train val test; do
    count=$(find "$OUT_DIR/$split" -name "*.npz" 2>/dev/null | wc -l)
    echo "  $split: $count"
done
