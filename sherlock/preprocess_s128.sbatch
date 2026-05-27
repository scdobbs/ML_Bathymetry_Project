#!/bin/bash
#SBATCH --job-name=preprocess_s128
#SBATCH -p serc
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/preprocess_s128_%j.out
#SBATCH --error=logs/preprocess_s128_%j.err

# ============================================================
# Stride-128 preprocessing — ~4× more patches than stride-256.
#
# Re-uses the existing shared HR/LR staging dir, writes a fresh
# patches_512_s128/ alongside the existing patches_512/ so we
# can compare model behavior at the two patch densities.
#
# Runs against all three LR grids sequentially with first-run-wins
# (skip-if-exists) so the Morro Bay LR provides authoritative
# conditioning for Morro Bay-area surveys.
# ============================================================

mkdir -p logs

type conda >/dev/null 2>&1 && {
    conda deactivate >/dev/null 2>&1 || true
    conda deactivate >/dev/null 2>&1 || true
}
unset PYTHONPATH

module load python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312
source "${HOME}/bathy_diffusion_env/bin/activate"

PROJECT_DIR="/home/users/${USER}/GAN_Bathymetry_Project"
SHARED_DIR="/scratch/groups/hilley/bathy_diffusion_2026"

HR_DIR="${SHARED_DIR}/hr_tifs"
LR_DIR="${SHARED_DIR}/lr_tifs"
OUT_DIR="${SHARED_DIR}/patches_512_s128"

LR_MORROBAY="${LR_DIR}/MorroBay_all_2020_interp_32m_converted.tif"
LR_NORCAL="${LR_DIR}/norcal-to-triple-junction-32m.tif"
LR_SB="${LR_DIR}/Santa_Barbara_32m-clipped2.tif"

cd "$PROJECT_DIR"

echo "=== Stride-128 preprocessing (multi-LR, first-run-wins) ==="
echo "  HR dir:  $HR_DIR  ($(ls $HR_DIR/*.tif | wc -l) tifs)"
echo "  Out dir: $OUT_DIR"
echo "  Patch size: 512x512  Stride: 128  SR ratio: 16"
echo "  Started: $(date)"
echo

# Run 1: Morro Bay LR
echo "--- Run 1/3: Morro Bay LR ---"
python3 -m preprocessing.extract_paired_patches \
    --lr_path  "$LR_MORROBAY" \
    --hr_dir   "$HR_DIR" \
    --out_dir  "$OUT_DIR" \
    --patch_size 512 --stride 128 --sr_ratio 16 \
    --nodata_threshold 0.10 --flat_threshold 0.05

# Run 2: NorCal LR
echo
echo "--- Run 2/3: NorCal LR ---"
python3 -m preprocessing.extract_paired_patches \
    --lr_path  "$LR_NORCAL" \
    --hr_dir   "$HR_DIR" \
    --out_dir  "$OUT_DIR" \
    --patch_size 512 --stride 128 --sr_ratio 16 \
    --nodata_threshold 0.10 --flat_threshold 0.05

# Run 3: Santa Barbara LR
echo
echo "--- Run 3/3: Santa Barbara LR ---"
python3 -m preprocessing.extract_paired_patches \
    --lr_path  "$LR_SB" \
    --hr_dir   "$HR_DIR" \
    --out_dir  "$OUT_DIR" \
    --patch_size 512 --stride 128 --sr_ratio 16 \
    --nodata_threshold 0.10 --flat_threshold 0.05

echo
echo "=== Preprocessing complete ==="
echo "  Finished: $(date)"
echo "  Train: $(ls $OUT_DIR/train 2>/dev/null | wc -l)  Val: $(ls $OUT_DIR/val 2>/dev/null | wc -l)  Test: $(ls $OUT_DIR/test 2>/dev/null | wc -l)"
