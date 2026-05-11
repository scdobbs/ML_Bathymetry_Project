#!/bin/bash
#SBATCH --job-name=bathy_val_post
#SBATCH --partition=serc
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/users/scdobbs/bathy_diffusion/logs/val_postprocess_%j.out
#SBATCH --error=/scratch/users/scdobbs/bathy_diffusion/logs/val_postprocess_%j.err

set -euo pipefail

conda deactivate 2>/dev/null || true
unset PYTHONPATH

module load python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312
source ~/bathy_diffusion_env/bin/activate

PROJ_DIR="/home/users/scdobbs/GAN_Bathymetry_Project"
SCRATCH_DIR="/scratch/users/scdobbs/bathy_diffusion"
FIG_DIR="${PROJ_DIR}/results/figures"

mkdir -p "$FIG_DIR"
mkdir -p "${SCRATCH_DIR}/logs"

cd "$PROJ_DIR"

echo "=== Step 1: Uncertainty maps for 4 completed val regions ==="
for REGION in 600mGully BankFlankHoles PockmarkNorth LuciaChica2019_970; do
    REGION_DIR="${SCRATCH_DIR}/val_realizations/${REGION}"
    if [ -d "$REGION_DIR" ]; then
        echo "  Computing uncertainty maps: $REGION"
        python3 inference/uncertainty_maps.py \
            --realizations_dir "$REGION_DIR" \
            --out_dir          "${SCRATCH_DIR}/val_uncertainty_maps/${REGION}" \
            --priority_fraction 0.10
    else
        echo "  [skip] No realizations found for $REGION"
    fi
done

echo ""
echo "=== Step 2: Transect plots for test regions ==="
python3 notebooks/transect_plot.py \
    --realizations_dir "${SCRATCH_DIR}/realizations" \
    --uncertainty_dir  "${SCRATCH_DIR}/uncertainty_maps" \
    --lr_tif           "data_for_model/MorroBay_all_2020_interp_32m_converted.tif" \
    --hr_dir           "data_for_model/" \
    --out_dir          "$FIG_DIR"

echo ""
echo "=== Step 3: Comparison/realization/uncertainty figures for val regions ==="
python3 notebooks/visualize_results.py \
    --realizations_dir "${SCRATCH_DIR}/val_realizations" \
    --uncertainty_dir  "${SCRATCH_DIR}/val_uncertainty_maps" \
    --lr_tif           "data_for_model/MorroBay_all_2020_interp_32m_converted.tif" \
    --hr_dir           "data_for_model/" \
    --out_dir          "${FIG_DIR}/val" \
    --regions          600mGully BankFlankHoles PockmarkNorth LuciaChica2019_970

echo ""
echo "All done at $(date)"
echo "Figures: $FIG_DIR"
