#!/bin/bash
#SBATCH --job-name=bathy_figs
#SBATCH --partition=serc
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/users/scdobbs/bathy_diffusion/logs/figures_%j.out
#SBATCH --error=/scratch/users/scdobbs/bathy_diffusion/logs/figures_%j.err

set -euo pipefail

conda deactivate 2>/dev/null || true
unset PYTHONPATH

module load python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312
source ~/bathy_diffusion_env/bin/activate

PROJ_DIR="/home/users/scdobbs/GAN_Bathymetry_Project"
SCRATCH_DIR="/scratch/users/scdobbs/bathy_diffusion"
FIG_DIR="${PROJ_DIR}/results/figures"

mkdir -p "${FIG_DIR}/val"
cd "$PROJ_DIR"

echo "=== Transect profiles for test regions ==="
python3 notebooks/transect_plot.py \
    --realizations_dir "${SCRATCH_DIR}/realizations" \
    --uncertainty_dir  "${SCRATCH_DIR}/uncertainty_maps" \
    --lr_tif           "data_for_model/MorroBay_all_2020_interp_32m_converted.tif" \
    --hr_dir           "data_for_model/" \
    --out_dir          "$FIG_DIR"

echo ""
echo "=== Comparison/uncertainty figures for 4 val regions ==="
python3 notebooks/visualize_results.py \
    --realizations_dir "${SCRATCH_DIR}/val_realizations" \
    --uncertainty_dir  "${SCRATCH_DIR}/val_uncertainty_maps" \
    --lr_tif           "data_for_model/MorroBay_all_2020_interp_32m_converted.tif" \
    --hr_dir           "data_for_model/" \
    --out_dir          "${FIG_DIR}/val" \
    --regions          600mGully BankFlankHoles PockmarkNorth LuciaChica2019_970

echo ""
echo "Done at $(date)"
ls -lh "$FIG_DIR"/transect_*.png 2>/dev/null
ls -lh "${FIG_DIR}/val/"*.png 2>/dev/null
