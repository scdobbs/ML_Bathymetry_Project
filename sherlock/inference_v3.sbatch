#!/bin/bash
#SBATCH --job-name=bathy_inf_v3
#SBATCH --partition=serc
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/users/scdobbs/bathy_diffusion_v3/logs/inference_%j.out
#SBATCH --error=/scratch/users/scdobbs/bathy_diffusion_v3/logs/inference_%j.err

set -euo pipefail

conda deactivate 2>/dev/null || true
unset PYTHONPATH

module load python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312
source ~/bathy_diffusion_env/bin/activate

PROJ_DIR="/home/users/scdobbs/GAN_Bathymetry_Project"
SCRATCH_DIR="/scratch/users/scdobbs/bathy_diffusion_v3"
CHECKPOINT="${SCRATCH_DIR}/checkpoints/ckpt_final.pt"
FIG_DIR="${PROJ_DIR}/results/figures_v3"

mkdir -p "${SCRATCH_DIR}/logs"
mkdir -p "$FIG_DIR"

cd "$PROJ_DIR"

echo "=== v3 Inference: 300-epoch model trained on full CA coast dataset ==="
echo "Checkpoint: $CHECKPOINT"
echo "Started: $(date)"

echo ""
echo "=== Step 1: Generate SR realizations on held-out test surveys ==="
python3 inference/generate_realizations.py \
    --checkpoint     "$CHECKPOINT" \
    --lr_tif         "data_for_model/MorroBay_all_2020_interp_32m_converted.tif" \
    --out_dir        "${SCRATCH_DIR}/realizations" \
    --test_surveys \
    --n_realizations 50 \
    --ddim_steps     50 \
    --patch_size     64 \
    --stride         32 \
    --base_ch        64

echo ""
echo "=== Step 2: Uncertainty maps ==="
for REGION_DIR in "${SCRATCH_DIR}/realizations"/*/; do
    REGION_NAME=$(basename "$REGION_DIR")
    echo "  Processing: $REGION_NAME"
    python3 inference/uncertainty_maps.py \
        --realizations_dir "$REGION_DIR" \
        --out_dir          "${SCRATCH_DIR}/uncertainty_maps/${REGION_NAME}" \
        --priority_fraction 0.10
done

echo ""
echo "=== Step 3: LR / SR / HR comparison panels ==="
python3 notebooks/visualize_results.py \
    --realizations_dir "${SCRATCH_DIR}/realizations" \
    --uncertainty_dir  "${SCRATCH_DIR}/uncertainty_maps" \
    --lr_tif           "data_for_model/MorroBay_all_2020_interp_32m_converted.tif" \
    --hr_dir           "data_for_model/" \
    --out_dir          "$FIG_DIR" \
    --regions          HeadlessCanyon LuciaChica_2007 BankTopEastofCanyon3

echo ""
echo "=== Step 4: Transect profile plots ==="
python3 notebooks/transect_plot.py \
    --realizations_dir "${SCRATCH_DIR}/realizations" \
    --uncertainty_dir  "${SCRATCH_DIR}/uncertainty_maps" \
    --lr_tif           "data_for_model/MorroBay_all_2020_interp_32m_converted.tif" \
    --hr_dir           "data_for_model/" \
    --out_dir          "$FIG_DIR"

echo ""
echo "=== All done: $(date) ==="
ls -lh "$FIG_DIR"
