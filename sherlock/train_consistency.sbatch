#!/bin/bash
#SBATCH --job-name=bathy_train_cons
#SBATCH -p serc
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPU_SKU:H100_SXM5
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_consistency_%j.out
#SBATCH --error=logs/train_consistency_%j.err

# ============================================================
# Data-consistency loss experiment.
#
# Hypothesis: jigsaw artifacts and "high topography in the middle of a
# low-elevation area" come from a weak LR conditioning pathway. The
# only signal pulling x0_pred toward the LR is the deep-level
# cross-attention, which is too indirect to anchor absolute elevation.
#
# Fix: add an auxiliary loss that coarsens x0_pred back to the LR grid
# (avg_pool by sr_ratio) and matches it against lr_raw. SNR-weighted so
# it dominates at low t (where x0_pred is reliable) and CFG-keep-masked
# so it skips samples whose lr_raw was replaced by the null token.
#
# Same model/data as Run 2 (base_ch=32, patches_512_s128, 125 epochs):
# only the loss changes.
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
TRAIN_TAG="${TRAIN_TAG:-consistency_s128}"
CONS_WEIGHT="${CONS_WEIGHT:-1.0}"
PATCHES_DIR="/scratch/groups/hilley/bathy_diffusion_2026/patches_512_s128"
TRAIN_OUT="/scratch/users/${USER}/bathy_diffusion/training_${TRAIN_TAG}"

cd "$PROJECT_DIR"

echo "======================================================================"
echo "Bathymetry SR training — data-consistency loss"
echo "  Host         : $(hostname)"
echo "  TRAIN_TAG    : $TRAIN_TAG"
echo "  PATCHES_DIR  : $PATCHES_DIR"
echo "  TRAIN_OUT    : $TRAIN_OUT"
echo "  base_ch      : 32"
echo "  n_epochs     : 125 (~350K steps)"
echo "  CONS_WEIGHT  : $CONS_WEIGHT"
echo "  GPU          : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "  Branch       : $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)"
echo "  Started      : $(date)"
echo "======================================================================"

python3 -m torch.distributed.run \
    --standalone \
    --nproc_per_node="${SLURM_GPUS_ON_NODE:-1}" \
    --module training.train \
    --patches_dir "$PATCHES_DIR" \
    --out_dir     "$TRAIN_OUT" \
    --batch_size  16 \
    --n_epochs    125 \
    --base_ch     32 \
    --precision   bf16 \
    --consistency_loss_weight "$CONS_WEIGHT" \
    --compile

echo "Training script returned: $(date)"
