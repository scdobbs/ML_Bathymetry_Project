#!/bin/bash
#SBATCH --job-name=bathy_train_resh
#SBATCH -p serc
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPU_SKU:H100_SXM5
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_residual_huber_%j.out
#SBATCH --error=logs/train_residual_huber_%j.err

# ============================================================
# Residual-prediction experiment v2: Huber-loss velocity + consistency.
#
# Previous residual_s128 run achieved val_loss=0.04 at epoch 10 (better
# than any v3 epoch) but collapsed catastrophically at epoch 20 — outlier
# patches with extreme HR-bicubic residuals eventually overwhelmed
# grad_clip and pushed the model into NaN.
#
# Fix: Huber loss on velocity (smooth_l1 beta=1) — quadratic for
# |error| < 1 (same gradient as MSE in healthy regime), linear above.
# Same fix that stabilized the consistency loss in v3.
#
# Also: consistency_loss_weight=1.0 reinforces the LR-respecting
# property the architecture already biases toward. Realizations from
# the previous run respected LR well but produced structureless HF
# noise; if this run also produces patchy results at epoch 125, the
# next step is spectral_loss_weight > 0.
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
TRAIN_TAG="${TRAIN_TAG:-residual_huber_s128}"
CONS_WEIGHT="${CONS_WEIGHT:-1.0}"
PATCHES_DIR="/scratch/groups/hilley/bathy_diffusion_2026/patches_512_s128"
TRAIN_OUT="/scratch/users/${USER}/bathy_diffusion/training_${TRAIN_TAG}"

cd "$PROJECT_DIR"

echo "======================================================================"
echo "Bathymetry SR training — residual + Huber + consistency"
echo "  Host         : $(hostname)"
echo "  TRAIN_TAG    : $TRAIN_TAG"
echo "  PATCHES_DIR  : $PATCHES_DIR"
echo "  TRAIN_OUT    : $TRAIN_OUT"
echo "  base_ch      : 32"
echo "  n_epochs     : 125 (~350K steps)"
echo "  CONS_WEIGHT  : $CONS_WEIGHT"
echo "  Mode         : residual (HR = bicubic + model_output)"
echo "  FM loss      : huber (smooth_l1 beta=1)"
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
    --residual_mode \
    --fm_loss_type huber \
    --consistency_loss_weight "$CONS_WEIGHT" \
    --compile

echo "Training script returned: $(date)"
