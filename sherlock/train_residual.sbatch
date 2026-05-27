#!/bin/bash
#SBATCH --job-name=bathy_train_res
#SBATCH -p serc
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPU_SKU:H100_SXM5
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_residual_%j.out
#SBATCH --error=logs/train_residual_%j.err

# ============================================================
# Residual-prediction experiment.
#
# Hypothesis: cross-attention onto LR alone is too indirect to enforce
# LR-respecting predictions (still see jigsaw + wrong elevation after
# the data-consistency-loss run). Switch to an architectural prior:
# the model takes bicubic(LR) as a 2nd input channel and predicts the
# HR RESIDUAL on top of it. The bicubic baseline is added back at
# sample time. Useful output = high-frequency residual; LR-respecting
# baseline is automatic.
#
# Trade-off (observed on old DDPM branch): inter-realization variance
# tends to drop because the model can "give up" and output zero
# residual. We accept this for now to verify the LR-respect property.
#
# Same model size and dataset as v3 (base_ch=32, patches_512_s128).
# Consistency loss left ON at weight 1.0 since it now plays nicely
# with Huber + t-gate, and reinforces the same objective the
# architecture is now structurally biased toward.
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
TRAIN_TAG="${TRAIN_TAG:-residual_s128}"
CONS_WEIGHT="${CONS_WEIGHT:-0.0}"   # default OFF — isolate the residual-mode effect
PATCHES_DIR="/scratch/groups/hilley/bathy_diffusion_2026/patches_512_s128"
TRAIN_OUT="/scratch/users/${USER}/bathy_diffusion/training_${TRAIN_TAG}"

cd "$PROJECT_DIR"

echo "======================================================================"
echo "Bathymetry SR training — residual prediction + bicubic input"
echo "  Host         : $(hostname)"
echo "  TRAIN_TAG    : $TRAIN_TAG"
echo "  PATCHES_DIR  : $PATCHES_DIR"
echo "  TRAIN_OUT    : $TRAIN_OUT"
echo "  base_ch      : 32"
echo "  n_epochs     : 125 (~350K steps)"
echo "  CONS_WEIGHT  : $CONS_WEIGHT"
echo "  Mode         : residual (HR = bicubic + model_output)"
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
    --consistency_loss_weight "$CONS_WEIGHT" \
    --compile

echo "Training script returned: $(date)"
