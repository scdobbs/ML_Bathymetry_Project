#!/bin/bash
#SBATCH --job-name=bathy_train_small
#SBATCH -p serc
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPU_SKU:H100_SXM5
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_small_%j.out
#SBATCH --error=logs/train_small_%j.err

# ============================================================
# Smaller-model training experiment.
#
# Hypothesis: 85M params on ~11K (~1-2K independent) patches is too
# over-parameterized — the model learns local texture but not global
# morphology, producing jigsaw artifacts.  Try base_ch=32 (~22M params)
# on the denser stride-128 dataset (~45K patches) to better match
# capacity to effective data.
#
# Total optimizer steps target ≈ 350K (matches H100 baseline run):
#   45K patches / batch 16 = 2,800 steps/epoch
#   n_epochs = 125 → 350K steps
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
TRAIN_TAG="${TRAIN_TAG:-small_s128}"
PATCHES_DIR="/scratch/groups/hilley/bathy_diffusion_2026/patches_512_s128"
TRAIN_OUT="/scratch/users/${USER}/bathy_diffusion/training_${TRAIN_TAG}"

cd "$PROJECT_DIR"

echo "======================================================================"
echo "Bathymetry SR training — smaller model on denser dataset"
echo "  Host         : $(hostname)"
echo "  TRAIN_TAG    : $TRAIN_TAG"
echo "  PATCHES_DIR  : $PATCHES_DIR"
echo "  TRAIN_OUT    : $TRAIN_OUT"
echo "  base_ch      : 32 (vs 64 baseline)"
echo "  n_epochs     : 125 (~350K steps)"
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
    --compile

echo "Training script returned: $(date)"
