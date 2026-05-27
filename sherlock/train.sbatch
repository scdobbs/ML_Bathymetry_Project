#!/bin/bash
#SBATCH --job-name=bathy_train
#SBATCH -p serc
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPU_SKU:H100_SXM5
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# ============================================================
# Bathymetry SR — rectified-flow training (production)
#
# Defaults to a single H100 on serc.  Override the GPU via:
#   sbatch --constraint=GPU_SKU:A100_SXM4 sherlock/train.sbatch
#
# Use TRAIN_TAG to namespace checkpoint dirs so concurrent A100/H100
# runs don't collide:
#   sbatch --constraint=GPU_SKU:A100_SXM4 --export=ALL,TRAIN_TAG=a100 sherlock/train.sbatch
#   sbatch --constraint=GPU_SKU:H100_SXM5 --export=ALL,TRAIN_TAG=h100 sherlock/train.sbatch
#
# Training auto-resumes from the latest checkpoint in TRAIN_OUT/checkpoints/
# if one exists, so jobs killed at the wall-time limit can simply be
# resubmitted to continue.
#
# Launches via `python3 -m torch.distributed.run` (not `torchrun`) so DDP
# workers run under the venv interpreter that has triton + rasterio.
# ============================================================

mkdir -p logs

# Conda's Python on $PATH can shadow the module-provided Python 3.12.
type conda >/dev/null 2>&1 && {
    conda deactivate >/dev/null 2>&1 || true
    conda deactivate >/dev/null 2>&1 || true
}
unset PYTHONPATH

module load python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312

# Activate the venv populated by sherlock/setup.sh (has rasterio with
# bundled libgdal, triton, einops, scipy, the editable project install).
source "${HOME}/bathy_diffusion_env/bin/activate"

PROJECT_DIR="/home/users/${USER}/GAN_Bathymetry_Project"
TRAIN_TAG="${TRAIN_TAG:-default}"
PATCHES_DIR="/scratch/groups/hilley/bathy_diffusion_2026/patches_512"
TRAIN_OUT="/scratch/users/${USER}/bathy_diffusion/training_${TRAIN_TAG}"

cd "$PROJECT_DIR"

echo "======================================================================"
echo "Bathymetry SR training"
echo "  Host         : $(hostname)"
echo "  TRAIN_TAG    : $TRAIN_TAG"
echo "  PATCHES_DIR  : $PATCHES_DIR"
echo "  TRAIN_OUT    : $TRAIN_OUT"
echo "  GPUs on node : ${SLURM_GPUS_ON_NODE:-1}"
echo "  GPU model    : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
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
    --n_epochs    500 \
    --precision   bf16 \
    --compile

echo "Training script returned: $(date)"
