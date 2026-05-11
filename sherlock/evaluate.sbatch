#!/bin/bash
#SBATCH --job-name=bathy_eval
#SBATCH --partition=serc
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/users/scdobbs/bathy_diffusion/logs/evaluate_%j.out
#SBATCH --error=/scratch/users/scdobbs/bathy_diffusion/logs/evaluate_%j.err

set -euo pipefail

conda deactivate 2>/dev/null || true
unset PYTHONPATH

module load python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312
source ~/bathy_diffusion_env/bin/activate

PROJ_DIR="/home/users/scdobbs/GAN_Bathymetry_Project"
SCRATCH_DIR="/scratch/users/scdobbs/bathy_diffusion"
CHECKPOINT="${SCRATCH_DIR}/training/checkpoints/ckpt_step01135000.pt"
TEST_PATCHES="${SCRATCH_DIR}/patches/test"
OUT_DIR="${PROJ_DIR}/results/evaluation"

mkdir -p "$OUT_DIR"
mkdir -p "${SCRATCH_DIR}/logs"

cd "$PROJ_DIR"

echo "Starting evaluation at $(date)"
echo "Checkpoint: $CHECKPOINT"
echo "Test patches: $TEST_PATCHES"

python3 evaluation/evaluate.py \
    --checkpoint    "$CHECKPOINT" \
    --test_patches  "$TEST_PATCHES" \
    --out_dir       "$OUT_DIR" \
    --n_realizations 50 \
    --ddim_steps     50 \
    --base_ch        64 \
    --max_patches    1000

echo "Evaluation complete at $(date)"
echo "Results saved to: $OUT_DIR"
