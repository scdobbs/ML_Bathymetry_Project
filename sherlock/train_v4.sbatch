#!/bin/bash
#SBATCH --job-name=bathy_v4
#SBATCH --partition=serc
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/users/scdobbs/bathy_diffusion_v4/logs/train_%j.out
#SBATCH --error=/scratch/users/scdobbs/bathy_diffusion_v4/logs/train_%j.err

set -euo pipefail

conda deactivate 2>/dev/null || true
unset PYTHONPATH

module load python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312
source ~/bathy_diffusion_env/bin/activate

PROJ_DIR="/home/users/scdobbs/GAN_Bathymetry_Project"

PATCHES_ORIG="/scratch/users/scdobbs/bathy_diffusion/patches"
PATCHES_NEW="/scratch/users/scdobbs/bathy_diffusion/patches_v3"

mkdir -p /scratch/users/scdobbs/bathy_diffusion_v4/logs

cd "$PROJ_DIR"

echo "=== v4 training: residual learning (predict HR - LR_bicubic) ==="
echo "  Train patches: ${PATCHES_ORIG}/train + ${PATCHES_NEW}/train"
echo "  Val patches:   ${PATCHES_NEW}/val"
echo "Started: $(date)"

python3 training/train.py \
    --patches_dir "$PATCHES_ORIG" "$PATCHES_NEW" \
    --out_dir     /scratch/users/scdobbs/bathy_diffusion_v4 \
    --batch_size  64 \
    --n_epochs    300 \
    --lr          1e-4 \
    --base_ch     64

echo "Training complete: $(date)"
