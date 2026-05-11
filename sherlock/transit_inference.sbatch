#!/bin/bash
#SBATCH --job-name=bathy_transit
#SBATCH --partition=serc
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/users/scdobbs/bathy_diffusion/logs/transit_%j.out
#SBATCH --error=/scratch/users/scdobbs/bathy_diffusion/logs/transit_%j.err

set -euo pipefail

conda deactivate 2>/dev/null || true
unset PYTHONPATH

module load python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312
source ~/bathy_diffusion_env/bin/activate

PROJ_DIR="/home/users/scdobbs/GAN_Bathymetry_Project"
SCRATCH_DIR="/scratch/users/scdobbs/bathy_diffusion"
CHECKPOINT="${SCRATCH_DIR}/training/checkpoints/ckpt_step01135000.pt"

mkdir -p "${SCRATCH_DIR}/logs"

cd "$PROJ_DIR"

echo "Starting Transit inference at $(date)"

python3 inference/generate_realizations.py \
    --checkpoint    "$CHECKPOINT" \
    --lr_tif        "data_for_model/MorroBay_all_2020_interp_32m_converted.tif" \
    --out_dir       "${SCRATCH_DIR}/val_realizations" \
    --bbox          624075 3913994 650503 3937197 \
    --region_name   Transit \
    --n_realizations 50 \
    --ddim_steps     50 \
    --patch_size     64 \
    --stride         32 \
    --base_ch        64

echo "Inference done. Computing uncertainty maps..."

python3 inference/uncertainty_maps.py \
    --realizations_dir "${SCRATCH_DIR}/val_realizations/Transit" \
    --out_dir          "${SCRATCH_DIR}/val_uncertainty_maps/Transit" \
    --priority_fraction 0.10

echo "Transit complete at $(date)"
