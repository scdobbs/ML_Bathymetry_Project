#!/bin/bash
#SBATCH --job-name=bathy_v3
#SBATCH --partition=serc
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/users/scdobbs/bathy_diffusion_v3/logs/train_%j.out
#SBATCH --error=/scratch/users/scdobbs/bathy_diffusion_v3/logs/train_%j.err

set -euo pipefail

conda deactivate 2>/dev/null || true
unset PYTHONPATH

module load python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312
source ~/bathy_diffusion_env/bin/activate

PROJ_DIR="/home/users/scdobbs/GAN_Bathymetry_Project"

# Combined patches: original Morro Bay AUV surveys + all new CA coast surveys
PATCHES_ORIG="/scratch/users/scdobbs/bathy_diffusion/patches"
PATCHES_NEW="/scratch/users/scdobbs/bathy_diffusion/patches_v3"

mkdir -p /scratch/users/scdobbs/bathy_diffusion_v3/logs

cd "$PROJ_DIR"

echo "=== v3 training: full CA coast dataset, train from scratch ==="
echo "  Train patches: ${PATCHES_ORIG}/train + ${PATCHES_NEW}/train (~1.04M patches)"
echo "  Val patches:   ${PATCHES_NEW}/val (38k patches)"
echo "Started: $(date)"

python3 training/train.py \
    --patches_dir "$PATCHES_ORIG" "$PATCHES_NEW" \
    --out_dir     /scratch/users/scdobbs/bathy_diffusion_v3 \
    --batch_size  64 \
    --n_epochs    300 \
    --lr          1e-4 \
    --base_ch     64

echo "Training complete: $(date)"
