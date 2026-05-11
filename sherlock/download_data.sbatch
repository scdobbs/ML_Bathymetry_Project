#!/bin/bash
#SBATCH --job-name=bathy_download
#SBATCH --partition=serc
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=/home/users/scdobbs/GAN_Bathymetry_Project/logs/download_%j.out
#SBATCH --error=/home/users/scdobbs/GAN_Bathymetry_Project/logs/download_%j.err

set -euo pipefail

conda deactivate 2>/dev/null || true
unset PYTHONPATH

module load python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312
source ~/bathy_diffusion_env/bin/activate

PROJ_DIR="/home/users/scdobbs/GAN_Bathymetry_Project"
SUPP_DIR="${PROJ_DIR}/data_for_model/supplementary"

mkdir -p "$SUPP_DIR"
mkdir -p "${PROJ_DIR}/logs"

cd "$PROJ_DIR"

echo "=== Downloading supplementary bathymetry data ==="
echo "Started: $(date)"

# Download CRM (LR) + DS781 (HR)
python3 preprocessing/download_data.py \
    --out_dir "$SUPP_DIR" \
    --crm \
    --ds781

echo "Done: $(date)"
