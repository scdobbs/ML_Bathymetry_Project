#!/bin/bash
#SBATCH --job-name=setup_env
#SBATCH -p dev
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=logs/setup_env_%j.out
#SBATCH --error=logs/setup_env_%j.err

# ============================================================
# Create the bathy_diffusion conda environment via mamba.
# Run once to set up dependencies.
# ============================================================

set -e

PROJECT_DIR="/home/users/${USER}/GAN_Bathymetry_Project"
MAMBA="/home/users/${USER}/miniforge3/bin/mamba"
ENV_PREFIX="/home/users/${USER}/.conda/envs/bathy_diffusion"

echo "=== Environment setup starting on $(hostname) at $(date) ==="
echo "Using mamba: $($MAMBA --version)"

# Remove any partial previous install
rm -rf "$ENV_PREFIX"

# Create the environment
$MAMBA env create \
    -f "${PROJECT_DIR}/sherlock/environment.yml" \
    --prefix "$ENV_PREFIX" \
    -y

echo "=== Environment created at $ENV_PREFIX ==="

# Activate and verify
source /home/users/${USER}/miniforge3/etc/profile.d/conda.sh
source /home/users/${USER}/miniforge3/etc/profile.d/mamba.sh
mamba activate "$ENV_PREFIX"

echo "=== Verifying imports ==="
python -c "
import torch; print(f'torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')
import torchvision; print(f'torchvision: {torchvision.__version__}')
import rasterio; print(f'rasterio: {rasterio.__version__}')
import pyproj; print(f'pyproj: {pyproj.__version__}')
import numpy; print(f'numpy: {numpy.__version__}')
import scipy; print(f'scipy: {scipy.__version__}')
import matplotlib; print(f'matplotlib: {matplotlib.__version__}')
import skimage; print(f'scikit-image: {skimage.__version__}')
import tensorboard; print(f'tensorboard: {tensorboard.__version__}')
print('All imports successful!')
"

echo "=== Setup complete at $(date) ==="
