#!/bin/bash
#SBATCH --job-name=bathy_bench
#SBATCH -p serc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=logs/benchmark_%j.out
#SBATCH --error=logs/benchmark_%j.err

set -e
mkdir -p logs

# Prevent conda/PYTHONPATH from shadowing modules
conda deactivate 2>/dev/null || true
unset PYTHONPATH

# Load modules and activate venv
module load python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312
source /home/users/${USER}/bathy_diffusion_env/bin/activate

PROJECT_DIR="/home/users/${USER}/GAN_Bathymetry_Project"
SCRATCH_DIR="/scratch/users/${USER}/bathy_diffusion"

cd "$PROJECT_DIR"

echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo ""
python3 training/benchmark.py \
    --patches_dir "${SCRATCH_DIR}/patches" \
    --batch_sizes 8,16,32,64,128 \
    --n_iters 20

echo "Benchmark complete."
