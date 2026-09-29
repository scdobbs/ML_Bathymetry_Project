#!/bin/bash
#SBATCH --job-name=r03_prep
#SBATCH -p serc
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/users/scdobbs/grl_review/R03/logs/%x_%j.out
#SBATCH --error=/scratch/users/scdobbs/grl_review/R03/logs/%x_%j.err
type conda >/dev/null 2>&1 && { conda deactivate >/dev/null 2>&1 || true; conda deactivate >/dev/null 2>&1 || true; }
unset PYTHONPATH; module load python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312; source "${HOME}/bathy_diffusion_env/bin/activate"
cd /home/users/scdobbs/GAN_Bathymetry_Project; echo "=== R03 expanded clean preprocess $(date) $(hostname) code=$(git rev-parse --short HEAD) ==="
python /scratch/users/scdobbs/grl_review/R03/run_extract_expanded.py
echo "=== done $(date) ==="
