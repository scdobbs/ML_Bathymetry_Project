#!/bin/bash
#SBATCH --job-name=r01b
#SBATCH -p serc
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/users/scdobbs/grl_review/R01b/logs/%x_%j.out
#SBATCH --error=/scratch/users/scdobbs/grl_review/R01b/logs/%x_%j.err
type conda >/dev/null 2>&1 && { conda deactivate >/dev/null 2>&1 || true; conda deactivate >/dev/null 2>&1 || true; }
unset PYTHONPATH
module load python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312
source "${HOME}/bathy_diffusion_env/bin/activate"
cd /home/users/scdobbs/GAN_Bathymetry_Project
echo "=== R01b task1 $(date) $(hostname) ==="; python /scratch/users/scdobbs/grl_review/R01b/task1_mechanism.py
echo "=== R01b task2 $(date) ==="; python /scratch/users/scdobbs/grl_review/R01b/task2_transfer_meters.py
echo "=== done $(date) ==="
