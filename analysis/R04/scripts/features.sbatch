#!/bin/bash
#SBATCH --job-name=r04_features
#SBATCH -p serc
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/users/scdobbs/grl_review/R04/logs/%x_%j.out
#SBATCH --error=/scratch/users/scdobbs/grl_review/R04/logs/%x_%j.err
type conda >/dev/null 2>&1 && { conda deactivate >/dev/null 2>&1 || true; conda deactivate >/dev/null 2>&1 || true; }
unset PYTHONPATH; module load python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312; source "${HOME}/bathy_diffusion_env/bin/activate"
export OMP_NUM_THREADS=1; cd /home/users/scdobbs/GAN_Bathymetry_Project/grl_reviews-and-prompt-reports/directives_and_reports/R04/scripts; echo "=== $(date) $(hostname) ==="
python build_features.py --workers 32 --leak 100
echo "=== done $(date) ==="
