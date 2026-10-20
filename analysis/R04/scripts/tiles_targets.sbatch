#!/bin/bash
#SBATCH --job-name=r04_tiles
#SBATCH -p serc
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/users/scdobbs/grl_review/R04/logs/%x_%j.out
#SBATCH --error=/scratch/users/scdobbs/grl_review/R04/logs/%x_%j.err
# R04 §1: windows, tiles_v1, band-passed fields, targets, R02 sigma_m means, grid checks, §1.1 consistency check.
type conda >/dev/null 2>&1 && { conda deactivate >/dev/null 2>&1 || true; conda deactivate >/dev/null 2>&1 || true; }
unset PYTHONPATH; module load python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312; source "${HOME}/bathy_diffusion_env/bin/activate"
export OMP_NUM_THREADS=1; cd /home/users/scdobbs/GAN_Bathymetry_Project/grl_reviews-and-prompt-reports/directives_and_reports/R04/scripts; echo "=== $(date) $(hostname) ==="
python build_tiles_targets.py --workers 16
echo "=== done $(date) ==="
