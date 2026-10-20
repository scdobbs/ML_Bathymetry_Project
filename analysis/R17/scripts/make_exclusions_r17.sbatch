#!/bin/bash
#SBATCH --job-name=r17_exclusions
#SBATCH -p serc
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/users/scdobbs/grl_review/R17/logs/%x_%j.out
#SBATCH --error=/scratch/users/scdobbs/grl_review/R17/logs/%x_%j.err
# R17 H1 ruling: the fixed exclusion list (make_exclusions_r17.py), tile ids and counts only.
set -o pipefail
type conda >/dev/null 2>&1 && { conda deactivate >/dev/null 2>&1 || true; conda deactivate >/dev/null 2>&1 || true; }
unset PYTHONPATH; module load python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312; source "${HOME}/bathy_diffusion_env/bin/activate"
export OMP_NUM_THREADS=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1; cd /home/users/scdobbs/GAN_Bathymetry_Project/grl_reviews-and-prompt-reports/directives_and_reports/R17/scripts
echo "=== $(date) $(hostname) job=${SLURM_JOB_ID} git=$(git rev-parse --short HEAD) ==="
python -B make_exclusions_r17.py; rc=$?; echo "=== done $(date) rc=$rc ==="; exit $rc
