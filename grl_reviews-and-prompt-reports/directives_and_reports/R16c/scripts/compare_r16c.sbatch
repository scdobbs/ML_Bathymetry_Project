#!/bin/bash
#SBATCH --job-name=r16c_compare
#SBATCH -p serc
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/users/scdobbs/grl_review/R16c/logs/%x_%j.out
#SBATCH --error=/scratch/users/scdobbs/grl_review/R16c/logs/%x_%j.err
# R16c must-hold 3.2 rerun of compare_r16c_vs_r16.py on the finished dress rehearsal (job 47710277) after dropping the k-provenance acq_record path key.
type conda >/dev/null 2>&1 && { conda deactivate >/dev/null 2>&1 || true; conda deactivate >/dev/null 2>&1 || true; }
unset PYTHONPATH; module load python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312; source "${HOME}/bathy_diffusion_env/bin/activate"
export PYTHONUNBUFFERED=1 R16C_ROOT=mock; cd /home/users/scdobbs/GAN_Bathymetry_Project/grl_reviews-and-prompt-reports/directives_and_reports/R16c/scripts
python -B compare_r16c_vs_r16.py; rc=$?; echo "=== done $(date) rc=$rc ==="; exit $rc
