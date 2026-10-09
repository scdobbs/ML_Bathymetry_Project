#!/bin/bash
#SBATCH --job-name=r07_cpu
#SBATCH -p serc
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/users/scdobbs/grl_review/R07/logs/%x_%j.out
#SBATCH --error=/scratch/users/scdobbs/grl_review/R07/logs/%x_%j.err
# Generic CPU step (scorer self-test, scoring, estimate, freeze): sbatch --export=ALL,CMD="python score_r07.py" cpu_step.sbatch
# Keeps sustained computation off the login nodes.
type conda >/dev/null 2>&1 && { conda deactivate >/dev/null 2>&1 || true; conda deactivate >/dev/null 2>&1 || true; }
unset PYTHONPATH; module load python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312; source "${HOME}/bathy_diffusion_env/bin/activate"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONUNBUFFERED=1; cd /home/users/scdobbs/GAN_Bathymetry_Project/grl_reviews-and-prompt-reports/directives_and_reports/R07/scripts
echo "=== $(date) $(hostname) CMD=${CMD:?} ==="; eval "$CMD"; rc=$?; echo "=== done $(date) rc=$rc ==="; exit $rc
