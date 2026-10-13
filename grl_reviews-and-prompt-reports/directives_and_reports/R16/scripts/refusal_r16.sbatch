#!/bin/bash
#SBATCH --job-name=r16_refusal
#SBATCH -p serc
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/users/scdobbs/grl_review/R16/logs/%x_%j.out
#SBATCH --error=/scratch/users/scdobbs/grl_review/R16/logs/%x_%j.err
# R16 must-hold 3.9: R12c's 39 refusal tests ported to Phase 5 + the tests for items 3.2-3.6 and the supplement locks, against the Phase 5 scripts and R16/PREREG_phase5_v3.md.
type conda >/dev/null 2>&1 && { conda deactivate >/dev/null 2>&1 || true; conda deactivate >/dev/null 2>&1 || true; }
unset PYTHONPATH; module load python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312; source "${HOME}/bathy_diffusion_env/bin/activate"
export OMP_NUM_THREADS=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1 R16_ROOT=mock; cd /home/users/scdobbs/GAN_Bathymetry_Project/grl_reviews-and-prompt-reports/directives_and_reports/R16/scripts
echo "=== $(date) $(hostname) job=${SLURM_JOB_ID} git=$(git rev-parse --short HEAD) ==="; sha256sum ../PREREG_phase5_v3.md ../../phase5/scripts/*.py
python -B refusal_tests_r16.py; rc=$?; echo "=== done $(date) rc=$rc ==="; exit $rc
