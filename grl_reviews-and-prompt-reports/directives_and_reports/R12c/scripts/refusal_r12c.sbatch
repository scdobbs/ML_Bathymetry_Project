#!/bin/bash
#SBATCH --job-name=r12c_refusal
#SBATCH -p serc
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/users/scdobbs/grl_review/R12c/logs/%x_%j.out
#SBATCH --error=/scratch/users/scdobbs/grl_review/R12c/logs/%x_%j.err
# R12c must-hold 5: the derived refusal-test suite (R11's 33 tests + 27f + 30a-e) against the edited phase4_common.py and PREREG_phase4_v2_1.md.
type conda >/dev/null 2>&1 && { conda deactivate >/dev/null 2>&1 || true; conda deactivate >/dev/null 2>&1 || true; }
unset PYTHONPATH; module load python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312; source "${HOME}/bathy_diffusion_env/bin/activate"
export OMP_NUM_THREADS=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1; cd /home/users/scdobbs/GAN_Bathymetry_Project/grl_reviews-and-prompt-reports/directives_and_reports/R12c/scripts
echo "=== $(date) $(hostname) job=${SLURM_JOB_ID} git=$(git rev-parse --short HEAD) ==="
python -B refusal_tests_r12c.py; rc=$?; echo "=== done $(date) rc=$rc ==="; exit $rc
