#!/bin/bash
#SBATCH --job-name=r12c_repro
#SBATCH -p serc
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/users/scdobbs/grl_review/R12c/logs/%x_%j.out
#SBATCH --error=/scratch/users/scdobbs/grl_review/R12c/logs/%x_%j.err
# R12c must-hold 4: plain rehearsal builds of the three R10 pairs (KM0923, AR26, EW0207) with the edited phase4_common.py, then compare_r11.a1 against the frozen artifacts.
set -o pipefail
type conda >/dev/null 2>&1 && { conda deactivate >/dev/null 2>&1 || true; conda deactivate >/dev/null 2>&1 || true; }
unset PYTHONPATH; module load python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312; source "${HOME}/bathy_diffusion_env/bin/activate"
export OMP_NUM_THREADS=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1 R12C_ROOT=repro; cd /home/users/scdobbs/GAN_Bathymetry_Project/grl_reviews-and-prompt-reports/directives_and_reports/R12c/scripts
echo "=== $(date) $(hostname) job=${SLURM_JOB_ID} git=$(git rev-parse --short HEAD) ==="; grep -m1 "model name" /proc/cpuinfo
python -B run_plain_r12c.py phase4_build.py --mode rehearsal --pairs "KM0923__MGDS_31429 AR26__MGDS_31838 EW0207__MGDS_32556" || { echo "build FAILED"; exit 2; }
python -B repro_r10_r12c.py; rc=$?; echo "=== done $(date) rc=$rc ==="; exit $rc
