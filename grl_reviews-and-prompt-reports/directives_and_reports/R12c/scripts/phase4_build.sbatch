#!/bin/bash
#SBATCH --job-name=phase4_build
#SBATCH -p serc
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/users/scdobbs/grl_review/R12c/logs/%x_%j.out
#SBATCH --error=/scratch/users/scdobbs/grl_review/R12c/logs/%x_%j.err
# R12c Part E2: rerun of R12 Part C2 under pre-registration v2.1 (tag phase4-prereg-v2.1), same script and arguments as jobs 47445500 / 47541164 (pre-registration v2 B3 step 2): phase4_build.py --mode phase4 on the five sealed pairs, one CPU job, no rehearsal-only argument.
# The script runs its own locks (sign-off, pre-registration, artifacts, code, supplement, acquisition run record) before the allow-list.
# NOTE: this job's log prints each built pair's amplitude statistic (phase4_build.py line 77). Steve must not open it before his calls are committed.
type conda >/dev/null 2>&1 && { conda deactivate >/dev/null 2>&1 || true; conda deactivate >/dev/null 2>&1 || true; }
unset PYTHONPATH; module load python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312; source "${HOME}/bathy_diffusion_env/bin/activate"
export OMP_NUM_THREADS=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1 PYTHONWARNINGS=always; cd /home/users/scdobbs/GAN_Bathymetry_Project/grl_reviews-and-prompt-reports/directives_and_reports/phase4/scripts
PAIRS="KN182L03__MGDS_30193 RC2511__MGDS_32239 SUM1004__MGDS_33090 AT42-03__MGDS_32007 TN159__MGDS_21981"
echo "=== $(date) $(hostname) job=${SLURM_JOB_ID} cpus=${SLURM_CPUS_PER_TASK} git=$(git rev-parse --short HEAD) ==="; grep -m1 "model name" /proc/cpuinfo
echo "CMD: python phase4_build.py --mode phase4 --pairs \"$PAIRS\""
python phase4_build.py --mode phase4 --pairs "$PAIRS"; rc=$?; echo "=== done $(date) rc=$rc ==="; exit $rc
