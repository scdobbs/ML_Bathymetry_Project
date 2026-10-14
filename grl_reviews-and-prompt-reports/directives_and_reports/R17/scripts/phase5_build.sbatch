#!/bin/bash
#SBATCH --job-name=phase5_build
#SBATCH -p serc
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/users/scdobbs/grl_review/R17/logs/%x_%j.out
#SBATCH --error=/scratch/users/scdobbs/grl_review/R17/logs/%x_%j.err
# R17 §2.1 (pre-registration v3.1 B3 step 2): phase5_build.py --mode phase5 on the eight sites (the five eligible are built) (ACQ-R13, freeze v4.2), one CPU job, no rehearsal-only argument.
# The script runs its own locks (sign-off, pre-registration, artifacts, code, supplement, acquisition run record) before the allow-list.
# NOTE: this job's log prints each built site's amplitude statistic (phase4_build.py). Steve must not open it before his calls are committed.
type conda >/dev/null 2>&1 && { conda deactivate >/dev/null 2>&1 || true; conda deactivate >/dev/null 2>&1 || true; }
unset PYTHONPATH; module load python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312; source "${HOME}/bathy_diffusion_env/bin/activate"
export OMP_NUM_THREADS=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1 PYTHONWARNINGS=always; cd /home/users/scdobbs/GAN_Bathymetry_Project/grl_reviews-and-prompt-reports/directives_and_reports/phase5/scripts
PAIRS="TN188__MGDS_18210 Escanaba__MGDS_30368 Monterey__MGDS_24756 KN161L04__MGDS_22436 MV1209__MGDS_21415 KN173L02__MGDS_31675 MV1003__MGDS_31860 AT26-24__MGDS_31954"   # all eight (v3.1 B3 step 2: a not-testable site is not opened; only its status is written)
echo "=== $(date) $(hostname) job=${SLURM_JOB_ID} cpus=${SLURM_CPUS_PER_TASK} git=$(git rev-parse --short HEAD) ==="; grep -m1 "model name" /proc/cpuinfo
echo "CMD: python phase5_build.py --mode phase5 --pairs \"$PAIRS\""
python phase5_build.py --mode phase5 --pairs "$PAIRS"; rc=$?; echo "=== done $(date) rc=$rc ==="; exit $rc
