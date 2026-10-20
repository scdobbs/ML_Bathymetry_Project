#!/bin/bash
#SBATCH --job-name=r12_a2
#SBATCH -p serc
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --output=/scratch/users/scdobbs/grl_review/R12/logs/%x_%j.out
#SBATCH --error=/scratch/users/scdobbs/grl_review/R12/logs/%x_%j.err
# R12 Part A (rehearsal mode, development data, scratch only), one job:
#   1. phase4_build.py --mode rehearsal --rows <k = 32 harness> on H1 (FK151121) and H2 (NA080__MGDS_31289)      -> R12/rehearsal/build, fields, inputs
#   2. the same two pairs built UNMODIFIED (v2.1 rows) in this environment, --out_root R12/rehearsal/unmodified   (warning baseline; F_hr finite fraction)
#   3. phase4_predict.py --mode rehearsal on the harness builds (runs the EW0207 environment control first = §A3; not scored)
#   4. check_a2_r12.py: the five must-hold items, counts only
# Every Phase 4 script runs unchanged through run_phase4_script_r12.py (ROOT[rehearsal] re-pointed to R12). PYTHONWARNINGS=always so that any warning is logged.
set -o pipefail
type conda >/dev/null 2>&1 && { conda deactivate >/dev/null 2>&1 || true; conda deactivate >/dev/null 2>&1 || true; }
unset PYTHONPATH; module load python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312; source "${HOME}/bathy_diffusion_env/bin/activate"
export OMP_NUM_THREADS=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1 PYTHONWARNINGS=always; cd /home/users/scdobbs/GAN_Bathymetry_Project/grl_reviews-and-prompt-reports/directives_and_reports/R12/scripts
LOGS=/scratch/users/scdobbs/grl_review/R12/logs; ROOT=/scratch/users/scdobbs/grl_review/R12/rehearsal; J=${SLURM_JOB_ID}
PAIRS="FK151121__MGDS_22383 NA080__MGDS_31289"
echo "=== $(date) $(hostname) job=$J cpus=${SLURM_CPUS_PER_TASK} git=$(git rev-parse --short HEAD) ==="; grep -m1 "model name" /proc/cpuinfo
echo "--- 1. harness build (k = 32)"; python run_phase4_script_r12.py phase4_build.py --mode rehearsal --pairs "$PAIRS" --rows $ROOT/harness/harness_k32.parquet 2>&1 | tee $LOGS/a2_harness_build_$J.log || { echo "harness build FAILED"; exit 2; }
echo "--- 2. unmodified build (v2.1 rows), same environment"; python run_phase4_script_r12.py phase4_build.py --mode rehearsal --pairs "$PAIRS" --out_root $ROOT/unmodified 2>&1 | tee $LOGS/a2_unmodified_build_$J.log || { echo "unmodified build FAILED"; exit 3; }
echo "--- 3. harness predictions (environment control first)"; python run_phase4_script_r12.py phase4_predict.py --mode rehearsal --pairs "$PAIRS" 2>&1 | tee $LOGS/a2_harness_predict_$J.log || { echo "predict FAILED"; exit 4; }
echo "--- 4. checks"; python check_a2_r12.py $LOGS/a2_harness_build_$J.log $LOGS/a2_harness_predict_$J.log $LOGS/a2_unmodified_build_$J.log 2>&1 | tee $LOGS/a2_checks_$J.log; rc=$?
echo "=== done $(date) rc=$rc ==="; exit $rc
