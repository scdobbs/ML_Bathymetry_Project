#!/bin/bash
#SBATCH --job-name=r12c_dress
#SBATCH -p serc
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/users/scdobbs/grl_review/R12c/logs/%x_%j.out
#SBATCH --error=/scratch/users/scdobbs/grl_review/R12c/logs/%x_%j.err
# R12c Part C (rehearsal mode, development data copied into a sealed-style layout, scratch only), one job, in the directive's order:
#   C1 make_mock -> 1 phase4_build.py -> 2 phase4_figures.py -> 3 mock calls file (+ score inputs) -> 4 phase4_predict.py (EW0207 control first) -> 5 phase4_score.py -> checks.
# Every Phase 4 script runs unchanged through run_mock_r12c.py (start -> rehearsal locks -> mock tokens treated as sealed -> apply_allowlist -> probes -> opened-path recorder).
set -o pipefail
type conda >/dev/null 2>&1 && { conda deactivate >/dev/null 2>&1 || true; conda deactivate >/dev/null 2>&1 || true; }
unset PYTHONPATH; module load python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312; source "${HOME}/bathy_diffusion_env/bin/activate"
export OMP_NUM_THREADS=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1 PYTHONWARNINGS=always R12C_ROOT=mock; cd /home/users/scdobbs/GAN_Bathymetry_Project/grl_reviews-and-prompt-reports/directives_and_reports/R12c/scripts
L=/scratch/users/scdobbs/grl_review/R12c/logs; ROOT=/scratch/users/scdobbs/grl_review/R12c/rehearsal; J=${SLURM_JOB_ID}; PAIRS="FK151121__MGDS_22383 TN399__MGDS_33068 KM0923__MGDS_31429"; SUP=/scratch/users/scdobbs/grl_review/R12c/mock/mock_supplement.parquet
echo "=== $(date) $(hostname) job=$J cpus=${SLURM_CPUS_PER_TASK} git=$(git rev-parse --short HEAD) ==="; grep -m1 "model name" /proc/cpuinfo
step () { local name=$1; shift; echo "--- $name: $*"; "$@" 2>&1 | tee $L/dress_${name}_$J.log; local rc=${PIPESTATUS[0]}; echo "=== step $name rc=$rc ===" | tee -a $L/dress_${name}_$J.log; [ $rc -eq 0 ] || { echo "STEP $name FAILED rc=$rc"; exit 2; }; }
step mock    python -B make_mock_r12c.py
step build   python -B run_mock_r12c.py phase4_build.py   --mode rehearsal --pairs "$PAIRS" --rows $SUP
step figures python -B run_mock_r12c.py phase4_figures.py --mode rehearsal --pairs "$PAIRS"
step predict python -B run_mock_r12c.py phase4_predict.py --mode rehearsal --pairs "$PAIRS"
step inputs  python -B assemble_inputs_r12c.py
step score   python -B run_mock_r12c.py phase4_score.py   --mode rehearsal --pairs "$PAIRS" --rows $SUP --inputs $ROOT/score_inputs/inputs_D1.parquet --calls $ROOT/score_inputs/calls_mock.csv --stats $ROOT/score_inputs/stats.json --out $ROOT/scores --tag D1
python -B check_dress_r12c.py $L/dress_build_$J.log $L/dress_figures_$J.log $L/dress_predict_$J.log $L/dress_score_$J.log 2>&1 | tee $L/dress_check_$J.log; rc=${PIPESTATUS[0]}
echo "=== done $(date) rc=$rc ==="; exit $rc
