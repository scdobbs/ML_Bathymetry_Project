#!/bin/bash
#SBATCH --job-name=r16c_dress
#SBATCH -p serc
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/users/scdobbs/grl_review/R16c/logs/%x_%j.out
#SBATCH --error=/scratch/users/scdobbs/grl_review/R16c/logs/%x_%j.err
# R16c §2 (= R16 Part C / R16b rerun, against acquisition freeze v2): mock layout from the ACQ-R11b rehearsal outputs -> build -> figures -> mock calls
# -> predict (EW0207 control first) -> score (main calls; then the artifact-call file) -> checks. Phase 5 scripts unchanged, through run_mock_r16c.py.
set -o pipefail
type conda >/dev/null 2>&1 && { conda deactivate >/dev/null 2>&1 || true; conda deactivate >/dev/null 2>&1 || true; }
unset PYTHONPATH; module load python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312; source "${HOME}/bathy_diffusion_env/bin/activate"
export OMP_NUM_THREADS=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1 PYTHONWARNINGS=always R16C_ROOT=mock; cd /home/users/scdobbs/GAN_Bathymetry_Project/grl_reviews-and-prompt-reports/directives_and_reports/R16c/scripts
L=/scratch/users/scdobbs/grl_review/R16c/logs; ROOT=/scratch/users/scdobbs/grl_review/R16c/rehearsal; J=${SLURM_JOB_ID}; PAIRS="FK151121__MGDS_22383 NA080__MGDS_31289 KM0923__MGDS_31429 KM0923__MOCK_R09repeat"
REC=/scratch/users/scdobbs/grl_review/R16c/mock/reports_post_grl_review/PHASE5/step1_run_rehearsal_r11b.json
echo "=== $(date) $(hostname) job=$J cpus=${SLURM_CPUS_PER_TASK} git=$(git rev-parse --short HEAD) ==="; grep -m1 "model name" /proc/cpuinfo; sha256sum ../../phase5/scripts/*.py
[ -e $ROOT ] && { echo "rehearsal root exists: $ROOT (refusing to mix runs)"; exit 3; }
step () { local name=$1; shift; echo "--- $name: $*"; "$@" 2>&1 | tee $L/dress_${name}_$J.log; local rc=${PIPESTATUS[0]}; echo "=== step $name rc=$rc ===" | tee -a $L/dress_${name}_$J.log; [ $rc -eq 0 ] || { echo "STEP $name FAILED rc=$rc"; exit 2; }; }
step mock      python -B make_mock_r16c.py
step build     python -B run_mock_r16c.py phase5_build.py   --mode rehearsal --pairs "$PAIRS"
step figures   python -B run_mock_r16c.py phase5_figures.py --mode rehearsal --pairs "$PAIRS"
step calls     python -B make_calls_r16c.py
step predict   python -B run_mock_r16c.py phase5_predict.py --mode rehearsal --pairs "$PAIRS" --calls $ROOT/calls/calls_mock.csv
step score     python -B run_mock_r16c.py phase5_score.py   --mode rehearsal --pairs "$PAIRS" --calls $ROOT/calls/calls_mock.csv --acq_record $REC --out $ROOT/scores --tag D1
step score2    python -B run_mock_r16c.py phase5_score.py   --mode rehearsal --pairs "$PAIRS" --calls $ROOT/calls/calls_mock_artifact.csv --acq_record $REC --out $ROOT/scores --tag D1_artifact
python -B check_dress_r16c.py $L/dress_build_$J.log $L/dress_figures_$J.log $L/dress_predict_$J.log $L/dress_score_$J.log $L/dress_score2_$J.log 2>&1 | tee $L/dress_check_$J.log; rc=${PIPESTATUS[0]}
python -B compare_r16c_vs_r16.py 2>&1 | tee $L/dress_compare_$J.log; rc2=${PIPESTATUS[0]}
echo "=== done $(date) rc=$rc compare_rc=$rc2 ==="; [ $rc -eq 0 ] && [ $rc2 -eq 0 ]
