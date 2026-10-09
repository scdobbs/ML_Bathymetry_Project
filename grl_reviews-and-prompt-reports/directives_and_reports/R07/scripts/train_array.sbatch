#!/bin/bash
#SBATCH --job-name=r07_train
#SBATCH -p serc,gpu
#SBATCH -G 1
#SBATCH -C "GPU_SKU:H100_SXM5|GPU_SKU:A100_SXM4"
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/users/scdobbs/grl_review/R07/logs/%x_%A_%a.out
#SBATCH --error=/scratch/users/scdobbs/grl_review/R07/logs/%x_%j.err
# R07 §3 fine-tuning array. Usage: sbatch --array=1-N%8 --export=ALL,TASKS=tasks_pretrained.txt train_array.sbatch
# (R06 train_array.sbatch with the R07 trainer; the init checkpoint is derived from the arm.) At most 8 concurrent GPUs on serc.
type conda >/dev/null 2>&1 && { conda deactivate >/dev/null 2>&1 || true; conda deactivate >/dev/null 2>&1 || true; }
unset PYTHONPATH; module load python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312; source "${HOME}/bathy_diffusion_env/bin/activate"
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1; cd /home/users/scdobbs/GAN_Bathymetry_Project/grl_reviews-and-prompt-reports/directives_and_reports/R07/scripts
R07=/scratch/users/scdobbs/grl_review/R07; TASKS=${TASKS:?}
echo "=== $(date) $(hostname) GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) task=$SLURM_ARRAY_TASK_ID tasks=$TASKS ==="
test -f /scratch/users/scdobbs/grl_review/R06/inputs/DONE || { echo "R06 inputs not built"; exit 1; }
test -f $R07/checks/scratch_equiv/result.json || { echo "no scratch-equivalence result — refusing to fine-tune (§3.3)"; exit 1; }
python -c "import json; d=json.load(open('$R07/checks/scratch_equiv/result.json')); assert d['predictions_identical'], 'scratch-equivalence test not identical (HOLD)'"
python -c "import json; d=json.load(open('$R07/pretrain/P-V1_pilot/pretrain_log.json')); assert d['pilot_rule']['verdict']=='PASS', d['pilot_rule']" || { echo "pilot rule not passed (HOLD)"; exit 1; }
test -f $R07/compute_estimate.json || python estimate_r07.py || { echo "no compute estimate — refusing to run (§6)"; exit 1; }
python -c "import json; d=json.load(open('$R07/compute_estimate.json')); assert d['within_budget'], 'estimate exceeds the budget (HOLD)'"
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASKS"); read -r ARM SEED FOLDS <<< "$LINE"; echo "task: arm=$ARM seed=$SEED folds=$FOLDS"
INIT=""; case "$ARM" in P-V1) INIT="--init $R07/pretrain/P-V1/body_P-V1.pt";; P-V2) INIT="--init $R07/pretrain/P-V2/body_P-V2.pt";; esac
if [ -n "$INIT" ]; then test -f "${INIT#--init }" || { echo "pretrained body missing: $INIT"; exit 1; }; fi
for f in ${FOLDS//,/ }; do
  python train_r07.py --arm "$ARM" --fold "$f" --seed "$SEED" --workers 8 $INIT || { echo "FAILED $ARM $SEED $f"; exit 1; }
done
echo "=== done $(date) ==="
