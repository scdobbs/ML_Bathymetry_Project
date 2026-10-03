#!/bin/bash
#SBATCH --job-name=r06_train
#SBATCH -p serc,gpu
#SBATCH -G 1
#SBATCH -C "GPU_SKU:H100_SXM5|GPU_SKU:A100_SXM4"
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/users/scdobbs/grl_review/R06/logs/%x_%A_%a.out
#SBATCH --error=/scratch/users/scdobbs/grl_review/R06/logs/%x_%A_%a.err
# R06 §2 / §5.2 full run: array over tasks.txt (variant seed folds). Submit with --array=1-N after the pilot passed §5.3.
type conda >/dev/null 2>&1 && { conda deactivate >/dev/null 2>&1 || true; conda deactivate >/dev/null 2>&1 || true; }
unset PYTHONPATH; module load python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312; source "${HOME}/bathy_diffusion_env/bin/activate"
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1; cd /home/users/scdobbs/GAN_Bathymetry_Project/grl_reviews-and-prompt-reports/directives_and_reports/R06/scripts
echo "=== $(date) $(hostname) GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) task=$SLURM_ARRAY_TASK_ID ==="
test -f /scratch/users/scdobbs/grl_review/R06/inputs/DONE || { echo "inputs not built"; exit 1; }
test -f /scratch/users/scdobbs/grl_review/R06/compute_estimate.json || { echo "no pilot estimate — refusing to run the full set"; exit 1; }
python -c "import json; d=json.load(open('/scratch/users/scdobbs/grl_review/R06/compute_estimate.json')); assert d['within_budget'], 'pilot estimate exceeds the budget'"
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" tasks.txt); read -r VAR SEED FOLDS <<< "$LINE"; echo "task: variant=$VAR seed=$SEED folds=$FOLDS"
for f in ${FOLDS//,/ }; do
  python train_r06.py --variant "$VAR" --fold "$f" --seed "$SEED" --workers 8 || { echo "FAILED $VAR $SEED $f"; exit 1; }
done
echo "=== done $(date) ==="
