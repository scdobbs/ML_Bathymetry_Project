#!/bin/bash
#SBATCH --job-name=r07_pretrain
#SBATCH -p serc,gpu
#SBATCH -G 1
#SBATCH -C GPU_SKU:H100_SXM5
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH --time=20:00:00
#SBATCH --output=/scratch/users/scdobbs/grl_review/R07/logs/%x_%j.out
#SBATCH --error=/scratch/users/scdobbs/grl_review/R07/logs/%x_%j.err
# R07 §2: pretraining of one body. Usage: sbatch --export=ALL,BODY=P-V1[,PILOT=1] pretrain.sbatch
# The window cache is staged once to node-local disk ($L_SCRATCH_JOB) when it fits; training never reads OAK.
set -uo pipefail
type conda >/dev/null 2>&1 && { conda deactivate >/dev/null 2>&1 || true; conda deactivate >/dev/null 2>&1 || true; }
unset PYTHONPATH; module load python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312; source "${HOME}/bathy_diffusion_env/bin/activate"
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1; cd /home/users/scdobbs/GAN_Bathymetry_Project/grl_reviews-and-prompt-reports/directives_and_reports/R07/scripts
CACHE=/scratch/users/scdobbs/grl_review/R07/cache; BODY=${BODY:?}; PILOT=${PILOT:-0}
echo "=== $(date) $(hostname) GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) body=$BODY pilot=$PILOT ==="
test -f $CACHE/DONE || { echo "cache not built"; exit 1; }
NEED=$(du -sm $CACHE | cut -f1); AVAIL=$(df -m --output=avail "$L_SCRATCH" | tail -1); echo "cache ${NEED} MB; node-local avail ${AVAIL} MB"
USE=$CACHE
if [ "$AVAIL" -gt $(( NEED + 2048 )) ]; then
  mkdir -p "$L_SCRATCH_JOB/cache"; t0=$(date +%s); rsync -rlt "$CACHE/" "$L_SCRATCH_JOB/cache/" && USE="$L_SCRATCH_JOB/cache" && echo "staged cache to node-local disk in $(( $(date +%s) - t0 )) s"
else echo "node-local disk too small; reading the cache from \$SCRATCH"; fi
if [ "$PILOT" = "1" ]; then python pretrain_r07.py --body "$BODY" --pilot --cache "$USE" --workers 16 --eval_workers 8
else
  # §2.3 / §6 gates: the pilot must have passed its rule and the estimate (pilot + §3.3 timings) must be within the 300 GPU-h cap
  python -c "import json; d=json.load(open('/scratch/users/scdobbs/grl_review/R07/pretrain/P-V1_pilot/pretrain_log.json')); assert d['pilot_rule']['verdict']=='PASS', d['pilot_rule']" || { echo "PILOT RULE NOT PASSED (HOLD) — not starting the full pretraining"; exit 2; }
  test -f /scratch/users/scdobbs/grl_review/R07/compute_estimate.json || python estimate_r07.py || { echo "estimate failed"; exit 3; }
  python -c "import json; d=json.load(open('/scratch/users/scdobbs/grl_review/R07/compute_estimate.json')); assert d['within_budget'], d['gpu_h_estimate']" || { echo "ESTIMATE EXCEEDS THE BUDGET (HOLD) — not starting"; exit 4; }
  python pretrain_r07.py --body "$BODY" --cache "$USE" --workers 12 --eval_workers 6 --resume
fi
rc=$?; echo "=== done $(date) rc=$rc ==="; exit $rc
