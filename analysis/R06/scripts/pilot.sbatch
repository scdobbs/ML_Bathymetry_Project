#!/bin/bash
#SBATCH --job-name=r06_pilot
#SBATCH -p serc
#SBATCH -G 1
#SBATCH -C GPU_SKU:H100_SXM5
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/users/scdobbs/grl_review/R06/logs/%x_%j.out
#SBATCH --error=/scratch/users/scdobbs/grl_review/R06/logs/%x_%j.err
# R06 §5.1 pilot: V1, held-out lu00, seed 20261002, full schedule; wall time and GPU-hours recorded in train_log.json.
# Plus a 200-step timing probe of V0 / V2 / V3 (discarded) so the §5.2 extrapolation is per variant.
type conda >/dev/null 2>&1 && { conda deactivate >/dev/null 2>&1 || true; conda deactivate >/dev/null 2>&1 || true; }
unset PYTHONPATH; module load python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312; source "${HOME}/bathy_diffusion_env/bin/activate"
export OMP_NUM_THREADS=1; cd /home/users/scdobbs/GAN_Bathymetry_Project/grl_reviews-and-prompt-reports/directives_and_reports/R06/scripts
echo "=== $(date) $(hostname) GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) ==="
test -f /scratch/users/scdobbs/grl_review/R06/inputs/DONE || { echo "inputs not built — refusing to train"; exit 1; }
python - <<'PY'
import json; d=json.load(open('/scratch/users/scdobbs/grl_review/R06/inputs/leak_test.json'))
assert d['n_identical']==d['n']>0, f"input leak test not clean: {d['n_identical']}/{d['n']}"; print("input leak test gate OK", d['n_identical'], '/', d['n'])
PY
python train_r06.py --variant V1 --fold lu00 --seed 20261002 --workers 8
for v in V0 V2 V3; do python train_r06.py --variant $v --fold lu00 --seed 20261002 --workers 8 --timing_only 200 2>&1 | grep -E "TIMING|Error"; done
python pilot_extrapolate.py
echo "=== done $(date) ==="
