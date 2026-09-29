#!/bin/bash
#SBATCH --job-name=r02_train
#SBATCH -p serc
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPU_SKU:H100_SXM5
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=03:30:00
#SBATCH --output=/scratch/users/scdobbs/grl_review/R02/logs/%x_%j.out
#SBATCH --error=/scratch/users/scdobbs/grl_review/R02/logs/%x_%j.err
type conda >/dev/null 2>&1 && { conda deactivate >/dev/null 2>&1 || true; conda deactivate >/dev/null 2>&1 || true; }
unset PYTHONPATH
module load python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312
source "${HOME}/bathy_diffusion_env/bin/activate"
cd /home/users/scdobbs/GAN_Bathymetry_Project
# gate: the leak test must have passed (preprocess job exits non-zero otherwise; afterok dependency enforces it)
test -f /scratch/users/scdobbs/grl_review/R02/patches_clean/leak_test.json || { echo "no leak_test.json — refusing to train"; exit 1; }
python - <<'PY'
import json; d=json.load(open('/scratch/users/scdobbs/grl_review/R02/patches_clean/leak_test.json'))
assert d['n_pass']==d['n']>0, f"leak test not clean: {d['n_pass']}/{d['n']}"; print("leak test gate OK", d['n_pass'], '/', d['n'])
PY
echo "=== R02 train $(date) $(hostname) GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) ==="
python /scratch/users/scdobbs/grl_review/R02/het_clean.py --workers 10 --seed 20260928
echo "=== done $(date) ==="
