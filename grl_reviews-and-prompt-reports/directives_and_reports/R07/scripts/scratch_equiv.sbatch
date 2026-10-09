#!/bin/bash
#SBATCH --job-name=r07_equiv
#SBATCH -p serc
#SBATCH -G 1
#SBATCH -C GPU_SKU:H100_SXM5
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/users/scdobbs/grl_review/R07/logs/%x_%j.out
#SBATCH --error=/scratch/users/scdobbs/grl_review/R07/logs/%x_%j.err
# R07 §3.3 scratch-equivalence test: R07 training code, init none, V1 / lu00 / 20261002 on an H100 (R06's GPU for that fold).
type conda >/dev/null 2>&1 && { conda deactivate >/dev/null 2>&1 || true; conda deactivate >/dev/null 2>&1 || true; }
unset PYTHONPATH; module load python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312; source "${HOME}/bathy_diffusion_env/bin/activate"
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1; cd /home/users/scdobbs/GAN_Bathymetry_Project/grl_reviews-and-prompt-reports/directives_and_reports/R07/scripts
echo "=== $(date) $(hostname) GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) ==="
test -f /scratch/users/scdobbs/grl_review/R06/inputs/DONE || { echo "R06 inputs not present"; exit 1; }
python scratch_equiv_r07.py
echo "=== done $(date) ==="
