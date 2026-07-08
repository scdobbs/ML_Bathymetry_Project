#!/bin/bash
#SBATCH --job-name=het_louo
#SBATCH -p serc
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPU_SKU:H100_SXM5
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/users/scdobbs/step0_bakeoff/het_louo_%j.out
#SBATCH --error=/scratch/users/scdobbs/step0_bakeoff/het_louo_%j.err
# Overnight §2b: mean-learning heteroscedastic engine, LOUO across 8 native folds.
# Incremental per-fold save; 6h wall caps cost < 10 GPU-hr (single GPU). NA090 fold first.
type conda >/dev/null 2>&1 && { conda deactivate >/dev/null 2>&1 || true; conda deactivate >/dev/null 2>&1 || true; }
unset PYTHONPATH
module load python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312
source "${HOME}/bathy_diffusion_env/bin/activate"
cd "/home/users/${USER}/GAN_Bathymetry_Project"
python /scratch/users/scdobbs/step0_bakeoff/het_louo.py \
    --patches_dir /scratch/groups/hilley/bathy_diffusion_2026/patches_native_phase2 \
    --out_dir /scratch/users/scdobbs/step0_bakeoff/het_louo \
    --steps 8000 --base_ch 32 --batch 32 --cap_per_pair 2500 --workers 10
echo "LOUO Done $(date)."
