#!/bin/bash
#SBATCH --job-name=r02_prep
#SBATCH -p serc
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/users/scdobbs/grl_review/R02/logs/%x_%j.out
#SBATCH --error=/scratch/users/scdobbs/grl_review/R02/logs/%x_%j.err
type conda >/dev/null 2>&1 && { conda deactivate >/dev/null 2>&1 || true; conda deactivate >/dev/null 2>&1 || true; }
unset PYTHONPATH
module load python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312
source "${HOME}/bathy_diffusion_env/bin/activate"
cd /home/users/scdobbs/GAN_Bathymetry_Project
echo "=== R02 clean preprocess $(date) $(hostname) code=$(git rev-parse --short HEAD) ==="
python -m preprocessing.extract_clean_patches \
  --split_path /scratch/users/scdobbs/grl_review/R01/patches_restored/patches_native_phase2/split_assignment.parquet \
  --out_dir /scratch/users/scdobbs/grl_review/R02/patches_clean \
  --hr_px 256 --train_stride_frac 0.25 --eval_stride_frac 0.5 --min_valid 0.50 --flat_threshold_m 0.05 \
  --s_floor 0.5 --leak_test_tiles 100 --workers 9
echo "=== done $(date) ==="
