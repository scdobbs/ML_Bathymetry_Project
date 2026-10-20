#!/bin/bash
#SBATCH --job-name=r03_feat_nat
#SBATCH -p serc
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/users/scdobbs/grl_review/R03/logs/%x_%j.out
#SBATCH --error=/scratch/users/scdobbs/grl_review/R03/logs/%x_%j.err
type conda >/dev/null 2>&1 && { conda deactivate >/dev/null 2>&1 || true; conda deactivate >/dev/null 2>&1 || true; }
unset PYTHONPATH; module load python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312; source "${HOME}/bathy_diffusion_env/bin/activate"
export OMP_NUM_THREADS=1; cd /home/users/scdobbs/GAN_Bathymetry_Project; echo "=== $(date) $(hostname) ==="
python /scratch/users/scdobbs/grl_review/R03/features_context.py --patches /scratch/users/scdobbs/grl_review/R02/patches_clean \
  --units NA090,NA080,TN399,RR1506,AT37-13,FK181031,TN299,EW0207 --cap 200 --out /scratch/users/scdobbs/grl_review/R03/s4/features_native.parquet --leak 100
echo "=== done $(date) ==="
