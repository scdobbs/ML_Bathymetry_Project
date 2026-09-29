#!/bin/bash
#SBATCH --job-name=r03_s4
#SBATCH -p serc
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/users/scdobbs/grl_review/R03/logs/%x_%j.out
#SBATCH --error=/scratch/users/scdobbs/grl_review/R03/logs/%x_%j.err
type conda >/dev/null 2>&1 && { conda deactivate >/dev/null 2>&1 || true; conda deactivate >/dev/null 2>&1 || true; }
unset PYTHONPATH; module load python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312; source "${HOME}/bathy_diffusion_env/bin/activate"
export OMP_NUM_THREADS=1; cd /home/users/scdobbs/GAN_Bathymetry_Project; echo "=== $(date) $(hostname) ==="
python /scratch/users/scdobbs/grl_review/R03/s4_baselines.py
echo "=== done $(date) ==="
