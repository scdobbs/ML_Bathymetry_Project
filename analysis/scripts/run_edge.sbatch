#!/bin/bash
#SBATCH --job-name=grl_edge
#SBATCH -p serc
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/users/scdobbs/grl_review/%x_%j.out
#SBATCH --error=/scratch/users/scdobbs/grl_review/%x_%j.err
type conda >/dev/null 2>&1 && { conda deactivate >/dev/null 2>&1 || true; conda deactivate >/dev/null 2>&1 || true; }
unset PYTHONPATH
module load python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312
source "${HOME}/bathy_diffusion_env/bin/activate"
cd "/home/users/${USER}/GAN_Bathymetry_Project"
echo "=== running edge_followup.py $(date) on $(hostname) ==="
python /scratch/users/scdobbs/grl_review/edge_followup.py
echo "=== done $(date) ==="
