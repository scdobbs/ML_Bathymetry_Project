#!/bin/bash
#SBATCH --job-name=r02_eval
#SBATCH -p serc
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/users/scdobbs/grl_review/R02/logs/%x_%j.out
#SBATCH --error=/scratch/users/scdobbs/grl_review/R02/logs/%x_%j.err
# R02 §3.2-3.7: CPU evaluation on the mosaic re-cut tiles + single-tile secondary column, then visual QA.
type conda >/dev/null 2>&1 && { conda deactivate >/dev/null 2>&1 || true; conda deactivate >/dev/null 2>&1 || true; }
unset PYTHONPATH
module load python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312
source "${HOME}/bathy_diffusion_env/bin/activate"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd /home/users/scdobbs/GAN_Bathymetry_Project
echo "=== R02 eval $(date) $(hostname) ==="
python /scratch/users/scdobbs/grl_review/R02/evaluate.py
echo "=== R02 visual QA $(date) ==="
python /scratch/users/scdobbs/grl_review/R02/visual_qa.py
echo "=== done $(date) ==="
