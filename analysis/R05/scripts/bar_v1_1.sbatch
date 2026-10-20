#!/bin/bash
#SBATCH --job-name=r05_bar
#SBATCH -p serc
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/users/scdobbs/grl_review/R05/logs/%x_%j.out
#SBATCH --error=/scratch/users/scdobbs/grl_review/R05/logs/%x_%j.err
type conda >/dev/null 2>&1 && { conda deactivate >/dev/null 2>&1 || true; conda deactivate >/dev/null 2>&1 || true; }
unset PYTHONPATH; module load python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312; source "${HOME}/bathy_diffusion_env/bin/activate"
export OMP_NUM_THREADS=1; cd /home/users/scdobbs/GAN_Bathymetry_Project/grl_reviews-and-prompt-reports/directives_and_reports/R05/scripts; echo "=== $(date) $(hostname) ==="
python run_bar_v1_1.py --workers 32
python report_tables_v1_1.py
echo "=== done $(date) ==="
