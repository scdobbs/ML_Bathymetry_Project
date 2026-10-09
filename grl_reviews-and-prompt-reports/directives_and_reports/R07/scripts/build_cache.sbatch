#!/bin/bash
#SBATCH --job-name=r07_cache
#SBATCH -p serc
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/users/scdobbs/grl_review/R07/logs/%x_%j.out
#SBATCH --error=/scratch/users/scdobbs/grl_review/R07/logs/%x_%j.err
# R07 §1: (1) the §1.4 equivalence test on 200 development windows (stops the directive if it fails), then (2) the
# pretraining window cache on $SCRATCH from the staged corpus (never from OAK).
set -uo pipefail
type conda >/dev/null 2>&1 && { conda deactivate >/dev/null 2>&1 || true; conda deactivate >/dev/null 2>&1 || true; }
unset PYTHONPATH; module load python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312; source "${HOME}/bathy_diffusion_env/bin/activate"
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 GDAL_CACHEMAX=512; cd /home/users/scdobbs/GAN_Bathymetry_Project/grl_reviews-and-prompt-reports/directives_and_reports/R07/scripts
echo "=== $(date) $(hostname) ==="
test -f /scratch/users/scdobbs/grl_review/R07/corpus/STAGED_OK || { echo "corpus not staged"; exit 1; }
python equiv_test_r07.py 200 || { echo "§1.4 EQUIVALENCE TEST FAILED — stopping (directive §1.4)"; exit 2; }
python build_cache_r07.py --workers 32 || { echo "CACHE BUILD FAILED"; exit 3; }
du -sh /scratch/users/scdobbs/grl_review/R07/cache; echo "=== done $(date) ==="
