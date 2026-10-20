#!/bin/bash
#SBATCH --job-name=r04_checks
#SBATCH -p serc
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/users/scdobbs/grl_review/R04/logs/%x_%j.out
#SBATCH --error=/scratch/users/scdobbs/grl_review/R04/logs/%x_%j.err
# §6.2 contract validation on every development pair (+ EW0207), §4 Check A, §5 Check B.
type conda >/dev/null 2>&1 && { conda deactivate >/dev/null 2>&1 || true; conda deactivate >/dev/null 2>&1 || true; }
unset PYTHONPATH; module load python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312; source "${HOME}/bathy_diffusion_env/bin/activate"
export OMP_NUM_THREADS=1; cd /home/users/scdobbs/GAN_Bathymetry_Project/grl_reviews-and-prompt-reports/directives_and_reports/R04/scripts; echo "=== $(date) $(hostname) ==="
mkdir -p /scratch/users/scdobbs/grl_review/R04/validation /scratch/users/scdobbs/grl_review/R04/checks
python contract_v2_1_validate.py --test
python contract_v2_1_validate.py --out /scratch/users/scdobbs/grl_review/R04/validation; echo "validator exit: $?"
python check_a.py
python check_b.py
echo "=== done $(date) ==="
