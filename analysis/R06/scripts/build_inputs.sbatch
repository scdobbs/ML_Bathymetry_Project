#!/bin/bash
#SBATCH --job-name=r06_inputs
#SBATCH -p serc
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/users/scdobbs/grl_review/R06/logs/%x_%j.out
#SBATCH --error=/scratch/users/scdobbs/grl_review/R06/logs/%x_%j.err
# R06 §1.2 / §0.2 rule 3: model inputs for every tiles_v1.1 window through the unchanged R02 path, consistency checks
# against the frozen tables / R04 mosaics, OAK hash check of the field mosaics, input leak test.
type conda >/dev/null 2>&1 && { conda deactivate >/dev/null 2>&1 || true; conda deactivate >/dev/null 2>&1 || true; }
unset PYTHONPATH; module load python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312; source "${HOME}/bathy_diffusion_env/bin/activate"
export OMP_NUM_THREADS=1; cd /home/users/scdobbs/GAN_Bathymetry_Project/grl_reviews-and-prompt-reports/directives_and_reports/R06/scripts; echo "=== $(date) $(hostname) ==="
python build_inputs_r06.py --workers 32 --leak 100 && touch /scratch/users/scdobbs/grl_review/R06/inputs/DONE
echo "=== done $(date) ==="
