#!/bin/bash
#SBATCH --job-name=r02_mosaic_cpu
#SBATCH -p serc
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/users/scdobbs/grl_review/R02/logs/%x_%j.out
#SBATCH --error=/scratch/users/scdobbs/grl_review/R02/logs/%x_%j.err
# R02 §3.1 mosaics for ALL 9 pairs on CPU (Steve's ruling 2026-09-29 after the GPU gate was exceeded: 5.14 GPU-h billed).
# LR tiles come from lr_tile_cache/ (exact ecp.tile_ship_side output, job 45897637). rf_probe/border_profile already
# written by GPU job 45797098 (--skip_probe). GPU partial results (NA090, NA080) kept in mosaics_gpu_partial/.
type conda >/dev/null 2>&1 && { conda deactivate >/dev/null 2>&1 || true; conda deactivate >/dev/null 2>&1 || true; }
unset PYTHONPATH; module load python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312; source "${HOME}/bathy_diffusion_env/bin/activate"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd /home/users/scdobbs/GAN_Bathymetry_Project; echo "=== R02 mosaic (CPU) $(date) $(hostname) threads=$OMP_NUM_THREADS ==="
python /scratch/users/scdobbs/grl_review/R02/mosaic_infer.py --rim 48 --s_floor 0.5 --skip_probe --skip_done --device cpu
echo "=== done $(date) ==="
