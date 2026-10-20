#!/bin/bash
#SBATCH --job-name=r12_cpu
#SBATCH -p serc
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/users/scdobbs/grl_review/R12/logs/%x_%j.out
#SBATCH --error=/scratch/users/scdobbs/grl_review/R12/logs/%x_%j.err
# Generic CPU step (R12): sbatch --export=ALL,CMD="python ..." [-c 16 --mem 128G -t 01:00:00] cpu_step.sbatch   (pair lists space-separated: --export splits on commas)
type conda >/dev/null 2>&1 && { conda deactivate >/dev/null 2>&1 || true; conda deactivate >/dev/null 2>&1 || true; }
unset PYTHONPATH; module load python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312; source "${HOME}/bathy_diffusion_env/bin/activate"
export OMP_NUM_THREADS=${OMP_THREADS:-1} MPLBACKEND=Agg PYTHONUNBUFFERED=1 PYTHONWARNINGS=${PYTHONWARNINGS:-always}; cd /home/users/scdobbs/GAN_Bathymetry_Project/grl_reviews-and-prompt-reports/directives_and_reports/R12/scripts
echo "=== $(date) $(hostname) job=${SLURM_JOB_ID} cpus=${SLURM_CPUS_PER_TASK} CMD=${CMD:?} ==="; grep -m1 "model name" /proc/cpuinfo; eval "$CMD"; rc=$?; echo "=== done $(date) rc=$rc ==="; exit $rc
