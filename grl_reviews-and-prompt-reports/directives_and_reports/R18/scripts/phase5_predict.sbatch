#!/bin/bash
#SBATCH --job-name=phase5_predict
#SBATCH -p serc
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/users/scdobbs/grl_review/R18/logs/%x_%j.out
#SBATCH --error=/scratch/users/scdobbs/grl_review/R18/logs/%x_%j.err
# R18 §2 (pre-registration v3.1 B3 step 4): phase5_predict.py --mode phase5 once on CPU, EW0207 environment control first (inside the script; a failed control stops).
type conda >/dev/null 2>&1 && { conda deactivate >/dev/null 2>&1 || true; conda deactivate >/dev/null 2>&1 || true; }
unset PYTHONPATH; module load python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312; source "${HOME}/bathy_diffusion_env/bin/activate"
export OMP_NUM_THREADS=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1 PYTHONWARNINGS=always; cd /home/users/scdobbs/GAN_Bathymetry_Project/grl_reviews-and-prompt-reports/directives_and_reports/phase5/scripts
PAIRS="TN188__MGDS_18210 Escanaba__MGDS_30368 Monterey__MGDS_24756 KN161L04__MGDS_22436 MV1209__MGDS_21415 KN173L02__MGDS_31675 MV1003__MGDS_31860 AT26-24__MGDS_31954"
CALLS=/home/users/scdobbs/GAN_Bathymetry_Project/grl_reviews-and-prompt-reports/directives_and_reports/R18/phase5_calls.csv
echo "=== $(date) $(hostname) job=${SLURM_JOB_ID} cpus=${SLURM_CPUS_PER_TASK} git=$(git rev-parse --short HEAD) ==="; grep -m1 "model name" /proc/cpuinfo
echo "CMD: python phase5_predict.py --mode phase5 --pairs \"$PAIRS\" --calls $CALLS"
python phase5_predict.py --mode phase5 --pairs "$PAIRS" --calls $CALLS; rc=$?; echo "=== done $(date) rc=$rc ==="; exit $rc
