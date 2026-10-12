#!/bin/bash
#SBATCH --job-name=phase4_predict
#SBATCH -p serc
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/users/scdobbs/grl_review/R13/logs/%x_%j.out
#SBATCH --error=/scratch/users/scdobbs/grl_review/R13/logs/%x_%j.err
# R13 Part C (pre-registration v2.1 B3 step 4): phase4_predict.py --mode phase4 once, EW0207 environment control first (inside the script),
# calls lock on the committed R13/phase4_calls.csv (sha256 in CHANGE_LEDGER row 40). No rehearsal-only argument.
type conda >/dev/null 2>&1 && { conda deactivate >/dev/null 2>&1 || true; conda deactivate >/dev/null 2>&1 || true; }
unset PYTHONPATH; module load python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312; source "${HOME}/bathy_diffusion_env/bin/activate"
export OMP_NUM_THREADS=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1; cd /home/users/scdobbs/GAN_Bathymetry_Project/grl_reviews-and-prompt-reports/directives_and_reports/phase4/scripts
PAIRS="KN182L03__MGDS_30193 RC2511__MGDS_32239 SUM1004__MGDS_33090 AT42-03__MGDS_32007 TN159__MGDS_21981"; CALLS=/home/users/scdobbs/GAN_Bathymetry_Project/grl_reviews-and-prompt-reports/directives_and_reports/R13/phase4_calls.csv
echo "=== $(date) $(hostname) job=${SLURM_JOB_ID} cpus=${SLURM_CPUS_PER_TASK} git=$(git rev-parse --short HEAD) ==="; grep -m1 "model name" /proc/cpuinfo; sha256sum $CALLS
echo "CMD: python phase4_predict.py --mode phase4 --pairs \"$PAIRS\" --calls $CALLS"
python phase4_predict.py --mode phase4 --pairs "$PAIRS" --calls $CALLS; rc=$?; echo "=== done $(date) rc=$rc ==="; exit $rc
