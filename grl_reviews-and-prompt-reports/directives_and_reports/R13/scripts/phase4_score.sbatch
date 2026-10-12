#!/bin/bash
#SBATCH --job-name=phase4_score
#SBATCH -p serc
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/users/scdobbs/grl_review/R13/logs/%x_%j.out
#SBATCH --error=/scratch/users/scdobbs/grl_review/R13/logs/%x_%j.err
# R13 Part D (pre-registration v2.1 B3 step 5): phase4_score.py --mode phase4 ONCE (single-run lock on scores/results_D1.json), with the
# inputs, joined calls (sha256 in the ledger) and stats assembled by assemble_inputs_r13.py. Then SHA256SUMS rewritten and verified.
type conda >/dev/null 2>&1 && { conda deactivate >/dev/null 2>&1 || true; conda deactivate >/dev/null 2>&1 || true; }
unset PYTHONPATH; module load python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312; source "${HOME}/bathy_diffusion_env/bin/activate"
export OMP_NUM_THREADS=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1; cd /home/users/scdobbs/GAN_Bathymetry_Project/grl_reviews-and-prompt-reports/directives_and_reports/phase4/scripts
P4=/oak/stanford/groups/hilley/auv_ship_colocated_bathy/grl_review_archive/phase4; PAIRS="KN182L03__MGDS_30193 RC2511__MGDS_32239 SUM1004__MGDS_33090 AT42-03__MGDS_32007 TN159__MGDS_21981"
echo "=== $(date) $(hostname) job=${SLURM_JOB_ID} git=$(git rev-parse --short HEAD) ==="; sha256sum $P4/score_inputs/phase4_calls_joined.csv
echo "CMD: python phase4_score.py --mode phase4 --pairs \"$PAIRS\" --inputs $P4/score_inputs/inputs_D1.parquet --calls $P4/score_inputs/phase4_calls_joined.csv --stats $P4/score_inputs/stats.json --tag D1"
python phase4_score.py --mode phase4 --pairs "$PAIRS" --inputs $P4/score_inputs/inputs_D1.parquet --calls $P4/score_inputs/phase4_calls_joined.csv --stats $P4/score_inputs/stats.json --tag D1; rc=$?; echo "=== score rc=$rc ==="
[ $rc -eq 0 ] || exit $rc
( cd "$P4" && find . -type f ! -name 'SHA256SUMS*' -print0 | sort -z | xargs -0 -P 4 -n 32 sha256sum | sort -k2 > /tmp/r13_sums_$$ && mv /tmp/r13_sums_$$ SHA256SUMS && sha256sum -c SHA256SUMS --quiet && echo "SHA256SUMS VERIFIED: $(wc -l < SHA256SUMS) files" ) || { echo "SHA256SUMS FAILED"; exit 4; }
echo "=== done $(date) rc=0 ==="
