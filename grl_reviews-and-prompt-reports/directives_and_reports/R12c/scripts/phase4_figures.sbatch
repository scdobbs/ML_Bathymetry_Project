#!/bin/bash
#SBATCH --job-name=phase4_figures
#SBATCH -p serc
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/users/scdobbs/grl_review/R12c/logs/%x_%j.out
#SBATCH --error=/scratch/users/scdobbs/grl_review/R12c/logs/%x_%j.err
# R12c Part E2 (pre-registration v2.1 B3 step 3; R12b Part C): phase4_figures.py --mode phase4 once on the five sealed pairs (a no-k pair gets no page), then
# the page checks (pypdf text extraction, page count, codes from file names; key never opened), the calls template, and SHA256SUMS updated.
# NOTE: phase4_figures.py prints the no-k pair list only (no amplitude); fields_r08 prints nothing per page.
set -o pipefail
type conda >/dev/null 2>&1 && { conda deactivate >/dev/null 2>&1 || true; conda deactivate >/dev/null 2>&1 || true; }
unset PYTHONPATH; module load python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312; source "${HOME}/bathy_diffusion_env/bin/activate"
export OMP_NUM_THREADS=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1; P4=/oak/stanford/groups/hilley/auv_ship_colocated_bathy/grl_review_archive/phase4
cd /home/users/scdobbs/GAN_Bathymetry_Project/grl_reviews-and-prompt-reports/directives_and_reports/phase4/scripts
PAIRS="KN182L03__MGDS_30193 RC2511__MGDS_32239 SUM1004__MGDS_33090 AT42-03__MGDS_32007 TN159__MGDS_21981"
echo "=== $(date) $(hostname) job=${SLURM_JOB_ID} git=$(git rev-parse --short HEAD) ==="; grep -m1 "model name" /proc/cpuinfo
echo "CMD: python phase4_figures.py --mode phase4 --pairs \"$PAIRS\""
python phase4_figures.py --mode phase4 --pairs "$PAIRS" || { echo "figures FAILED"; exit 2; }
cd ../R12c/scripts && python -B check_pages_r12c.py || { echo "page check FAILED"; exit 3; }
( cd "$P4" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 -P 4 -n 32 sha256sum | sort -k2 > SHA256SUMS && sha256sum -c SHA256SUMS --quiet && echo "SHA256SUMS VERIFIED: $(wc -l < SHA256SUMS) files" ) || { echo "SHA256SUMS FAILED"; exit 4; }
echo "=== done $(date) rc=0 ==="
