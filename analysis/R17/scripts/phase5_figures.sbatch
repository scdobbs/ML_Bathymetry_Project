#!/bin/bash
#SBATCH --job-name=phase5_figures
#SBATCH -p serc
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/users/scdobbs/grl_review/R17/logs/%x_%j.out
#SBATCH --error=/scratch/users/scdobbs/grl_review/R17/logs/%x_%j.err
# R17 §2.4–2.5 (pre-registration v3.1 B3 step 2): phase5_figures.py --mode phase5 once on the eight sites (not-testable sites get no page and are
# listed in the key), then the page checks (pypdf, page count, codes from file names; key never opened), the calls template, and SHA256SUMS of phase5/.
set -o pipefail
type conda >/dev/null 2>&1 && { conda deactivate >/dev/null 2>&1 || true; conda deactivate >/dev/null 2>&1 || true; }
unset PYTHONPATH; module load python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312; source "${HOME}/bathy_diffusion_env/bin/activate"
export OMP_NUM_THREADS=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1; P5=/oak/stanford/groups/hilley/auv_ship_colocated_bathy/grl_review_archive/phase5
cd /home/users/scdobbs/GAN_Bathymetry_Project/grl_reviews-and-prompt-reports/directives_and_reports/phase5/scripts
PAIRS="TN188__MGDS_18210 Escanaba__MGDS_30368 Monterey__MGDS_24756 KN161L04__MGDS_22436 MV1209__MGDS_21415 KN173L02__MGDS_31675 MV1003__MGDS_31860 AT26-24__MGDS_31954"
echo "=== $(date) $(hostname) job=${SLURM_JOB_ID} git=$(git rev-parse --short HEAD) ==="
echo "CMD: python phase5_figures.py --mode phase5 --pairs \"$PAIRS\""
python phase5_figures.py --mode phase5 --pairs "$PAIRS" || { echo "figures FAILED"; exit 2; }
cd ../../R17/scripts && python -B check_pages_r17.py || { echo "page check FAILED"; exit 3; }
( cd "$P5" && find . -type f ! -name 'SHA256SUMS*' -print0 | sort -z | xargs -0 -P 4 -n 32 sha256sum | sort -k2 > SHA256SUMS.tmp && mv SHA256SUMS.tmp SHA256SUMS && sha256sum -c SHA256SUMS --quiet && echo "SHA256SUMS VERIFIED: $(wc -l < SHA256SUMS) files" ) || { echo "SHA256SUMS FAILED"; exit 4; }
echo "=== done $(date) rc=0 ==="
