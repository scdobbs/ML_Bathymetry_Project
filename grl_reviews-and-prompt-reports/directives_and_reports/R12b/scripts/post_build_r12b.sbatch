#!/bin/bash
#SBATCH --job-name=r12b_post
#SBATCH -p serc
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/users/scdobbs/grl_review/R12b/logs/%x_%j.out
#SBATCH --error=/scratch/users/scdobbs/grl_review/R12b/logs/%x_%j.err
# R12b Part B.4: write SHA256SUMS in grl_review_archive/phase4/ (outside the scripts) and verify it; then the must-hold checks (counts only).
# Usage: sbatch --export=ALL,BUILD_LOG=<path> post_build_r12b.sbatch
set -o pipefail
type conda >/dev/null 2>&1 && { conda deactivate >/dev/null 2>&1 || true; conda deactivate >/dev/null 2>&1 || true; }
unset PYTHONPATH; module load python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312; source "${HOME}/bathy_diffusion_env/bin/activate"
export MPLBACKEND=Agg PYTHONUNBUFFERED=1; P4=/oak/stanford/groups/hilley/auv_ship_colocated_bathy/grl_review_archive/phase4
echo "=== $(date) $(hostname) job=${SLURM_JOB_ID} ==="
( cd "$P4" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 -P 4 -n 32 sha256sum | sort -k2 > SHA256SUMS && sha256sum -c SHA256SUMS --quiet && echo "SHA256SUMS VERIFIED: $(wc -l < SHA256SUMS) files  $(du -sh . | cut -f1)" ) || { echo "SHA256SUMS FAILED"; exit 2; }
cd /home/users/scdobbs/GAN_Bathymetry_Project/grl_reviews-and-prompt-reports/directives_and_reports/R12b/scripts && python -B check_build_r12b.py "${BUILD_LOG:-}"; rc=$?
echo "=== done $(date) rc=$rc ==="; exit $rc
