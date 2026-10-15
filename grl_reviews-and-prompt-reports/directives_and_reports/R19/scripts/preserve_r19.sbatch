#!/bin/bash
#SBATCH --job-name=r19_preserve
#SBATCH -p serc
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/users/scdobbs/grl_review/R19/logs/%x_%j.out
#SBATCH --error=/scratch/users/scdobbs/grl_review/R19/logs/%x_%j.err
# R19: (1) re-verify the existing SHA256SUMS of grl_review_archive/R18/ and phase5/ (no rewrite); (2) copy R19/ and phase5/DEVIATIONS_phase5.md to
# grl_review_archive/R19/ with SHA256SUMS, and verify.
set -o pipefail
A=/oak/stanford/groups/hilley/auv_ship_colocated_bathy/grl_review_archive; DST=$A/R19; DR=/home/users/scdobbs/GAN_Bathymetry_Project/grl_reviews-and-prompt-reports/directives_and_reports
echo "=== $(date) $(hostname) job=${SLURM_JOB_ID} ==="
for d in R18 phase5; do ( cd "$A/$d" && sha256sum -c SHA256SUMS --quiet && echo "$d SHA256SUMS RE-VERIFIED: $(wc -l < SHA256SUMS) files" ) || { echo "$d SHA256SUMS FAILED"; exit 2; }; done
mkdir -p "$DST" && cp -a "$DR/R19/." "$DST/" && cp -a "$DR/phase5/DEVIATIONS_phase5.md" "$DR/R13/DEVIATION_phase4_04.md" "$DST/"
( cd "$DST" && find . -type f ! -name 'SHA256SUMS*' -print0 | sort -z | xargs -0 sha256sum | sort -k2 > SHA256SUMS.tmp && mv SHA256SUMS.tmp SHA256SUMS && sha256sum -c SHA256SUMS --quiet && echo "R19 SHA256SUMS VERIFIED: $(wc -l < SHA256SUMS) files" ) || { echo "R19 SHA256SUMS FAILED"; exit 3; }
echo "=== done $(date) rc=0 ==="
