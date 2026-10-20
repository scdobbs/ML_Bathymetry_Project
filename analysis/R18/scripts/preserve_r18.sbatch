#!/bin/bash
#SBATCH --job-name=r18_preserve
#SBATCH -p serc
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/users/scdobbs/grl_review/R18/logs/%x_%j.out
#SBATCH --error=/scratch/users/scdobbs/grl_review/R18/logs/%x_%j.err
# R18 §4.8: (1) refresh SHA256SUMS of grl_review_archive/phase5/ (predictions + scores added) and verify; (2) copy R18/ and its job logs to grl_review_archive/R18/ with SHA256SUMS, and verify.
set -o pipefail
A=/oak/stanford/groups/hilley/auv_ship_colocated_bathy/grl_review_archive; P5=$A/phase5; DST=$A/R18
SRC=/home/users/scdobbs/GAN_Bathymetry_Project/grl_reviews-and-prompt-reports/directives_and_reports/R18; L=/scratch/users/scdobbs/grl_review/R18/logs
echo "=== $(date) $(hostname) job=${SLURM_JOB_ID} ==="
( cd "$P5" && find . -type f ! -name 'SHA256SUMS*' -print0 | sort -z | xargs -0 sha256sum | sort -k2 > SHA256SUMS.tmp && mv SHA256SUMS.tmp SHA256SUMS && sha256sum -c SHA256SUMS --quiet && echo "phase5 SHA256SUMS VERIFIED: $(wc -l < SHA256SUMS) files" ) || { echo "phase5 SHA256SUMS FAILED"; exit 2; }
mkdir -p "$DST/logs" && cp -a "$SRC/." "$DST/" && cp -a $L/*.out $L/*.err "$DST/logs/" 2>/dev/null
( cd "$DST" && find . -type f ! -name 'SHA256SUMS*' -print0 | sort -z | xargs -0 sha256sum | sort -k2 > SHA256SUMS.tmp && mv SHA256SUMS.tmp SHA256SUMS && sha256sum -c SHA256SUMS --quiet && echo "R18 SHA256SUMS VERIFIED: $(wc -l < SHA256SUMS) files" ) || { echo "R18 SHA256SUMS FAILED"; exit 3; }
echo "=== done $(date) rc=0 ==="
