#!/bin/bash
#SBATCH --job-name=r17_preserve
#SBATCH -p serc
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/users/scdobbs/grl_review/R17/logs/%x_%j.out
#SBATCH --error=/scratch/users/scdobbs/grl_review/R17/logs/%x_%j.err
# R17 §2.6: (1) SHA256SUMS of grl_review_archive/phase5/ (build outputs; written outside the scripts) and verify; (2) copy R17/ (report, scripts,
# tables, the delivered files) and the job logs to grl_review_archive/R17/ (logs under logs/: the build log prints amplitudes; Steve must not open
# it before his calls are committed) with SHA256SUMS, and verify.
set -o pipefail
A=/oak/stanford/groups/hilley/auv_ship_colocated_bathy/grl_review_archive; P5=$A/phase5; DST=$A/R17
SRC=/home/users/scdobbs/GAN_Bathymetry_Project/grl_reviews-and-prompt-reports/directives_and_reports/R17; L=/scratch/users/scdobbs/grl_review/R17/logs
echo "=== $(date) $(hostname) job=${SLURM_JOB_ID} ==="
( cd "$P5" && find . -type f ! -name 'SHA256SUMS*' -print0 | sort -z | xargs -0 sha256sum | sort -k2 > SHA256SUMS.tmp && mv SHA256SUMS.tmp SHA256SUMS && sha256sum -c SHA256SUMS --quiet && echo "phase5 SHA256SUMS VERIFIED: $(wc -l < SHA256SUMS) files" ) || { echo "phase5 SHA256SUMS FAILED"; exit 2; }
mkdir -p "$DST/logs" && cp -a "$SRC/." "$DST/" && cp -a $L/*.out $L/*.err "$DST/logs/" 2>/dev/null
( cd "$DST" && find . -type f ! -name 'SHA256SUMS*' -print0 | sort -z | xargs -0 sha256sum | sort -k2 > SHA256SUMS.tmp && mv SHA256SUMS.tmp SHA256SUMS && sha256sum -c SHA256SUMS --quiet && echo "R17 SHA256SUMS VERIFIED: $(wc -l < SHA256SUMS) files" ) || { echo "R17 SHA256SUMS FAILED"; exit 3; }
echo "=== done $(date) rc=0 ==="
