#!/bin/bash
#SBATCH --job-name=r15_preserve
#SBATCH -p serc
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/users/scdobbs/grl_review/R15/logs/preserve_%j.out
#SBATCH --error=/scratch/users/scdobbs/grl_review/R15/logs/preserve_%j.err
# R15 §5 — inputs (with the Goff grid, the age grid and the census), run outputs, provenance, logs, tables, scripts and the report
# to OAK grl_review_archive/R15/ with SHA256SUMS verified at the destination. Additive only (a verified destination is skipped).
set -euo pipefail
ARCH=/oak/stanford/groups/hilley/auv_ship_colocated_bathy/grl_review_archive/R15; SRC=/scratch/users/scdobbs/grl_review/R15
REP=/home/users/scdobbs/GAN_Bathymetry_Project/grl_reviews-and-prompt-reports/directives_and_reports/R15
echo "=== $(date) $(hostname) ==="
copy_one () {
  local src=$1 dst=$2
  if [ -f "$dst/SHA256SUMS" ]; then echo "SKIP (already verified): $dst"; return 0; fi
  mkdir -p "$dst"; ( cd "$src" && find . -type f -print0 | sort -z | xargs -0 -n 32 sha256sum ) | sort -k2 > "$dst.SHA256SUMS.tmp"
  rsync -rlt --chmod=Dg+s "$src/" "$dst/"; mv "$dst.SHA256SUMS.tmp" "$dst/SHA256SUMS"
  ( cd "$dst" && sha256sum -c SHA256SUMS --quiet ) && echo "VERIFIED $(wc -l < "$dst/SHA256SUMS") files  $dst"
}
copy_one "$REP/inputs" "$ARCH/inputs"; copy_one "$SRC/out" "$ARCH/out"; copy_one "$SRC/provenance" "$ARCH/provenance"
copy_one "$REP/tables" "$ARCH/tables"; copy_one "$REP/scripts" "$ARCH/scripts"
mkdir -p "$ARCH/documents"; cp "$REP/REPORT_R15_2026-10-13.md" "$REP/REPORT_R15_template.md" "$ARCH/documents/"
( cd "$ARCH/documents" && sha256sum REPORT_R15_* > SHA256SUMS && sha256sum -c SHA256SUMS --quiet && echo "VERIFIED documents" )
grep -q "^07421ead49aa5908ed16d74cc4d01f0839ff82b0ffad83d24ba5b8b73b73d2f6  ./goff/abyssal_hill_rms_prediction_v2.0.grd$" "$ARCH/inputs/SHA256SUMS" && echo "Goff grid sha256 matches the recorded value"
echo "=== done $(date) ==="
# logs last (this job's own log is still open; it is copied by a second call after the job ends)
