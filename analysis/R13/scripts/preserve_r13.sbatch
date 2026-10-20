#!/bin/bash
#SBATCH --job-name=r13_preserve
#SBATCH -p serc
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/users/scdobbs/grl_review/R13/logs/preserve_%j.out
#SBATCH --error=/scratch/users/scdobbs/grl_review/R13/logs/preserve_%j.err
# R13 §7 — scripts, tables, logs and the report to OAK grl_review_archive/R13/ with SHA256SUMS verified at the destination. Additive only.
# Phase 4 outputs (predictions, score_inputs, scores) stay in grl_review_archive/phase4/ with their SHA256SUMS.
set -euo pipefail
ARCH=/oak/stanford/groups/hilley/auv_ship_colocated_bathy/grl_review_archive/R13; SRC=/scratch/users/scdobbs/grl_review/R13
REP=/home/users/scdobbs/GAN_Bathymetry_Project/grl_reviews-and-prompt-reports/directives_and_reports/R13
echo "=== $(date) $(hostname) ==="
copy_one () {
  local src=$1 dst=$2
  if [ -f "$dst/SHA256SUMS" ]; then echo "SKIP (already verified): $dst"; return 0; fi
  mkdir -p "$dst"; ( cd "$src" && find . -type f -print0 | sort -z | xargs -0 -n 32 sha256sum ) | sort -k2 > "$dst.SHA256SUMS.tmp"
  rsync -rlt --chmod=Dg+s "$src/" "$dst/"; mv "$dst.SHA256SUMS.tmp" "$dst/SHA256SUMS"
  ( cd "$dst" && sha256sum -c SHA256SUMS --quiet ) && echo "VERIFIED $(wc -l < "$dst/SHA256SUMS") files  $dst"
}
copy_one "$SRC/logs" "$ARCH/logs"; copy_one "$REP/tables" "$ARCH/tables"; copy_one "$REP/scripts" "$ARCH/scripts"
mkdir -p "$ARCH/documents"; for f in REPORT_R13_2026-10-12.md phase4_calls.csv; do [ -f "$REP/$f" ] && cp "$REP/$f" "$ARCH/documents/"; done
( cd "$ARCH/documents" && sha256sum * | grep -v SHA256SUMS > SHA256SUMS && sha256sum -c SHA256SUMS --quiet && echo "VERIFIED documents" )
echo "=== done $(date) ==="
