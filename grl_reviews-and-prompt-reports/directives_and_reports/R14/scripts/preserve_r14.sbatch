#!/bin/bash
#SBATCH --job-name=r14_preserve
#SBATCH -p serc
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/users/scdobbs/grl_review/R14/logs/preserve_%j.out
#SBATCH --error=/scratch/users/scdobbs/grl_review/R14/logs/preserve_%j.err
# R14 §7 — scripts, tables, logs, the run records (coarse_record.json + the coarsened-grid hash list, predictors / targets / gbt records,
# the fold models, scores) and the report to OAK grl_review_archive/R14/ with SHA256SUMS verified at the destination. Additive only.
# The coarsened rasters themselves stay on scratch (their sha256 list is preserved in records/SHA256SUMS_coarse.txt).
set -euo pipefail
ARCH=/oak/stanford/groups/hilley/auv_ship_colocated_bathy/grl_review_archive/R14; SRC=/scratch/users/scdobbs/grl_review/R14
REP=/home/users/scdobbs/GAN_Bathymetry_Project/grl_reviews-and-prompt-reports/directives_and_reports/R14
echo "=== $(date) $(hostname) ==="
copy_one () {
  local src=$1 dst=$2
  if [ -f "$dst/SHA256SUMS" ]; then echo "SKIP (already verified): $dst"; return 0; fi
  mkdir -p "$dst"; ( cd "$src" && find . -type f -print0 | sort -z | xargs -0 -n 32 sha256sum ) | sort -k2 > "$dst.SHA256SUMS.tmp"
  rsync -rlt --chmod=Dg+s "$src/" "$dst/"; mv "$dst.SHA256SUMS.tmp" "$dst/SHA256SUMS"
  ( cd "$dst" && sha256sum -c SHA256SUMS --quiet ) && echo "VERIFIED $(wc -l < "$dst/SHA256SUMS") files  $dst"
}
STAGE=$SRC/stage_records; rm -rf "$STAGE"; mkdir -p "$STAGE/coarse" "$STAGE/predictors" "$STAGE/targets"
cp "$SRC/coarse/coarse_record.json" "$SRC/coarse/SHA256SUMS_coarse.txt" "$SRC"/coarse/harness_*.parquet "$STAGE/coarse/"
for d in "$SRC"/coarse/*/; do p=$(basename "$d"); for l in "$d"/*/; do [ -f "$l/record.json" ] && { mkdir -p "$STAGE/coarse/$p/$(basename "$l")"; cp "$l/record.json" "$STAGE/coarse/$p/$(basename "$l")/"; }; done; done
for d in "$SRC"/predictors/*/; do p=$(basename "$d"); mkdir -p "$STAGE/predictors/$p"; cp "$d"/record.json "$STAGE/predictors/$p/"; done; cp "$SRC"/predictors/predictors_record*.json "$STAGE/predictors/"
for d in "$SRC"/targets/*/; do p=$(basename "$d"); mkdir -p "$STAGE/targets/$p"; cp "$d"/record.json "$STAGE/targets/$p/"; done; cp "$SRC"/targets/targets_record*.json "$STAGE/targets/"
copy_one "$STAGE" "$ARCH/records"; copy_one "$SRC/gbt" "$ARCH/gbt"; copy_one "$SRC/scores" "$ARCH/scores"; copy_one "$SRC/provenance" "$ARCH/provenance"
copy_one "$SRC/logs" "$ARCH/logs"; copy_one "$REP/tables" "$ARCH/tables"; copy_one "$REP/scripts" "$ARCH/scripts"
mkdir -p "$ARCH/documents"; for f in REPORT_R14_2026-10-12.md; do [ -f "$REP/$f" ] && cp "$REP/$f" "$ARCH/documents/"; done
( cd "$ARCH/documents" && sha256sum * | grep -v SHA256SUMS > SHA256SUMS && sha256sum -c SHA256SUMS --quiet && echo "VERIFIED documents" )
echo "=== done $(date) ==="
