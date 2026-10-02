#!/bin/bash
#SBATCH --job-name=r05_preserve
#SBATCH -p serc
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/users/scdobbs/grl_review/R05/logs/%x_%j.out
#SBATCH --error=/scratch/users/scdobbs/grl_review/R05/logs/%x_%j.err
# R05 §4 — copy the R05 tables, bar predictions and scripts to OAK grl_review_archive/R05/, sha256 verified. Additive only.
set -euo pipefail
ARCH=/oak/stanford/groups/hilley/auv_ship_colocated_bathy/grl_review_archive/R05; SRC=/scratch/users/scdobbs/grl_review/R05
SCR=/home/users/scdobbs/GAN_Bathymetry_Project/grl_reviews-and-prompt-reports/directives_and_reports/R05/scripts
echo "=== $(date) $(hostname) ==="
copy_one () {
  local src=$1 dst=$2
  if [ -f "$dst/SHA256SUMS" ]; then echo "SKIP (already verified): $dst"; return 0; fi
  mkdir -p "$dst"; local t0=$(date +%s)
  ( cd "$src" && find . -type f -print0 | sort -z | xargs -0 -P 4 -n 32 sha256sum ) | sort -k2 > "$dst.SHA256SUMS.tmp"
  rsync -rlt --chmod=Dg+s "$src/" "$dst/"; mv "$dst.SHA256SUMS.tmp" "$dst/SHA256SUMS"
  ( cd "$dst" && sha256sum -c SHA256SUMS --quiet ) && echo "VERIFIED $(wc -l < "$dst/SHA256SUMS") files  $(du -sh "$dst" | cut -f1)  $(( $(date +%s) - t0 )) s  $dst"
}
for d in tiles bar eval provenance; do [ -d "$SRC/$d" ] && copy_one "$SRC/$d" "$ARCH/$d"; done
mkdir -p "$SRC/root_files"; cp "$SRC"/*.json "$SRC/root_files/" 2>/dev/null || true; copy_one "$SRC/root_files" "$ARCH/root_files"
copy_one "$SCR" "$ARCH/scripts"
du -sh $ARCH/*; echo "=== done $(date) ==="
