#!/bin/bash
#SBATCH --job-name=r04_preserve_r04
#SBATCH -p serc
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/users/scdobbs/grl_review/R04/logs/%x_%j.out
#SBATCH --error=/scratch/users/scdobbs/grl_review/R04/logs/%x_%j.err
# R04 §6.1 (second part): copy the R04 fields, tiles_v1, feature tables, prediction tables, checks and validation to OAK
# grl_review_archive/R04/, sha256 computed at the source and verified at the destination. Additive only.
set -euo pipefail
ARCH=/oak/stanford/groups/hilley/auv_ship_colocated_bathy/grl_review_archive/R04; SRC=/scratch/users/scdobbs/grl_review/R04
echo "=== $(date) $(hostname) ==="
copy_one () {
  local src=$1 dst=$2
  if [ -e "$dst" ]; then echo "SKIP (exists): $dst"; return 0; fi
  mkdir -p "$(dirname "$dst")"; local t0=$(date +%s)
  ( cd "$src" && find . -type f -print0 | sort -z | xargs -0 -P 8 -n 32 sha256sum ) | sort -k2 > "$dst.SHA256SUMS.tmp"
  rsync -rlt --chmod=Dg+s "$src/" "$dst/"; mv "$dst.SHA256SUMS.tmp" "$dst/SHA256SUMS"
  ( cd "$dst" && sha256sum -c SHA256SUMS --quiet ) && echo "VERIFIED $(wc -l < "$dst/SHA256SUMS") files  $(du -sh "$dst" | cut -f1)  $(( $(date +%s) - t0 )) s  $dst"
}
for d in fields tiles features bar checks validation provenance; do [ -d "$SRC/$d" ] && copy_one "$SRC/$d" "$ARCH/$d"; done
du -sh $ARCH/*; echo "=== done $(date) ==="
