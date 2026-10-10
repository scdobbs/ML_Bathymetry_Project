#!/bin/bash
#SBATCH --job-name=r10_preserve
#SBATCH -p serc
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/users/scdobbs/grl_review/R10/logs/preserve_%j.out
#SBATCH --error=/scratch/users/scdobbs/grl_review/R10/logs/preserve_%j.err
# R10 §E — tables, figures (both sets), the key file, the scratch outputs (scores, metrics, predictions, checks, provenance,
# logs) and scripts to OAK grl_review_archive/R08b/, sha256 verified at the destination. Additive only.
set -euo pipefail
ARCH=/oak/stanford/groups/hilley/auv_ship_colocated_bathy/grl_review_archive/R10; SRC=/scratch/users/scdobbs/grl_review/R10
REP=/home/users/scdobbs/GAN_Bathymetry_Project/grl_reviews-and-prompt-reports/directives_and_reports/R10
echo "=== $(date) $(hostname) ==="
copy_one () {
  local src=$1 dst=$2
  if [ -f "$dst/SHA256SUMS" ]; then echo "SKIP (already verified): $dst"; return 0; fi
  mkdir -p "$dst"; local t0=$(date +%s)
  ( cd "$src" && find . -type f -print0 | sort -z | xargs -0 -P 4 -n 32 sha256sum ) | sort -k2 > "$dst.SHA256SUMS.tmp"
  rsync -rlt --chmod=Dg+s "$src/" "$dst/"; mv "$dst.SHA256SUMS.tmp" "$dst/SHA256SUMS"
  ( cd "$dst" && sha256sum -c SHA256SUMS --quiet ) && echo "VERIFIED $(wc -l < "$dst/SHA256SUMS") files  $(du -sh "$dst" | cut -f1)  $(( $(date +%s) - t0 )) s  $dst"
}
for d in rehearsal refusal_tests logsearch logs; do [ -d "$SRC/$d" ] && copy_one "$SRC/$d" "$ARCH/$d"; done
copy_one "$REP/tables" "$ARCH/tables"; copy_one "$REP/scripts" "$ARCH/scripts"; copy_one "$REP/../phase4/scripts" "$ARCH/phase4_scripts"
du -sh $ARCH/*; echo "=== done $(date) ==="
