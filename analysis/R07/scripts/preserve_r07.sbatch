#!/bin/bash
#SBATCH --job-name=r07_preserve
#SBATCH -p serc
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/users/scdobbs/grl_review/R07/preserve_%j.out
#SBATCH --error=/scratch/users/scdobbs/grl_review/R07/preserve_%j.err
# R07 §5.2 — both pretrained bodies + pretext heads (ckpt_final), pretraining logs / curves / QC figures, every fine-tuning
# fold's checkpoint / log / predictions, scores, the cache manifest (not the cache), the held-out cruise list, checks,
# provenance, job logs and scripts to OAK grl_review_archive/R07/, sha256 verified at the destination. Additive only.
set -euo pipefail
ARCH=/oak/stanford/groups/hilley/auv_ship_colocated_bathy/grl_review_archive/R07; SRC=/scratch/users/scdobbs/grl_review/R07
SCR=/home/users/scdobbs/GAN_Bathymetry_Project/grl_reviews-and-prompt-reports/directives_and_reports/R07/scripts
echo "=== $(date) $(hostname) ==="
copy_one () {
  local src=$1 dst=$2
  if [ -f "$dst/SHA256SUMS" ]; then echo "SKIP (already verified): $dst"; return 0; fi
  mkdir -p "$dst"; local t0=$(date +%s)
  ( cd "$src" && find . -type f -print0 | sort -z | xargs -0 -P 8 -n 32 sha256sum ) | sort -k2 > "$dst.SHA256SUMS.tmp"
  rsync -rlt --chmod=Dg+s "$src/" "$dst/"; mv "$dst.SHA256SUMS.tmp" "$dst/SHA256SUMS"
  ( cd "$dst" && sha256sum -c SHA256SUMS --quiet ) && echo "VERIFIED $(wc -l < "$dst/SHA256SUMS") files  $(du -sh "$dst" | cut -f1)  $(( $(date +%s) - t0 )) s  $dst"
}
mkdir -p "$SRC/root_files"; cp "$SRC"/*.json "$SRC/root_files/" 2>/dev/null || true
mkdir -p "$SRC/cache_record"; cp "$SRC"/cache/*.json "$SRC"/cache/*.parquet "$SRC"/cache/*.csv "$SRC"/cache/*.txt "$SRC/cache_record/" 2>/dev/null || true
for d in pretrain models scores checks provenance logs root_files cache_record; do [ -d "$SRC/$d" ] && copy_one "$SRC/$d" "$ARCH/$d"; done
copy_one "$SCR" "$ARCH/scripts"
du -sh $ARCH/*; echo "=== done $(date) ==="
