#!/bin/bash
#SBATCH --job-name=r16_preserve
#SBATCH -p serc
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/users/scdobbs/grl_review/R16/logs/preserve_%j.out
#SBATCH --error=/scratch/users/scdobbs/grl_review/R16/logs/preserve_%j.err
# R16 preservation — the rehearsal outputs (mock layout, dress rehearsal, the stopped first dress root, R10 reproduction builds, refusal tests, logs), tables,
# scripts, the Phase 5 scripts and the documents to OAK grl_review_archive/R16/, sha256 verified at the destination. Additive only. Development data only.
set -euo pipefail
ARCH=/oak/stanford/groups/hilley/auv_ship_colocated_bathy/grl_review_archive/R16; SRC=/scratch/users/scdobbs/grl_review/R16
REP=/home/users/scdobbs/GAN_Bathymetry_Project/grl_reviews-and-prompt-reports/directives_and_reports/R16
echo "=== $(date) $(hostname) ==="
copy_one () {
  local src=$1 dst=$2
  if [ -f "$dst/SHA256SUMS" ]; then echo "SKIP (already verified): $dst"; return 0; fi
  mkdir -p "$dst"; local t0=$(date +%s)
  ( cd "$src" && find . -type f ! -name 'SHA256SUMS*' -print0 | sort -z | xargs -0 -P 4 -n 32 sha256sum ) | sort -k2 > "$dst.SHA256SUMS.tmp"
  rsync -rlt --chmod=Dg+s,Fu+w "$src/" "$dst/"; mv "$dst.SHA256SUMS.tmp" "$dst/SHA256SUMS"
  ( cd "$dst" && sha256sum -c SHA256SUMS --quiet ) && echo "VERIFIED $(wc -l < "$dst/SHA256SUMS") files  $(du -sh "$dst" | cut -f1)  $(( $(date +%s) - t0 )) s  $dst"
}
for d in mock rehearsal rehearsal_stopped_47680313 rehearsal_repro refusal_tests logs; do [ -d "$SRC/$d" ] && copy_one "$SRC/$d" "$ARCH/$d"; done
copy_one "$REP/tables" "$ARCH/tables"; copy_one "$REP/scripts" "$ARCH/scripts"; copy_one "$REP/../phase5/scripts" "$ARCH/phase5_scripts"
mkdir -p "$ARCH/documents"; for f in REPORT_R16_2026-10-13.md PREREG_phase5_v3.md SIGNOFF_phase5_prereg_v3_DRAFT.md ../R13/DEVIATION_phase4_04.md; do [ -f "$REP/$f" ] && cp "$REP/$f" "$ARCH/documents/"; done
( cd "$ARCH/documents" && sha256sum $(ls | grep -v SHA256SUMS) > SHA256SUMS && sha256sum -c SHA256SUMS --quiet && echo "VERIFIED documents" )
du -sh $ARCH/*; echo "=== done $(date) ==="
