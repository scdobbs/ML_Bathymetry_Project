#!/bin/bash
#SBATCH --job-name=r04_preserve
#SBATCH -p serc
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/users/scdobbs/grl_review/R04/logs/%x_%j.out
#SBATCH --error=/scratch/users/scdobbs/grl_review/R04/logs/%x_%j.err
# R04 §6.1 — copy R02 / R03 originals from scratch to OAK grl_review_archive/, sha256 computed at the source and
# verified at the destination. Additive only (never deletes); files already present with the same size/mtime are
# skipped by rsync. 8 parallel rsync streams (OAK is metadata-bound on many small files) and parallel verification.
# Rev 3 (rsync 3.1.2 has no --target-directory; dest passed as $0 to sh -c). Rev 2 replaced the single-stream version (job 46232525, cancelled after 8 GB / 29 GB in 20 min).
set -euo pipefail
ARCH=/oak/stanford/groups/hilley/auv_ship_colocated_bathy/grl_review_archive
SRC2=/scratch/users/scdobbs/grl_review/R02; SRC3=/scratch/users/scdobbs/grl_review/R03
echo "=== $(date) $(hostname) ==="
copy_one () {   # copy_one <src_dir> <dst_dir>
  local src=$1 dst=$2
  if [ -f "$dst/SHA256SUMS" ]; then echo "SKIP (already verified): $dst"; return 0; fi
  mkdir -p "$dst"; local t0=$(date +%s); local n=$(cd "$src" && find . -type f | wc -l)
  if [ -f "$dst.SHA256SUMS.tmp" ] && [ "$(wc -l < "$dst.SHA256SUMS.tmp")" -eq "$n" ]; then echo "reusing source sums ($n files)"; else
    ( cd "$src" && find . -type f -print0 | sort -z | xargs -0 -P 8 -n 32 sha256sum ) | sort -k2 > "$dst.SHA256SUMS.tmp"; fi
  echo "source sums: $n files $(( $(date +%s) - t0 )) s"
  ( cd "$src" && find . -type f -print0 | sort -z | xargs -0 -P 8 -n 400 sh -c 'rsync -ltR --chmod=Dg+s "$@" "$0"' "$dst/" ) 2>&1 | tail -3
  # second, single pass to catch anything a parallel stream missed
  rsync -rlt --chmod=Dg+s "$src/" "$dst/"
  echo "copied: $(( $(date +%s) - t0 )) s"
  mv "$dst.SHA256SUMS.tmp" "$dst/SHA256SUMS"
  ( cd "$dst" && split -n l/8 -d SHA256SUMS .sums_part_ && for p in .sums_part_0*; do sha256sum -c "$p" --quiet & done; wait; rm -f .sums_part_0* ) \
    && echo "VERIFIED $(wc -l < "$dst/SHA256SUMS") files  $(( $(date +%s) - t0 )) s  $dst"
}
copy_one $SRC2/patches_clean           $ARCH/R02/patches_clean
copy_one $SRC2/mosaics                 $ARCH/R02/mosaics
copy_one $SRC2/het_clean_r02           $ARCH/R02/het_clean_r02
copy_one $SRC3/patches_clean_expanded  $ARCH/R03/patches_clean_expanded
copy_one $SRC3/s3                      $ARCH/R03/s3
copy_one $SRC3/s4                      $ARCH/R03/s4
copy_one $SRC3/s6/pilot/raw            $ARCH/R03/s6/pilot/raw
for d in $ARCH/R02/patches_clean $ARCH/R02/mosaics $ARCH/R02/het_clean_r02 $ARCH/R03/patches_clean_expanded $ARCH/R03/s3 $ARCH/R03/s4 $ARCH/R03/s6/pilot/raw; do echo "$(wc -l < $d/SHA256SUMS) files  $d"; done
echo "=== done $(date) ==="
