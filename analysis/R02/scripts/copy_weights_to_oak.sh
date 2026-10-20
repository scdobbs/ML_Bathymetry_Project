#!/bin/bash
# R02 §2.3: additive copy of het_clean_r02 weights to OAK (new directory only), sha256-verified. Run AFTER the report is written.
set -euo pipefail
SRC=/scratch/users/scdobbs/grl_review/R02/het_clean_r02
DST=/oak/stanford/groups/hilley/auv_ship_colocated_bathy/models/het_clean_r02
[ -e "$DST" ] && { echo "refusing: $DST already exists (additive write only)"; exit 1; }
mkdir -p "$DST"
cp -r "$SRC"/. "$DST"/
(cd "$SRC" && find . -type f | sort | xargs sha256sum) > "$DST/SHA256SUMS.src"
(cd "$DST" && sha256sum -c SHA256SUMS.src --quiet) && echo "sha256 verified: $(wc -l < "$DST/SHA256SUMS.src") files" && du -sh "$DST"
