#!/bin/bash
#SBATCH --job-name=r07_stage
#SBATCH -p serc
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/users/scdobbs/grl_review/R07/logs/%x_%j.out
#SBATCH --error=/scratch/users/scdobbs/grl_review/R07/logs/%x_%j.err
# R07 §1.5 / §6: stage the pretraining corpus (255 per-cruise tars + meta, 19.6 GB) from OAK to $SCRATCH once, verify
# every file against the corpus SHA256SUMS at the destination (and the SHA256SUMS file itself against the directive's hash).
set -euo pipefail
SRC=/oak/stanford/groups/hilley/auv_ship_colocated_bathy/pretrain_v1; DST=/scratch/users/scdobbs/grl_review/R07/corpus
echo "=== $(date) $(hostname) ==="
rsync -rlt --info=progress2 "$SRC/" "$DST/"
cd "$DST"
echo "$(sha256sum SHA256SUMS | cut -d' ' -f1)  SHA256SUMS" > SHA256SUMS.self
grep -q "^bbcc6ff1e6e7ec9f7c840ce052b2b5c3932aaccf106c0fd617e4dc61ac72cb26  SHA256SUMS$" SHA256SUMS.self || { echo "SHA256SUMS hash differs from directive §0.2 — STOP"; exit 2; }
sha256sum -c SHA256SUMS --quiet && echo "VERIFIED $(wc -l < SHA256SUMS) files at $DST ($(du -sh . | cut -f1))" && touch "$DST/STAGED_OK"
echo "=== done $(date) ==="
