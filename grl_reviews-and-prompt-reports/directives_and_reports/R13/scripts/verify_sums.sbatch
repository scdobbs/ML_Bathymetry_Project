#!/bin/bash
#SBATCH --job-name=r13_sums
#SBATCH -p serc
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/users/scdobbs/grl_review/R13/logs/%x_%j.out
#SBATCH --error=/scratch/users/scdobbs/grl_review/R13/logs/%x_%j.err
# R13 §1.3: sha256sum -c on grl_review_archive/phase4/SHA256SUMS (count only reported)
P=/oak/stanford/groups/hilley/auv_ship_colocated_bathy/grl_review_archive/phase4
echo "=== $(date) $(hostname) ==="; cd $P && sha256sum -c SHA256SUMS --quiet && echo "VERIFIED $(wc -l < SHA256SUMS) files" || echo "SHA256SUMS FAILED"; echo "=== done $(date) ==="
