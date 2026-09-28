#!/bin/bash
#SBATCH --job-name=r01_restore
#SBATCH -p serc
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/users/scdobbs/grl_review/R01/logs/%x_%j.out
#SBATCH --error=/scratch/users/scdobbs/grl_review/R01/logs/%x_%j.err
set -u
TAR=/oak/stanford/groups/hilley/auv_ship_colocated_bathy/patches_native_phase2_frozen_2026-07-01.tar
DEST=/scratch/users/scdobbs/grl_review/R01/patches_restored
ORIG=/scratch/groups/hilley/bathy_diffusion_2026/patches_native_phase2
echo "start $(date) $(hostname)"
tar -xf "$TAR" -C "$DEST" && echo "extract ok $(date)"
R=$DEST/patches_native_phase2
echo "total files: $(find $R -type f | wc -l)"
for s in train val test; do echo "$s: $(ls $R/$s | wc -l)"; done
# sha256 on 200 random restored files vs scratch originals where they still exist
python3 - <<PY
import os, random, hashlib, glob
R="$R"; O="$ORIG"
files=sorted(glob.glob(os.path.join(R,"*","*.npz")))
random.seed(20260928); sample=random.sample(files,200)
def h(p):
    m=hashlib.sha256()
    with open(p,'rb') as f:
        for c in iter(lambda: f.read(1<<20), b''): m.update(c)
    return m.hexdigest()
match=mismatch=missing=0
for p in sample:
    o=os.path.join(O, os.path.relpath(p,R))
    if not os.path.exists(o): missing+=1; continue
    if h(p)==h(o): match+=1
    else: mismatch+=1; print("MISMATCH", p)
print(f"sha256 sample: match={match} mismatch={mismatch} original_missing={missing} of 200")
PY
echo "end $(date)"
