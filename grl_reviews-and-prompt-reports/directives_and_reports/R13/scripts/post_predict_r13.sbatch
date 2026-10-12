#!/bin/bash
#SBATCH --job-name=r13_post_predict
#SBATCH -p serc
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/users/scdobbs/grl_review/R13/logs/%x_%j.out
#SBATCH --error=/scratch/users/scdobbs/grl_review/R13/logs/%x_%j.err
# R13 Part C.3 + Part D prep: SHA256SUMS of grl_review_archive/phase4 rewritten (now with predictions/) and verified; then the score inputs
# and the joined calls assembled (assemble_inputs_r13.py, through phase4_common.start in phase4 mode); then SHA256SUMS again.
type conda >/dev/null 2>&1 && { conda deactivate >/dev/null 2>&1 || true; conda deactivate >/dev/null 2>&1 || true; }
unset PYTHONPATH; module load python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312; source "${HOME}/bathy_diffusion_env/bin/activate"
export MPLBACKEND=Agg PYTHONUNBUFFERED=1; P4=/oak/stanford/groups/hilley/auv_ship_colocated_bathy/grl_review_archive/phase4
sums () { ( cd "$P4" && find . -type f ! -name 'SHA256SUMS*' -print0 | sort -z | xargs -0 -P 4 -n 32 sha256sum | sort -k2 > /tmp/r13_sums_$$ && mv /tmp/r13_sums_$$ SHA256SUMS && sha256sum -c SHA256SUMS --quiet && echo "SHA256SUMS VERIFIED: $(wc -l < SHA256SUMS) files" ) || { echo "SHA256SUMS FAILED"; exit 2; }; }
echo "=== $(date) $(hostname) job=${SLURM_JOB_ID} ==="; sums
cd /home/users/scdobbs/GAN_Bathymetry_Project/grl_reviews-and-prompt-reports/directives_and_reports/R13/scripts && python -B assemble_inputs_r13.py || { echo "assemble FAILED"; exit 3; }
sums; echo "=== done $(date) rc=0 ==="
