#!/bin/bash
# R23 figure build. Run inside a Slurm allocation: srun -p dev -c 4 --mem=24G -t 0:40:00 bash run_r23.sh [fig1 fig2 fig3]
set -uo pipefail
module purge >/dev/null 2>&1; module load python/3.12.1 openblas/0.3.28 system poppler/0.47.0
source "$HOME/bathy_diffusion_env/bin/activate"
export MPLBACKEND=Agg OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MPLCONFIGDIR=/scratch/users/scdobbs/tmp_r23/mplcfg PYTHONUNBUFFERED=1
cd /home/users/scdobbs/GAN_Bathymetry_Project/analysis/R23/figures
echo "=== $(date) $(hostname) job=${SLURM_JOB_ID:-none} python=$(python -V 2>&1) matplotlib=$(python -c 'import matplotlib;print(matplotlib.__version__)') cartopy=$(python -c 'import cartopy;print(cartopy.__version__)') cmcrameri=$(python -c 'import cmcrameri;print(cmcrameri.__version__)') pdffonts=$(which pdffonts)"
rc_all=0
for f in ${@:-fig1 fig2 fig3}; do echo "=== $f"; python "$f.py"; rc=$?; echo "=== $f rc=$rc"; [ $rc -ne 0 ] && rc_all=$rc; done
exit $rc_all
