#!/bin/bash
# R10 rehearsal chain after the three-pair build: A2 compare -> A3 figures -> A4 predict (EW0207) -> A3/A4 checks -> inputs (with EW0207
# predictions) -> A6.1 scoring at D1 and at 125 m -> A6.5 cells. Each step stops the chain on failure.
set -euo pipefail; cd /home/users/scdobbs/GAN_Bathymetry_Project/grl_reviews-and-prompt-reports/directives_and_reports/phase4/scripts; R10=../../R10/scripts
P3='KM0923__MGDS_31429 AR26__MGDS_31838 EW0207__MGDS_32556'; ALL=$(python -c "import sys; sys.path.insert(0,'.'); import phase4_common as P4; print(' '.join(P4.ADMITTED['rehearsal']))")
python $R10/rehearsal_compare_r10.py "$P3"
python phase4_figures.py --mode rehearsal --pairs "$P3" --workers ${SLURM_CPUS_PER_TASK:-4}
python phase4_predict.py --mode rehearsal --pairs EW0207__MGDS_32556 --workers ${SLURM_CPUS_PER_TASK:-4}
python $R10/rehearsal_checks_r10.py "$P3"
python $R10/rehearsal_inputs_r10.py
SI=/scratch/users/scdobbs/grl_review/R10/rehearsal/score_inputs
python phase4_score.py --mode rehearsal --pairs "$ALL" --inputs $SI/inputs_D1.parquet --calls $SI/calls_rehearsal.csv --stats $SI/stats_D1.json --bmin ${BMIN:-8} --tag D1
python phase4_score.py --mode rehearsal --pairs "$ALL" --inputs $SI/inputs_D125.parquet --calls $SI/calls_rehearsal.csv --bmin ${BMIN:-8} --tag D125
python $R10/cells_r10.py
echo CHAIN_DONE
