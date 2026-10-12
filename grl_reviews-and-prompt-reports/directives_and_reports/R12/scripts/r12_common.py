"""R12 common: the R12 rehearsal root and the harness constants (directive R12 Part A). No Phase 4 script is edited: every R12
driver imports this module first, which re-points phase4_common.ROOT['rehearsal'] to the R12 scratch root (the Phase 4 scripts
read ROOT at run time), so the k = 32 rehearsal builds, predictions and provenance records land under R12 and never touch the
R11 rehearsal outputs. phase4 mode is unaffected (ROOT['phase4'] is not touched)."""
import os, sys
DR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))      # .../directives_and_reports
P4_DIR = os.path.join(DR, "phase4", "scripts")
if P4_DIR not in sys.path: sys.path.insert(0, P4_DIR)
import phase4_common as P4
ROOT = "/scratch/users/scdobbs/grl_review/R12/rehearsal"; LOGS = "/scratch/users/scdobbs/grl_review/R12/logs"
P4.ROOT["rehearsal"] = ROOT
HARNESS = f"{ROOT}/harness/harness_k32.parquet"
# §A2: two harness rows that keep the development data and change only lr_native_res_m and max_recoverable_k
HARNESS_ROWS = {"FK151121__MGDS_22383": dict(label="H1", lr_native_res_m=150.77, max_recoverable_k=32),   # SUM1004's sealed geometry (HR 1.0 m)
                "NA080__MGDS_31289": dict(label="H2", lr_native_res_m=124.62, max_recoverable_k=32)}      # RC2511's sealed geometry (HR 1.0 m)
PAIRS = list(HARNESS_ROWS)
EXPECTED_D1 = {"FK151121__MGDS_22383": 151, "NA080__MGDS_31289": 125}
F_HI_CLIP = 0.45                                                   # r04_common.band_edges: f_hi = min(k f_lr, 0.45)
FROZEN_FIELDS = f"{P4.C4.ROOT}/fields"                              # the unmodified R04 build of every development pair (F_hr.tif, joint_v1.tif)
