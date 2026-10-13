"""R16 common (rehearsal scaffolding only; no pinned script is edited). Re-points phase5_common.ROOT['rehearsal'] (and
phase4_common.ROOT['rehearsal'], read by R11's compare_r11 at import) to the R16 scratch root selected by env R16_ROOT
('mock' = the dress rehearsal, 'repro' = the R10 reproduction builds) and holds the mock layout constants.
Mock layout (Part C1), name for name with the real Phase 5 roots, under /scratch/users/scdobbs/grl_review/R16/mock/:
  auv_ship_colocated_bathy/phase5/harmonized/<pair id>/     (real: OAK .../auv_ship_colocated_bathy/phase5/harmonized/<pair id>/)
  auv_ship_colocated_bathy/sealed_candidates_v3/<hr id>/    (real: $GROUP_SCRATCH .../auv_ship_colocated_bathy/sealed_candidates_v3/<hr id>/; empty here)
  manifest/pairs_phase5_sealed.parquet                      (real: <acquisition repo>/manifest/pairs_phase5_sealed.parquet)
  reports_post_grl_review/PHASE5/{freeze_record_phase5.json, step1_run_rehearsal_r09.json}   (byte copies of the ACQ-R09 files)
Sites: NA080__MGDS_31289 and KM0923__MGDS_31429 (ACQ-R09 rehearsal r09 outputs), FK151121__MGDS_22383 (the ACQ-R09 harness
output, ruling S3) and KM0923__MOCK_R09repeat (ACQ-R09 mock repeat site, row set eligible = False: the not-testable row)."""
import os, sys
DR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); P5_DIR = os.path.join(DR, "phase5", "scripts")
if P5_DIR not in sys.path: sys.path.insert(0, P5_DIR)
import phase5_common as P5
P4 = P5.P4
BASE = "/scratch/users/scdobbs/grl_review/R16"; LOGS = f"{BASE}/logs"; MOCK = f"{BASE}/mock"
ROOTS = {"mock": f"{BASE}/rehearsal", "repro": f"{BASE}/rehearsal_repro"}; WHICH = os.environ.get("R16_ROOT", "mock"); ROOT = ROOTS[WHICH]
P5.ROOT["rehearsal"] = ROOT; P4.ROOT["rehearsal"] = ROOT
MOCK_DATA = f"{MOCK}/auv_ship_colocated_bathy"; MOCK_HARM = f"{MOCK_DATA}/phase5/harmonized"; MOCK_CAND = f"{MOCK_DATA}/sealed_candidates_v3"
MOCK_SUP = f"{MOCK}/manifest/pairs_phase5_sealed.parquet"; MOCK_LEDGER = f"{MOCK}/CHANGE_LEDGER_mock.md"; MOCK_PHASE5 = f"{MOCK}/reports_post_grl_review/PHASE5"
MOCK_FREEZE = f"{MOCK_PHASE5}/freeze_record_phase5.json"; MOCK_RECORD = f"{MOCK_PHASE5}/step1_run_rehearsal_r09.json"
ACQ_REH = P5.ACQ_REHEARSAL_DIR; REH_RECORD = f"{ACQ_REH}/step1_run_rehearsal_r09.json"; REH_SUP = f"{ACQ_REH}/supplement_rehearsal_r09.parquet"
REH_SUP_SHA = "11d0868d73bb7db77c11c2a4aed620f12b4a41cb9bcfdb0929e345bab978bc88"; REH_RECORD_SHA8 = "8c17ac50"
MOCK_SITES = ("FK151121__MGDS_22383", "NA080__MGDS_31289", "KM0923__MGDS_31429", "KM0923__MOCK_R09repeat")
NOT_TESTABLE_MOCK = "KM0923__MOCK_R09repeat"; ELIGIBLE_MOCK = tuple(p for p in MOCK_SITES if p != NOT_TESTABLE_MOCK)
FAKE_ID = "FAKE__MGDS_00000"
REPRO_PAIRS = ["KM0923__MGDS_31429", "AR26__MGDS_31838", "EW0207__MGDS_32556"]
FILES = ("hr.tif", "lr.tif", "hr_valid.tif", "lr_valid.tif", "joint_valid.tif")
def mock_dir(pid): return f"{MOCK_HARM}/{pid}"
def mock_tokens(pids):
    """The real Phase 5 token pattern for the mock sites: pair id, ship cruise, HR id (the roots are already Phase 5 tokens)."""
    toks = set()
    for pid in pids: cruise, _, hr = pid.partition("__"); toks |= {pid, cruise, hr}
    return sorted(toks)
