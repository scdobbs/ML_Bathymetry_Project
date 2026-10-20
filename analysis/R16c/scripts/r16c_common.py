"""R16c common (rehearsal scaffolding only; no pinned script is edited by it). The R16 dress rehearsal (R16/scripts/r16_common.py)
re-run against acquisition freeze v2: the mock layout is rebuilt from the ACQ-R11b rehearsal outputs, with the ACQ-R11b rehearsal
record (step1_run_rehearsal_r11b.json, 11949c48...) and freeze_record_phase5_v2.json (f5a30b87...) byte-copied into the mock PHASE5/.
Re-points phase5_common.ROOT['rehearsal'] (and phase4_common.ROOT['rehearsal'], read by R11's compare_r11 at import) to the R16c
scratch root selected by env R16C_ROOT ('mock' = the dress rehearsal, 'repro' = the R10 reproduction builds).
Mock layout, name for name with the real Phase 5 roots, under /scratch/users/scdobbs/grl_review/R16c/mock/ (as R16):
  auv_ship_colocated_bathy/phase5/harmonized/<pair id>/     auv_ship_colocated_bathy/sealed_candidates_v3/<hr id>/ (empty)
  manifest/pairs_phase5_sealed.parquet                      reports_post_grl_review/PHASE5/{freeze_record_phase5_v2.json, step1_run_rehearsal_r11b.json}
Sites (R16's four): NA080__MGDS_31289, KM0923__MGDS_31429 (ACQ-R11b r11b outputs), FK151121__MGDS_22383 (ACQ-R11b harness output,
ruling S3) and KM0923__MOCK_R09repeat (row set eligible = False: the mock not-testable row). NR07-1__MGDS_31813 (the ACQ-R11b
positive control, not testable in the r11b run) is not a mock site: R16c reruns R16 Part C with R16's four sites so that the outputs
can be compared with R16's."""
import os, sys
DR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); P5_DIR = os.path.join(DR, "phase5", "scripts")
if P5_DIR not in sys.path: sys.path.insert(0, P5_DIR)
import phase5_common as P5
P4 = P5.P4
BASE = "/scratch/users/scdobbs/grl_review/R16c"; LOGS = f"{BASE}/logs"; MOCK = f"{BASE}/mock"
ROOTS = {"mock": f"{BASE}/rehearsal", "repro": f"{BASE}/rehearsal_repro"}; WHICH = os.environ.get("R16C_ROOT", "mock"); ROOT = ROOTS[WHICH]
P5.ROOT["rehearsal"] = ROOT; P4.ROOT["rehearsal"] = ROOT
MOCK_DATA = f"{MOCK}/auv_ship_colocated_bathy"; MOCK_HARM = f"{MOCK_DATA}/phase5/harmonized"; MOCK_CAND = f"{MOCK_DATA}/sealed_candidates_v3"
MOCK_SUP = f"{MOCK}/manifest/pairs_phase5_sealed.parquet"; MOCK_LEDGER = f"{MOCK}/CHANGE_LEDGER_mock.md"; MOCK_PHASE5 = f"{MOCK}/reports_post_grl_review/PHASE5"
MOCK_FREEZE = f"{MOCK_PHASE5}/freeze_record_phase5_v2.json"; MOCK_RECORD = f"{MOCK_PHASE5}/step1_run_rehearsal_r11b.json"
# ACQ-R11b rehearsal (freeze v2). phase5_common.ACQ_REHEARSAL_DIR still names ACQ-R09's records (directive R16c §1.4: only l.44 / l.47 / l.48 change).
ACQ_REH = f"{P5.ACQ_REPO}/reports_post_grl_review/ACQ-R11b/rehearsal"; REH_RECORD = f"{ACQ_REH}/step1_run_rehearsal_r11b.json"; REH_SUP = f"{ACQ_REH}/supplement_rehearsal_r11b.parquet"
REH_SUP_SHA = "d77a38e57229767a068e0dc0ba259ab602e7d5569dcc35ab494959e4ce390a95"; REH_RECORD_SHA8 = "11949c48"; REH_N_PAIRS = 5
REH_HARNESS = f"{ACQ_REH}/harness_fk151121_r11b.json"
# freeze v1 (superseded) and the ACQ-R09 rehearsal record that cites it: refusal cases (must-hold 3.1)
FREEZE_V1 = f"{P5.ACQ_RECORDS_DIR}/freeze_record_phase5.json"; FREEZE_V1_SHA = "6769583198da63d874472ea5a2ca62cb26c2f9f55003eb6e97838bfebab1d9d1"
R09_REH = f"{P5.ACQ_REPO}/reports_post_grl_review/ACQ-R09/rehearsal"; R09_SUP_SHA = "11d0868d73bb7db77c11c2a4aed620f12b4a41cb9bcfdb0929e345bab978bc88"
FREEZE_V2_SHA = "f5a30b87c05a90bb9155f37bbdf61de6d20fa55423d97662e59ff42b02809438"; STEP1_V2_SHA = "83af2875e53a28ef9b01c95a7b533ef7befd7a00625dfa6b0fe87178a2accc47"
R16_ROOT_B = "/scratch/users/scdobbs/grl_review/R16/rehearsal_b"   # the R16b dress rehearsal (reference outputs for must-hold 3.2)
MOCK_SITES = ("FK151121__MGDS_22383", "NA080__MGDS_31289", "KM0923__MGDS_31429", "KM0923__MOCK_R09repeat")
NOT_TESTABLE_MOCK = "KM0923__MOCK_R09repeat"; ELIGIBLE_MOCK = tuple(p for p in MOCK_SITES if p != NOT_TESTABLE_MOCK)
FAKE_ID = "FAKE__MGDS_00000"
REPRO_PAIRS = ["KM0923__MGDS_31429", "AR26__MGDS_31838", "EW0207__MGDS_32556"]
FILES = ("hr.tif", "lr.tif", "hr_valid.tif", "lr_valid.tif", "joint_valid.tif")
TABLES = f"{DR}/R16c/tables"
def mock_dir(pid): return f"{MOCK_HARM}/{pid}"
def mock_tokens(pids):
    """The real Phase 5 token pattern for the mock sites: pair id, ship cruise, HR id (the roots are already Phase 5 tokens)."""
    toks = set()
    for pid in pids: cruise, _, hr = pid.partition("__"); toks |= {pid, cruise, hr}
    return sorted(toks)
