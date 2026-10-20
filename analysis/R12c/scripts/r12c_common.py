"""R12c common (rehearsal scaffolding only; no pinned script is edited). Re-points phase4_common.ROOT['rehearsal'] to the R12c scratch root
(env R12C_ROOT selects the root: 'mock' = the dress rehearsal, 'repro' = the R10 reproduction builds) and holds the mock layout constants.
Mock layout (Part C1): development pairs COPIED into a sealed-style layout under /scratch/users/scdobbs/grl_review/R12c/mock/
  harmonized_confirmatory/nu98/FK151121__MGDS_22383/   (k defined, products usable -> 'used';   like SUM1004)
  harmonized_confirmatory/nu97/TN399__MGDS_33068/      (k defined, flag set False in the mock row -> 'masked'; like RC2511)
  harmonized/KM0923__MGDS_31429/                        (lockbox-style; like AT42-03 / TN159)
with the same file names as the real confirmatory pairs (hr.tif, lr.tif, hr_valid.tif, lr_valid.tif, joint_valid.tif, ship_products_v2_1/).
The mock supplement rows (real supplement schema = manifest v2.1 columns) carry ABSOLUTE paths (resolve_pair_paths joins DATA with an
absolute path and gets the absolute path)."""
import os, sys
DR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); P4_DIR = os.path.join(DR, "phase4", "scripts")
if P4_DIR not in sys.path: sys.path.insert(0, P4_DIR)
import phase4_common as P4
BASE = "/scratch/users/scdobbs/grl_review/R12c"; LOGS = f"{BASE}/logs"; MOCK = f"{BASE}/mock"
ROOTS = {"mock": f"{BASE}/rehearsal", "repro": f"{BASE}/rehearsal_repro"}; WHICH = os.environ.get("R12C_ROOT", "mock"); ROOT = ROOTS[WHICH]
P4.ROOT["rehearsal"] = ROOT
MOCK_ROWS = {"FK151121__MGDS_22383": dict(style="confirmatory", unit="nu98", rel="harmonized_confirmatory/nu98/FK151121__MGDS_22383", usable=True),
             "TN399__MGDS_33068": dict(style="confirmatory", unit="nu97", rel="harmonized_confirmatory/nu97/TN399__MGDS_33068", usable=False),
             "KM0923__MGDS_31429": dict(style="lockbox", unit="lu07", rel="harmonized/KM0923__MGDS_31429", usable=True)}
MOCK_PAIRS = list(MOCK_ROWS); MOCK_SUP = f"{MOCK}/mock_supplement.parquet"; FAKE_PROBE = f"{MOCK}/harmonized_confirmatory/nu99/FAKE__MGDS_00000/hr.tif"
REPRO_PAIRS = ["KM0923__MGDS_31429", "AR26__MGDS_31838", "EW0207__MGDS_32556"]       # the three R10 rehearsal pairs (must-hold 4)
FILES = ("hr.tif", "lr.tif", "hr_valid.tif", "lr_valid.tif", "joint_valid.tif")
def mock_dir(pid): return f"{MOCK}/{MOCK_ROWS[pid]['rel']}"
def mock_tokens(pids):
    """What the harness treats as sealed for the mock pairs (the real sealed_tokens pattern: pair id, cruise, MGDS, every path of the row)."""
    toks = set()
    for pid in pids:
        toks.add(pid); cruise, _, mgds = pid.partition("__"); toks.add(cruise); toks.add(mgds); d = mock_dir(pid); toks.add(d)
        for f in FILES: toks.add(f"{d}/{f}")
    return sorted(toks)
