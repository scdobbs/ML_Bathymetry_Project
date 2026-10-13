"""R15 common (directive R15, descriptive site-level comparison with Goff's predicted abyssal hill RMS heights; CPU only;
reviewer request, not pre-registered; changes no Phase 4 result).
Imports phase4_common FIRST (installs the R04 / R07 / R09 guards on import), re-points phase4_common.ROOT['rehearsal'] to
the R15 scratch root, and start() runs phase4_common.start in rehearsal mode (R09 guard installed and self-tested 702 / 702,
rehearsal locks, pair admission of the 38 development pairs + EW0207) before anything else. Nothing sealed, lockbox or
second-test is sampled: the unit list is the 16 development units of manifest v2.1 plus EW0207 (lu16), separate.
Holds the R15 constants (inputs and their sha256, seeds, neighbourhood, classification rule) and the statistics helpers."""
from __future__ import annotations
import hashlib, json, os, sys
DR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))      # .../directives_and_reports
P4_DIR = os.path.join(DR, "phase4", "scripts")
for p in (P4_DIR, os.path.join(DR, "R05", "scripts"), os.path.join(DR, "R04", "scripts")):
    if p not in sys.path: sys.path.insert(0, p)
import phase4_common as P4
C4, R8, R9, RB = P4.C4, P4.R8, P4.R9, P4.R9.RB
REPO = P4.REPO
ROOT = "/scratch/users/scdobbs/grl_review/R15"; P4.ROOT["rehearsal"] = ROOT
OUT, LOGS, PROV = (f"{ROOT}/{d}" for d in ("out", "logs", "provenance"))
REP = f"{DR}/R15"; TABLES = f"{REP}/tables"; FIGURES = f"{REP}/figures"; INPUTS = f"{REP}/inputs"
OAK_ARCHIVE = R8.OAK_ARCHIVE; OAK_R15 = f"{OAK_ARCHIVE}/R15"

# ---- §1 inputs (sha256 recorded at receipt / download, 2026-10-13; every run re-hashes and refuses on a difference)
GOFF_GRID = f"{INPUTS}/goff/abyssal_hill_rms_prediction_v2.0.grd"
GOFF_GRID_SHA = "07421ead49aa5908ed16d74cc4d01f0839ff82b0ffad83d24ba5b8b73b73d2f6"
GOFF_SI = f"{INPUTS}/goff/grl60637-sup-0001-2020gl088162-si.docx"
GOFF_SI_SHA = "6089367cbc366d8b65eb06c8707449bcae637f8b3330f203ecd71a3ce12f3004"
GOFF_NODATA = -999.0                     # the grid's missing-value sentinel (it carries no NaN node; §1.1 facts)
AGE_GRID = f"{INPUTS}/proxies/age.2020.1.GTS2012.6m.nc"
AGE_GRID_SHA = "3abeed4f6e9bfe9332ac6b5dff4c4f19c8a6374528a185f87e677db8152ba39e"
AGE_GRID_URL = "https://www.earthbyte.org/webdav/ftp/earthbyte/agegrid/2020/Grids/age.2020.1.GTS2012.6m.nc"
AGE_README = f"{INPUTS}/proxies/README.txt"
AGE_README_SHA = "e872faa247d8f61ae5772d28145bd7ccac45ec60f600cf5fa1f7f915f631bb02"
AGE_README_URL = "https://www.earthbyte.org/webdav/ftp/earthbyte/agegrid/2020/Grids/README.txt"
KW = f"{INPUTS}/proxies/KWSMTSv01.txt"
KW_SHA = "91c93302c44463a424835aa4051b7b2a1ea04d6675d928ca8405b231ae7cea9a"
KW_URL = "https://www.soest.hawaii.edu/PT/SMTS/kwsmts/KWSMTSv01.txt"
INPUT_SHAS = {GOFF_GRID: GOFF_GRID_SHA, GOFF_SI: GOFF_SI_SHA, AGE_GRID: AGE_GRID_SHA, AGE_README: AGE_README_SHA, KW: KW_SHA}

# ---- R08 / R08b unit values (OAK-verified copies)
OAK_R08_LEVEL = f"{OAK_ARCHIVE}/R08/level_geometry"
OAK_R08B_INT = f"{OAK_ARCHIVE}/R08b/interior"
R08_RHO, R08_CI = 0.697, (0.305, 0.866)        # R08 §4 level_correlations.json, level_raw_vs_s_lr (must-hold reference)
MUST_HOLD_TOL = 0.01

# ---- §2 / §3 constants (stated before sampling)
SEED = 20261014; N_BOOT = 2000
NEIGH_DEG = 0.5                          # §2.2(b): nodes with |dlon| <= 0.5 and |dlat| <= 0.5 deg of the centroid (one 1 x 1 deg cell)
AGE_MIN_MYR = 1.0                        # §2.3
SMT_RADIUS_FACTOR = 2.0                  # §2.3: seamount centre within twice its radius of the unit footprint
# Kim & Wessel (2011) columns 4 / 5 are the major / minor axes of the basal ellipse (km); radius = semi-major axis = major / 2.
# Sensitivity (reported, not used for the class): radius = major axis.
SMT_SEARCH_DEG = 6.0                     # pre-filter on great-circle distance of seamount centre to unit centroid
UNITS16 = list(C4.UNITS); TEN = list(RB.TEN); SIX = list(RB.SIX); EW_UNIT = R8.EW_UNIT

def sha256_file(p): return P4.sha256_file(p)
def dump(o, p): os.makedirs(os.path.dirname(p), exist_ok=True); json.dump(o, open(p, "w"), indent=1, default=R8._default)

def check_inputs():
    out = {}
    for p, exp in INPUT_SHAS.items():
        h = sha256_file(p); out[os.path.relpath(p, REP)] = dict(sha256=h, expected=exp, ok=(h == exp))
        if h != exp: raise SystemExit(f"STOP (R15 §1): {p} sha256 {h} != recorded {exp}")
    return out

def oak_verify(d, rel):
    """d/rel must hash to the d/SHA256SUMS entry for ./rel; returns the path."""
    sums = {}
    for line in open(f"{d}/SHA256SUMS"):
        h, _, n = line.rstrip("\n").partition("  "); sums[n.lstrip("./")] = h
    h = sha256_file(f"{d}/{rel}")
    if h != sums[rel]: raise SystemExit(f"STOP: {d}/{rel} sha256 {h} != SHA256SUMS {sums[rel]}")
    return f"{d}/{rel}"

def start(script):
    """Guard first (R09, 702 / 702), rehearsal locks, pair admission (38 development pairs + EW0207); then the R15 input hashes."""
    for d in (ROOT, OUT, LOGS, PROV): os.makedirs(d, exist_ok=True)
    pairs = list(C4.DEV_PAIRS) + [R8.EW_PAIR]
    m, root, rec = P4.start(script, "rehearsal", pairs); assert root == ROOT, root
    rec_inputs = check_inputs(); dump(dict(script=script, inputs=rec_inputs, guard=f"{rec['guard_probes_blocked']}/{rec['guard_probes']}"), f"{PROV}/{script}_inputs.json")
    assert rec["guard_probes_blocked"] == rec["guard_probes"] == 702
    print(f"[r15_common.start] {script}: guard {rec['guard_probes_blocked']}/{rec['guard_probes']}; {len(rec_inputs)} R15 inputs verified", flush=True)
    return m, rec

# ---- statistics (Spearman with a bootstrap over units, R08 helpers with the R15 seed)
def spearman_boot(x, y): return R8.spearman_boot(x, y, seed=SEED, n_boot=N_BOOT)

def partial_boot(a, b, z):
    """Partial Spearman rho of a, b given z (R08 definition) with a 2,000-resample percentile bootstrap over units."""
    import numpy as np
    a, b, z = (np.asarray(v, float) for v in (a, b, z)); ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(z); a, b, z = a[ok], b[ok], z[ok]; n = len(a)
    val = R8.partial_spearman(a, b, z)
    rng = np.random.default_rng(SEED); bs = []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n); bs.append(R8.partial_spearman(a[i], b[i], z[i]))
    bs = np.array(bs, float)
    return dict(rho=float(val), ci95=[float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))], n=int(n), n_boot_nan=int(np.isnan(bs).sum()))
