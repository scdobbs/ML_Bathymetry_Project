"""R14 common (directive R14, exploratory ship-coarsening test on development pairs; CPU only).
Imports phase4_common FIRST (which installs the R04 / R07 / R09 guards on import and re-exports the frozen common modules),
re-points phase4_common.ROOT['rehearsal'] to the R14 scratch root (so every provenance record written by phase4_common.start
lands under R14, as the R12 drivers did), and holds the R14 constants: the five coarsening levels, the pair set (the 15 pairs
of the ten real-relief units; the 14 C1-set pairs of R10 §6 are derived at run time from the R10b per-pair table), the output
roots, the harness-row builder (manifest v2.1 row with ONLY lr_native_res_m, harmonized_path_lr and lr_valid changed) and the
OAK-verified loaders of the R10b scoring inputs. No Phase 4 script, frozen artifact or pinned file is edited.
Nothing sealed is opened: start() runs phase4_common.start in rehearsal mode (R09 guard installed and self-tested 702 / 702,
rehearsal locks, pair admission) before anything else."""
from __future__ import annotations
import hashlib, json, math, os, sys
DR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))      # .../directives_and_reports
P4_DIR = os.path.join(DR, "phase4", "scripts")
for p in (P4_DIR, os.path.join(DR, "R05", "scripts"), os.path.join(DR, "R04", "scripts")):
    if p not in sys.path: sys.path.insert(0, p)
import phase4_common as P4
import phase4_score as PS
C4, R8, R9, RB = P4.C4, P4.R8, P4.R9, P4.R9.RB
REPO = P4.REPO
ROOT = "/scratch/users/scdobbs/grl_review/R14"; P4.ROOT["rehearsal"] = ROOT
COARSE, PRED, TGT, GBT, SCORES, LOGS, PROV = (f"{ROOT}/{d}" for d in ("coarse", "predictors", "targets", "gbt", "scores", "logs", "provenance"))
REP = f"{DR}/R14"; TABLES = f"{REP}/tables"
OAK10B = f"{R8.OAK_ARCHIVE}/R10b/rehearsal"
LEVELS = (8.0, 4.0, 3.0, 2.1, 1.7)                                # target ship cells across the 256-px window (directive §3.1)
WIN = 256; S_FLOOR = 0.5; COVER_MIN = 0.5                           # frozen window, frozen s_lr floor; §3.2 coarse-cell validity rule
BOOT_SEED, POOL_SEED, N_BOOT, B_MIN = P4.BOOT_SEED, P4.POOL_SEED, P4.N_BOOT, P4.B_MIN_DEFAULT    # 20261009, 20261013, 2000, 8
CFG18 = 18                                                         # the all-16 Phase 4 artifact's configuration index (R09 record)
PAIRS15 = [p for u in RB.TEN for p in C4.UNITS[u]]                 # the 15 pairs of the ten real-relief units (R10 §6)

def tag(c): return f"c{c}"                                        # directory tag of a level: c8.0, c4.0, c3.0, c2.1, c1.7
def sha256_file(p): return P4.sha256_file(p)
def sha256_text(s): return hashlib.sha256(s.encode()).hexdigest()
def dump(o, p): os.makedirs(os.path.dirname(p), exist_ok=True); json.dump(o, open(p, "w"), indent=1, default=_default)
def _default(v):
    import numpy as np
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return float(v)
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, (np.bool_,)): return bool(v)
    return str(v)

def start(script, pairs=None):
    """Guard first (R09, 702 / 702), rehearsal locks (frozen artifacts / code against v1), pair admission; provenance under R14."""
    for d in (ROOT, LOGS, PROV): os.makedirs(d, exist_ok=True)
    m, root, rec = P4.start(script, "rehearsal", list(pairs or PAIRS15)); assert root == ROOT, root
    return m, rec

def load_r10b():
    """The R10b scoring inputs and per-pair table (OAK-verified), as R11 Part B used them."""
    import pandas as pd
    for rel in ("score_inputs/inputs_D1.parquet", "scores/per_pair_D1.csv", "score_inputs/calls_rehearsal.csv", "score_inputs/stats_D1.json"):
        R8.oak_verify("R10b/rehearsal", rel, f"{OAK10B}/{rel}")
    SI = pd.read_parquet(f"{OAK10B}/score_inputs/inputs_D1.parquet"); PP = pd.read_csv(f"{OAK10B}/scores/per_pair_D1.csv").set_index("pair")
    calls = pd.read_csv(f"{OAK10B}/score_inputs/calls_rehearsal.csv").fillna(""); cmap = {r["pair"]: r for r in calls.to_dict("records")}
    stats = json.load(open(f"{OAK10B}/score_inputs/stats_D1.json"))
    return SI, PP, cmap, stats

def c1_pairs(PP):
    """The 14 pairs of the ten real-relief units in R10's C1 set (status 'in C1' or 'in C1 (flagged)')."""
    return [p for p in PAIRS15 if str(PP.loc[p, "status"]).startswith("in C1")]

def pair_geometry(tiles, pid):
    """hr_res (HR raster spacing, as the frozen tile table carries it), lr_native (manifest), k, native cells across, D1."""
    t = tiles[tiles.pair == pid]; hr = float(t.hr_res.iloc[0]); lr = float(t.lr_native.iloc[0]); k = int(t.k.iloc[0])
    return dict(hr_res=hr, lr_native=lr, k=k, cells_native=WIN * hr / lr, D1=int(math.ceil(lr / hr)), n_tiles=int(len(t)))

def level_plan(hr_res, lr_native):
    """{level: lr_target} for the levels not skipped (§3.1: skip when lr_target <= lr_native), plus the skipped ones."""
    plan, skipped = {}, {}
    for c in LEVELS:
        lt = WIN * hr_res / c
        (plan if lt > lr_native else skipped)[c] = lt
    return plan, skipped

def coarse_dir(pid, c): return f"{COARSE}/{pid}/{tag(c)}"

def harness_row(m, pid, lr_target, cdir):
    """Manifest v2.1 row of pid with ONLY lr_native_res_m -> lr_target, harmonized_path_lr -> <cdir>/lr.tif and lr_valid ->
    <cdir>/lr_valid.tif (absolute paths; resolve_pair_paths joins them to the data root, which os.path.join ignores for absolute
    paths, and r04_common.products_dir derives <cdir>/ship_products_v2_1 from harmonized_path_lr). Everything else frozen."""
    row = C4.pair_row(m, pid).copy()
    row["lr_native_res_m"] = float(lr_target); row["harmonized_path_lr"] = f"{cdir}/lr.tif"; row["lr_valid"] = f"{cdir}/lr_valid.tif"
    return row

def harness_changes(m, pid, row):
    base = C4.pair_row(m, pid); out = []
    for c in m.columns:
        a, b = base[c], row[c]
        same = (a is b) or (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b))
        try: same = same or bool(a == b)
        except Exception: pass
        if not same: out.append(c)
    return out

def view_table(X):
    """The B4 view of one pair's tile table exactly as phase4_score.score_pair chooses it (interior >= 10 tiles in >= B_min
    blocks; else fallback; else not scorable). Returns (view, tab, target column)."""
    import numpy as np
    t = X[np.isfinite(X.A_fine.values)]; vi = t[t.interior.values.astype(bool)]; vf = t[np.isfinite(t.A_fine_int.values)]
    if len(vi) >= P4.MIN_TILES and len(set(PS.blocks_of(vi))) >= B_MIN: return "interior", vi, "A_fine"
    if len(vf) >= P4.MIN_TILES and len(set(PS.blocks_of(vf))) >= B_MIN: return "fallback", vf, "A_fine_int"
    return "not_scorable", None, None
