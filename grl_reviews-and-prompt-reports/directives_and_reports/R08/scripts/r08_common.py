"""R08 (exploratory phase, diagnostics, CPU only) shared definitions. Importing this module imports r07_common, which
installs the R04 lockbox + confirmatory guard (builtins.open / rasterio.open / numpy.load / pandas.read_parquet) and its
R07 extension (geopandas, pyogrio, pyarrow parquet). start() self-tests the guard (72 probes), checks the pairs manifest
sha256, verifies every fixed input of directive R08 §0.2 (R05 §6 hashes and the OAK sums of the R05 bar tables and the
R04 feature table via r06_common.check_frozen; features_v1.py against the R04 freeze record; the R06 scripts against the
phase2-r1 freeze record; the R06 and R07 score tables against their OAK SHA256SUMS; the ACQ-R05 context_v1.parquet
against the directive's sha256) and refuses to run if any differs. The pre-registered constants of §1-§6 live here."""
from __future__ import annotations
import datetime, hashlib, json, os, subprocess, sys
REPO = "/home/users/scdobbs/GAN_Bathymetry_Project"
DR = f"{REPO}/grl_reviews-and-prompt-reports/directives_and_reports"
R04_SCRIPTS, R05_SCRIPTS, R06_SCRIPTS, R07_SCRIPTS, R08_SCRIPTS = (f"{DR}/R0{i}/scripts" for i in (4, 5, 6, 7, 8))
for p in (REPO, R04_SCRIPTS, R05_SCRIPTS, R06_SCRIPTS, R07_SCRIPTS, R08_SCRIPTS):
    if p not in sys.path: sys.path.insert(0, p)
import r07_common as R7              # installs the R04 guard (+ R07 extension on install_guard_ext)
import r06_common as R6
import r04_common as C4
ROOT = "/scratch/users/scdobbs/grl_review/R08"
REP = f"{DR}/R08"; TABLES = f"{REP}/tables"; FIGURES = f"{REP}/figures"
OUT_WITHIN, OUT_IMP, OUT_FIELDS, OUT_LEVEL, OUT_WIDE, CHECKS, PROV, LOGS = (f"{ROOT}/{d}" for d in ("within_site", "importance", "fields", "level_geometry", "wide_window", "checks", "provenance", "logs"))
OAK_ARCHIVE = R6.OAK_ARCHIVE
R05_TILES, R05_BLOCKS, R05_BAR, R04_FEATURES, R04_FIELDS = R6.TILES, R6.BLOCKS, R6.BAR, R6.FEATURES, R6.FIELDS
R06_SCORES = f"{R6.ROOT}/scores"; R07_SCORES = R7.SCORES
PRED_TILE_R07 = f"{R07_SCORES}/pred_tile_r07.parquet"          # R05 bar tile table + the R06 / R07 CNN arms (verified against OAK R07/scores)
PRED_TILE_R05 = f"{R05_BAR}/pred_tile.parquet"
PRED_TILE_EW_R05 = f"{R05_BAR}/pred_tile_EW0207.parquet"
PAIR_RULES = f"{R05_BAR}/pair_rules.json"
CHOSEN_TILE = f"{R05_BAR}/chosen_configs_tile.json"

# ---------------- ACQ-R05 context grids (§0, §6) ----------------
ACQ_REPO = "/home/users/scdobbs/co-located-auv-ship-based-multibeam-acquisition"
CONTEXT_PARQUET = f"{ACQ_REPO}/reports_post_grl_review/ACQ-R05/context_v1.parquet"
CONTEXT_PARQUET_SHA256 = "d09a9faef2b3ce65bdcaf1ebe5628b1b3ee0f2e57de8b71ba1113e3744b6bfa2"
FEATURES_V1_SHA256_R04 = json.load(open(f"{DR}/R04/freeze_record.json"))["code"]["features_v1.py"]   # R04 freeze record (r04-as-run)

# ---------------- pre-registered constants ----------------
SEED = 20261009                      # §1 folds, §2 permutations, §4/§5 bootstraps
SEED_FIGSET_A = 20261010             # §3.3 blind order
LOW_AMP = ("lu02", "lu06", "lu10", "lu11", "lu12", "vu02")           # §0.3 fixed low-amplitude group
OTHER = tuple(u for u in C4.UNITS if u not in LOW_AMP)              # the remaining ten
UNITS16 = tuple(C4.UNITS); UNITS15 = tuple(u for u in UNITS16 if u != "lu08")   # §0: primary / sensitivity without lu08 (ctx_disagrees)
USABLE_UNITS = ("lu00", "lu04", "lu05", "lu07", "lu08", "lu10", "lu11", "lu13", "lu14", "lu15", "nu00", "vu02")   # §5: ship products usable (12); verified against the feature table at run time
EW_UNIT, EW_PAIR = "lu16", C4.EW0207["lu16"][0]
# §1 within-site upper reference
WS_BLOCK_TILES, WS_FOLDS, WS_BUFFER_TILES = 4, 5, 2
WS_MIN_TILES, WS_MIN_BLOCKS, WS_MIN_TRAIN = 80, 5, 40
WS_HGB = dict(max_depth=3, max_iter=300, learning_rate=0.03, min_samples_leaf=20, random_state=SEED)
# §2 permutation importance
N_PERM = 30
# §3 fine-field inspection
SECTOR_DEG = 10; ISO_FRAC = SECTOR_DEG / 180.0          # 0.0556
EDGE_NEAR_PX, EDGE_FAR_PX = 3, 10
SUBWIN_M = 2000.0; SUBWIN_TRIGGER_M = 2500.0; SUBWIN_VALID_FRAC = 0.9; SCALEBAR_M = 500.0
# §4 / §5 bootstraps
N_BOOT = 2000
# §6 wide window
W_CELLS = 64; W_MIN_VALID = 0.5; LEAK_N = 100

def sha256_file(p): return C4.sha256_file(p)

def oak_verify(sub, rel, path):
    """path must hash to the OAK grl_review_archive/<sub>/SHA256SUMS entry for rel."""
    exp = R7.oak_sums(sub)[rel]; h = sha256_file(path)
    if h != exp: raise SystemExit(f"STOP (directive §0.2): {path} sha256 {h} != OAK {sub}/SHA256SUMS {rel} {exp}")
    return dict(path=path, sha256=h, expected=exp, source=f"{OAK_ARCHIVE}/{sub}/SHA256SUMS")

def check_fixed_inputs():
    rec = dict(frozen_r05=R6.check_frozen(verbose=False), r06_freeze=R7.check_r06_freeze())
    h = sha256_file(f"{R04_SCRIPTS}/features_v1.py"); rec["features_v1_py"] = dict(sha256=h, expected=FEATURES_V1_SHA256_R04, ok=(h == FEATURES_V1_SHA256_R04))
    if h != FEATURES_V1_SHA256_R04: raise SystemExit("STOP (directive §0.2): features_v1.py differs from the R04 freeze record")
    rec["r06_scores"] = {f: oak_verify("R06/scores", f, f"{R06_SCORES}/{f}") for f in ("pred_tile_r06.parquet", "pred_block_r06.parquet")}
    rec["r07_scores"] = {f: oak_verify("R07/scores", f, f"{R07_SCORES}/{f}") for f in ("pred_tile_r07.parquet", "pred_block_r07.parquet", "pred_tile_EW0207_r07.parquet")}
    h = sha256_file(CONTEXT_PARQUET); rec["context_v1_parquet"] = dict(sha256=h, expected=CONTEXT_PARQUET_SHA256, ok=(h == CONTEXT_PARQUET_SHA256))
    if h != CONTEXT_PARQUET_SHA256: raise SystemExit("STOP (directive §0.2): context_v1.parquet sha256 differs from the directive")
    return rec

def git_head(): return R7.git_head()

def start(name, check=True):
    """Every R08 script calls this first: guard (R04 + R07 extension) self-test, pairs manifest hash, fixed inputs, provenance."""
    R7.install_guard_ext(); n, n_probes, readers = R7.guard_selftest()
    m = C4.load_manifest()
    fixed = check_fixed_inputs() if check else {}
    for d in (ROOT, PROV, CHECKS, LOGS): os.makedirs(d, exist_ok=True)
    rec = dict(script=name, time=datetime.datetime.now().isoformat(timespec="seconds"), host=os.uname().nodename, slurm_job=os.environ.get("SLURM_JOB_ID"),
               manifest=C4.MANIFEST, manifest_sha256=m.attrs["sha256"], guard_probes_blocked=n, guard_probes=n_probes, guard_readers=readers,
               contract_sha256=sha256_file(C4.CONTRACT), units_version=C4.UNITS_VERSION, git_head=git_head(), fixed_inputs=fixed)
    with open(f"{PROV}/{name}_{rec['time'].replace(':', '')}_{os.getpid()}.json", "w") as f: json.dump(rec, f, indent=1, default=str)
    print(f"[r08_common.start] {name}: guard self-test blocked {n}/{n_probes} probes over {len(readers)} readers; pairs manifest OK ({m.attrs['sha256'][:12]}…); "
          f"fixed inputs {'verified' if fixed else 'not checked'}; git {rec['git_head'][:8] if rec['git_head'] else '?'}", flush=True)
    return m

# ---------------- shared loaders ----------------
def load_pred_tile():
    """Development tile table (R05 bar columns + R06 / R07 CNN arms), the R05 pair rules and the R05 tile table; the R05
    columns of the R07 table are asserted equal to the frozen R05 table."""
    import numpy as np, pandas as pd
    P7 = pd.read_parquet(PRED_TILE_R07); P5 = pd.read_parquet(PRED_TILE_R05)
    assert (P7.tile_id.values == P5.tile_id.values).all() and (P7.unit.values == P5.unit.values).all()
    for c in P5.columns:
        if c in ("unit", "pair", "tile_id"): continue
        a, b = P7[c].values.astype(float), P5[c].values.astype(float); assert np.array_equal(np.isnan(a), np.isnan(b)) and np.array_equal(a[~np.isnan(a)], b[~np.isnan(b)]), c
    rules = json.load(open(PAIR_RULES)); return P7, rules

def load_tiles(): import pandas as pd; return pd.read_parquet(R05_TILES)
def load_features(): import pandas as pd; return pd.read_parquet(R04_FEATURES)
def load_context():
    import pandas as pd; d = pd.read_parquet(CONTEXT_PARQUET)
    for c in ("cell_m", "width", "height", "lr_offset_col", "lr_offset_row"): d[c] = d[c].astype(float)
    return d

def verify_context_pair(row):
    """ctx_lr.tif and ctx_lr_valid.tif of a pair must hash to the pair's context_v1/SHA256SUMS and to the parquet's sha256 columns."""
    sums = {}
    for line in open(f"{row['context_dir']}/SHA256SUMS"):
        h, _, name = line.rstrip("\n").partition("  "); sums[name] = h
    out = {}
    for key, col in (("ctx_lr.tif", "sha256_ctx_lr"), ("ctx_lr_valid.tif", "sha256_ctx_lr_valid")):
        h = sha256_file(f"{row['context_dir']}/{key}"); ok = (h == sums[key] == row[col]); out[key] = dict(sha256=h, sums=sums[key], parquet=row[col], ok=ok)
        if not ok: raise SystemExit(f"STOP (directive §6): {row['pair_id']} {key} sha256 {h} != SHA256SUMS {sums[key]} / parquet {row[col]}")
    return out

def unit_groups(): return dict(six=list(LOW_AMP), ten=list(OTHER), all16=list(UNITS16), fifteen=list(UNITS15))

def boot_median_ci(vals, seed=SEED, n_boot=N_BOOT):
    """Median and 95 % percentile bootstrap CI over the given per-unit values (NaN dropped)."""
    import numpy as np
    v = np.asarray(vals, float); v = v[np.isfinite(v)]; n = len(v)
    if n == 0: return dict(value=float("nan"), ci95=[float("nan"), float("nan")], n=0)
    rng = np.random.default_rng(seed); bs = np.array([np.median(v[rng.integers(0, n, n)]) for _ in range(n_boot)])
    return dict(value=float(np.median(v)), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], n=int(n))

def spearman_boot(x, y, seed=SEED, n_boot=N_BOOT):
    """Spearman rho over paired unit values with a percentile bootstrap CI over units."""
    import numpy as np
    from scipy.stats import spearmanr
    x = np.asarray(x, float); y = np.asarray(y, float); ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]; n = len(x)
    if n < 3: return dict(rho=float("nan"), ci95=[float("nan"), float("nan")], n=int(n))
    rng = np.random.default_rng(seed); bs = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if np.std(x[i]) == 0 or np.std(y[i]) == 0: bs.append(np.nan); continue
        bs.append(spearmanr(x[i], y[i])[0])
    bs = np.array(bs, float)
    return dict(rho=float(spearmanr(x, y)[0]), ci95=[float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))], n=int(n))

def partial_spearman(a, b, z):
    """Partial Spearman rho of a and b given z: Pearson correlation of the residuals of rank(a) and rank(b) regressed on rank(z)."""
    import numpy as np
    from scipy.stats import rankdata
    a = np.asarray(a, float); b = np.asarray(b, float); z = np.asarray(z, float); ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(z)
    if ok.sum() < 5: return float("nan")
    ra, rb, rz = rankdata(a[ok]), rankdata(b[ok]), rankdata(z[ok])
    if np.std(ra) == 0 or np.std(rb) == 0: return float("nan")
    if np.std(rz) == 0: return float(np.corrcoef(ra, rb)[0, 1])
    Z = np.c_[rz, np.ones(len(rz))]; ea = ra - Z @ np.linalg.lstsq(Z, ra, rcond=None)[0]; eb = rb - Z @ np.linalg.lstsq(Z, rb, rcond=None)[0]
    if np.std(ea) == 0 or np.std(eb) == 0: return float("nan")
    return float(np.corrcoef(ea, eb)[0, 1])

def dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f: json.dump(obj, f, indent=1, default=_default)

def _default(o):
    import numpy as np
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)
