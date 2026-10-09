"""R07 (Phase 3) shared definitions. Importing this module imports r06_common (which installs the R04 lockbox +
confirmatory guard on builtins.open / rasterio.open / numpy.load / pandas.read_parquet) and EXTENDS the guard to the
readers ACQ-R04 found bypassing it: geopandas.read_file / read_parquet, pyogrio.read_dataframe / read_arrow / raw.read,
pyarrow.parquet.read_table / ParquetFile / ParquetDataset. start() self-tests every patched reader against lockbox and
confirmatory paths, checks the pairs manifest sha256, verifies every frozen input of directive §0.2 (R05 §6 hashes via
r06_common.check_frozen, the R06 scripts against R06/freeze_record.json, the corpus manifest and SHA256SUMS against the
directive's hashes) and refuses to run if any differs. The pre-registered constants of §1-§3 live here so that every
script and the report quote the same values."""
from __future__ import annotations
import datetime, hashlib, json, os, subprocess, sys
REPO = "/home/users/scdobbs/GAN_Bathymetry_Project"
DR = f"{REPO}/grl_reviews-and-prompt-reports/directives_and_reports"
R04_SCRIPTS, R05_SCRIPTS, R06_SCRIPTS, R07_SCRIPTS = f"{DR}/R04/scripts", f"{DR}/R05/scripts", f"{DR}/R06/scripts", f"{DR}/R07/scripts"
for p in (REPO, R04_SCRIPTS, R05_SCRIPTS, R06_SCRIPTS, R07_SCRIPTS):
    if p not in sys.path: sys.path.insert(0, p)
import r06_common as R6               # installs the R04 guard on import (via r04_common -> r03_common)
import r04_common as C4
ROOT = "/scratch/users/scdobbs/grl_review/R07"
CORPUS = f"{ROOT}/corpus"                      # staged copy of OAK pretrain_v1/ (stage_corpus.sbatch)
CACHE = f"{ROOT}/cache"                        # pretraining window cache (§1.5)
MODELS = f"{ROOT}/models"                      # fine-tuning arms: models/<arm>/seed<seed>/fold_<fold>/
PRETRAIN = f"{ROOT}/pretrain"                  # pretraining runs: pretrain/<body>/
SCORES = f"{ROOT}/scores"; LOGS = f"{ROOT}/logs"; CHECKS = f"{ROOT}/checks"; PROV = f"{ROOT}/provenance"
OAK_CORPUS = "/oak/stanford/groups/hilley/auv_ship_colocated_bathy/pretrain_v1"
OAK_ARCHIVE = R6.OAK_ARCHIVE                   # .../auv_ship_colocated_bathy/grl_review_archive
R06_MODELS = R6.MODELS; R06_INPUTS = R6.INPUTS; R06_FREEZE = f"{DR}/R06/freeze_record.json"
PHASE0_FREEZE_COMMIT = R6.PHASE0_FREEZE_COMMIT; PHASE2_R1_COMMIT = "683a99fd4d2c4dadd9c9207849c2cbef573018f0"

# ---------------- fixed inputs (directive §0.2) ----------------
CORPUS_MANIFEST_SHA256 = "12709204d3ed2fc1043d09740f754b45c8f624bab95eaa1421d1167b8226ca69"
CORPUS_SUMS_SHA256 = "bbcc6ff1e6e7ec9f7c840ce052b2b5c3932aaccf106c0fd617e4dc61ac72cb26"
CORPUS_MANIFEST = f"{CORPUS}/meta/pretrain_manifest_v1.parquet"; CORPUS_SUMS = f"{CORPUS}/SHA256SUMS"

# ---------------- pretraining data (§1) ----------------
CELL_MIN_M, CELL_MAX_M, N_BINS = 10.0, 256.0, 8          # §1.1 / §1.2: 8 equal-width log2 bins from log2 10 to log2 256
HOLDOUT_FRAC, HOLDOUT_SEED = 0.05, 20261007              # §1.3: 5 % of corpus cruises, stratified by instrument family
WIN, TILE, MARGIN, BLOCK, NB = R6.WIN, R6.TILE, R6.MARGIN, R6.BLOCK, R6.NB     # 256, 128, 64, 32, 4
MIN_VALID_FRAC = 0.5                                     # §1.4: a window needs >= 50 % of its LR cells valid
S_FLOOR = 0.5                                            # build_inputs s_lr floor (R02 / R06 value)
PRODUCTS = R6.PRODUCTS                                   # ("ship_rsd", "ship_count", "ship_xtrack_frac", "ship_beam_angle")
RSD_FLOOR_M = 1e-3                                       # log10 max(rsd, 1 mm): the R06 product normalisation floor, reused for the target
BLOCK_COVER_MIN = 0.5                                    # a 32-px block counts as covered by ship_rsd if >= 50 % of its area has finite rsd (executor's call)
N_TRAIN_WINDOWS = 60_000 * 64                            # one unique window per training presentation (steps x batch)
N_VAL_WINDOWS = 16_384                                   # held-out pretext validation set (fixed windows, fixed masks)
WINDOWS_PER_SHARD = 48_000

# ---------------- pretraining (§2) ----------------
MASK_FRAC = 0.30
PRE_STEPS, PRE_BATCH, PRE_LR, PRE_WD, PRE_ETA_MIN, PRE_GRAD_CLIP, PRE_SEED = 60_000, 64, 2e-4, 1e-4, 1e-6, 1.0, 20261007
PILOT_STEPS, PILOT_EVAL_STEPS = 2_000, (0, 500, 1000, 2000)
EVAL_EVERY = 2_000
BODIES = {"P-V1": dict(products=False, desc="R06 V1 body (UNetBody, FiLM); recon head + spread head"),
          "P-V2": dict(products=True, desc="R06 V2 body + ship-product branch; ship_rsd channel and mask zeroed on every pretraining window")}

# ---------------- fine-tuning (§3) ----------------
ARMS = {   # arm -> (R06 variant whose code / inputs / loss it uses, pretrained body or None)
    "P-V1": dict(variant="V1", init="P-V1", desc="R06 V1 initialised from the P-V1 body; new zero-init block head"),
    "P-V2": dict(variant="V2", init="P-V2", desc="R06 V2 initialised from the P-V2 body (incl. product branch); new zero-init block head"),
    "V2":   dict(variant="V2", init=None, desc="R06 V2 from scratch, seeds 20261003 / 20261004 (seed completion; seed 20261002 is R06's)"),
}
SEEDS = (20261002, 20261003, 20261004)
SEED_MAIN = 20261002
BUDGET_GPU_H = 300.0
MAX_CONCURRENT_GPUS = 8
EQUIV_FOLD, EQUIV_SEED, EQUIV_GPU = "lu00", 20261002, "NVIDIA H100 80GB HBM3"     # §3.3 (R06 V1/lu00/20261002 ran on this GPU)

def sha256_file(path): return C4.sha256_file(path)
def sha256_bytes(b): return hashlib.sha256(b).hexdigest()

# ---------------- guard extension ----------------
_ext_installed = False
def install_guard_ext():
    """Extend the R04 guard (already installed) to geopandas, pyogrio and pyarrow parquet readers (ACQ-R04 §0.2)."""
    global _ext_installed
    if _ext_installed: return
    C4.install_guard()
    def chk(p):
        """Path-like -> the R04 check; open file objects -> check their .name (their open() was already guarded); else pass."""
        if isinstance(p, (str, bytes, os.PathLike)): return C4._check(p)
        name = getattr(p, "name", None)
        if isinstance(name, (str, bytes, os.PathLike)): C4._check(name)
        return p
    import pyarrow.parquet as pq
    _rt, _pf, _pd = pq.read_table, pq.ParquetFile, pq.ParquetDataset
    def g_rt(source, *a, **k): chk(source); return _rt(source, *a, **k)
    class g_pf(_pf):
        def __init__(self, source, *a, **k): chk(source); super().__init__(source, *a, **k)
    class g_pd(_pd):
        def __init__(self, path_or_paths, *a, **k):
            for p in (path_or_paths if isinstance(path_or_paths, (list, tuple)) else [path_or_paths]): chk(p)
            super().__init__(path_or_paths, *a, **k)
    pq.read_table, pq.ParquetFile, pq.ParquetDataset = g_rt, g_pf, g_pd
    try:
        import pyogrio; _rd, _ra = pyogrio.read_dataframe, pyogrio.read_arrow
        def g_rd(path_or_buffer, *a, **k): chk(path_or_buffer); return _rd(path_or_buffer, *a, **k)
        def g_ra(path_or_buffer, *a, **k): chk(path_or_buffer); return _ra(path_or_buffer, *a, **k)
        pyogrio.read_dataframe, pyogrio.read_arrow = g_rd, g_ra
        import pyogrio.raw as praw; _rr = praw.read
        def g_rr(path_or_buffer, *a, **k): chk(path_or_buffer); return _rr(path_or_buffer, *a, **k)
        praw.read = g_rr
    except ImportError: pass
    try:
        import geopandas as gpd; _gf, _gp = gpd.read_file, gpd.read_parquet
        def g_gf(filename, *a, **k): chk(filename); return _gf(filename, *a, **k)
        def g_gp(path, *a, **k): chk(path); return _gp(path, *a, **k)
        gpd.read_file, gpd.read_parquet = g_gf, g_gp
    except ImportError: pass
    _ext_installed = True

def guard_selftest():
    """Every patched reader x every R04 probe path must raise PermissionError from the guard. Returns (n_blocked, n_probes)."""
    import rasterio, numpy as np, pandas as pd, pyarrow.parquet as pq
    readers = {"open": open, "rasterio.open": rasterio.open, "numpy.load": np.load, "pandas.read_parquet": pd.read_parquet,
               "pyarrow.parquet.read_table": pq.read_table, "pyarrow.parquet.ParquetFile": pq.ParquetFile, "pyarrow.parquet.ParquetDataset": pq.ParquetDataset}
    try:
        import pyogrio; readers.update({"pyogrio.read_dataframe": pyogrio.read_dataframe, "pyogrio.read_arrow": pyogrio.read_arrow})
        import pyogrio.raw as praw; readers["pyogrio.raw.read"] = praw.read
    except ImportError: pass
    try:
        import geopandas as gpd; readers.update({"geopandas.read_file": gpd.read_file, "geopandas.read_parquet": gpd.read_parquet})
    except ImportError: pass
    D = C4.DATA; L = C4.LOCKBOX; CF = C4.CONFIRMATORY
    probes = [f"{D}/harmonized/{L[0]}/hr.tif", f"{D}/harmonized/{L[1]}/lr.tif", f"{D}/harmonized_confirmatory/nu03/{CF[2]}/hr.tif",
              f"{D}/harmonized_confirmatory/nu01b/anything.gpkg", f"{D}/harmonized/{CF[0]}/hr.tif", f"{D}/harmonized_confirmatory/nu01a/{CF[0]}/footprint.geojson"]
    n = 0; rows = []
    for p in probes:
        for name, fn in readers.items():
            try: fn(p)
            except PermissionError as e:
                if "GUARD" not in str(e): raise AssertionError(f"PermissionError not from the guard for {name}({p}): {e}")
                n += 1; rows.append((name, p, "blocked")); continue
            except Exception as e: raise AssertionError(f"guard did not block {name}({p}): {type(e).__name__}: {e}")
            raise AssertionError(f"guard did not block {name}({p})")
    assert n == len(probes) * len(readers)
    return n, len(probes) * len(readers), sorted(readers)

# ---------------- frozen inputs ----------------
def check_r06_freeze():
    """The R06 scripts (phase2-r1) must hash to the values in R06/freeze_record.json."""
    fr = json.load(open(R06_FREEZE)); rec = {}; bad = []
    for f, exp in fr["scripts_sha256"].items():
        h = sha256_file(f"{R06_SCRIPTS}/{f}"); rec[f] = dict(sha256=h, expected=exp, ok=(h == exp))
        if h != exp: bad.append(f)
    if R6.code_hash() != fr["training_code_sha256"]: bad.append("training_code_sha256")
    if bad: raise SystemExit(f"STOP (directive §0.2): R06 scripts differ from the phase2-r1 freeze record: {bad}")
    return dict(scripts=rec, training_code_sha256=R6.code_hash(), expected=fr["training_code_sha256"])

def check_corpus(require=True):
    """Corpus manifest and SHA256SUMS (staged copy) against the directive's hashes."""
    out = {}
    for path, exp in ((CORPUS_MANIFEST, CORPUS_MANIFEST_SHA256), (CORPUS_SUMS, CORPUS_SUMS_SHA256)):
        if not os.path.exists(path):
            if require: raise SystemExit(f"STOP: corpus not staged: {path}")
            out[path] = dict(sha256=None, expected=exp, ok=False); continue
        h = sha256_file(path); out[path] = dict(sha256=h, expected=exp, ok=(h == exp))
        if h != exp: raise SystemExit(f"STOP (directive §0.2): {path} sha256 {h} != {exp}")
    return out

_oak_sums = {}
def oak_sums(sub):
    """Parsed SHA256SUMS of an OAK archive directory (e.g. 'R06/models')."""
    if sub not in _oak_sums:
        d = {}
        for line in open(f"{OAK_ARCHIVE}/{sub}/SHA256SUMS"):
            h, _, name = line.rstrip("\n").partition("  "); d[name[2:] if name.startswith("./") else name] = h
        _oak_sums[sub] = d
    return _oak_sums[sub]

def verify_r06_file(rel):
    """A file under R06 scratch models/ must hash to its OAK SHA256SUMS entry (the R06 outputs are fixed inputs, §0.2)."""
    exp = oak_sums("R06/models")[rel]; h = sha256_file(f"{R06_MODELS}/{rel}")
    if h != exp: raise SystemExit(f"STOP (directive §0.2): R06 output {rel} differs from the OAK archive")
    return h

def git_head():
    try: return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    except Exception: return None

def start(name, check_corpus_files=True, check=True):
    """Every R07 script calls this first: guard (R04 + extension) self-test, pairs-manifest hash, frozen inputs (R05 §6 via
    r06_common, R06 freeze record, corpus hashes), provenance record under provenance/."""
    install_guard_ext(); n, n_probes, readers = guard_selftest()
    m = C4.load_manifest()
    frozen = R6.check_frozen(verbose=False) if check else {}
    r06 = check_r06_freeze() if check else {}
    corpus = check_corpus(require=check_corpus_files) if check else {}
    rec = dict(script=name, time=datetime.datetime.now().isoformat(timespec="seconds"), host=os.uname().nodename,
               slurm_job=os.environ.get("SLURM_JOB_ID"), slurm_array_task=os.environ.get("SLURM_ARRAY_TASK_ID"), manifest=C4.MANIFEST,
               manifest_sha256=m.attrs["sha256"], guard_probes_blocked=n, guard_probes=n_probes, guard_readers=readers,
               contract_sha256=sha256_file(C4.CONTRACT), units_version=C4.UNITS_VERSION, git_head=git_head(),
               frozen_r05=frozen, r06_freeze=r06, corpus=corpus)
    os.makedirs(PROV, exist_ok=True)
    with open(f"{PROV}/{name}_{rec['time'].replace(':', '')}_{os.getpid()}.json", "w") as f: json.dump(rec, f, indent=1)
    print(f"[r07_common.start] {name}: guard self-test blocked {n}/{n_probes} probes over {len(readers)} readers; pairs manifest OK ({m.attrs['sha256'][:12]}…); "
          f"frozen R05 inputs {len(frozen)} OK; R06 freeze record OK; corpus hashes {'OK' if corpus and all(v['ok'] for v in corpus.values()) else 'not checked'}; git {rec['git_head'][:8] if rec['git_head'] else '?'}", flush=True)
    return m

def code_hash(files=None):
    """sha256 over the R07 window-builder / pretraining / fine-tuning sources (plus the unchanged R06 model code they import)."""
    files = files or CODE_FILES
    h = hashlib.sha256()
    for f in files:
        p = f"{R07_SCRIPTS}/{f}"
        if os.path.exists(p): h.update(open(p, "rb").read())
    h.update(open(f"{R06_SCRIPTS}/r06_model.py", "rb").read())
    return h.hexdigest()

CODE_FILES = ["r07_common.py", "r07_windows.py", "build_cache_r07.py", "r07_pretext_model.py", "pretrain_r07.py", "train_r07.py"]
def code_files_present(): return [f for f in CODE_FILES if os.path.exists(f"{R07_SCRIPTS}/{f}")]

def fold_units(fold): return R6.fold_units(fold)
