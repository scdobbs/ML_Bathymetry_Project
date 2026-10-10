"""R09 (Phase 4 preparation) shared definitions. Imports r08b_common (R04 guard + R07 extension + the R08/R08b fixed-input
checks) and EXTENDS the guard so that ANY path of a confirmatory or lockbox pair raises, through every reader and every
listing / stat entry point (builtins.open, io.open, rasterio.open, numpy.load, pandas.read_parquet, pyarrow, geopandas,
pyogrio, os.listdir / scandir / stat / lstat / walk, glob.glob / iglob, pathlib.Path.open). start() self-tests the guard on
the manifest-resolved paths of all five sealed pairs (hr, lr, masks, products directory) and on bare sealed ids, verifies
the R08b interior table and scores and the R07 P-V1 all-unit models against their OAK sums, and records provenance."""
from __future__ import annotations
import builtins, datetime, glob as _glob, io, json, os, pathlib, sys
R08B_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "R08b", "scripts")
if R08B_SCRIPTS not in sys.path: sys.path.insert(0, R08B_SCRIPTS)
import r08b_common as RB, r08_common as R8, r07_common as R7, r04_common as C4
ROOT = "/scratch/users/scdobbs/grl_review/R09"; REP = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); TABLES = f"{REP}/tables"
OUT_INT, OUT_SCORES, OUT_ART, PROV, LOGS = (f"{ROOT}/{d}" for d in ("targets_int", "scores", "artifacts", "provenance", "logs"))
OAK_R09 = f"{R8.OAK_ARCHIVE}/R09"
# ---- rulings of 2026-10-09 (directive §0)
TEN, NINE, SIX, ALL16 = RB.TEN, RB.NINE, RB.SIX, RB.ALL16
UNIT_SETS = RB.UNIT_SETS; NO_INTERIOR = ("lu14",)            # shown beside the interior rows, out of every median
CANDIDATES = {"s_lr": "ref_s_lr", "GBT_ship": "GBT_ship_A_fine", "P-V1": "P-V1_seedavg"}
# ---- Part A constants
MIN_INT_PX = 2048; CELL_STRESS_M = 125.0; VALIDATION_TOL = 0.05; INTERIOR_FRAC = RB.INTERIOR_FRAC
# ---- fixed inputs beyond R08b
R08B_INTERIOR = f"{RB.OUT_INT}/tiles_interior.parquet"; R08B_SCORES = f"{RB.OUT_SCORES}/scores.json"
PV1_ALL = {s: f"{R7.MODELS}/P-V1/seed{s}/fold_ALL" for s in R7.SEEDS}

def check_fixed_r09():
    rec = dict(r08b_interior=R8.oak_verify("R08b/interior", "tiles_interior.parquet", R08B_INTERIOR), r08b_scores=R8.oak_verify("R08b/scores", "scores.json", R08B_SCORES),
               r08b_lists={}, pv1_all={})
    for pid in C4.DEV_PAIRS + [R8.EW_PAIR]:
        for k in ("D1", "D2"): rec["r08b_lists"][f"{pid}_{k}"] = R8.oak_verify("R08b/interior", f"lists/{pid}_{k}.csv", f"{RB.OUT_INT}/lists/{pid}_{k}.csv")["sha256"]
    for s, d in PV1_ALL.items():
        rec["pv1_all"][str(s)] = {f: R8.oak_verify("R07/models", f"P-V1/seed{s}/fold_ALL/{f}", f"{d}/{f}")["sha256"] for f in ("ckpt.pt", "pred.parquet", "train_log.json")}
    return rec

# ---- guard extension: sealed tokens (pair ids, cruise and MGDS parts, every manifest path of a sealed row), listing / stat / io.open / pathlib
_ext = False; SEALED_TOKENS = []
def sealed_tokens(m):
    toks = set()
    for pid in C4.SEALED:
        row = m[m.pair_id == pid].iloc[0]; toks.add(pid); cruise, _, mgds = pid.partition("__"); toks.add(cruise); toks.add(mgds)
        for c in ("harmonized_path_hr", "harmonized_path_lr", "hr_valid", "lr_valid", "joint_valid", "harmonized_path", "raw_path_hr", "raw_path_lr", "qc_artifact_path", "hr_local_path"):
            v = row.get(c)
            if isinstance(v, str) and v: toks.add(v.rstrip("/"))
    toks.add("harmonized_confirmatory"); return sorted(toks)
def _is_sealed(s):
    return any(t in s for t in SEALED_TOKENS)
def _check_r09(path):
    s = str(os.fspath(path)) if not isinstance(path, int) else ""
    if _is_sealed(s): raise PermissionError(f"R09 GUARD (sealed pair token): refusing {s}")
    return C4._check_r04(path)
def _chk(p):
    if isinstance(p, (str, bytes, os.PathLike)): _check_r09(p)
    return p
def install_guard_r09(m):
    global _ext, SEALED_TOKENS
    if _ext: return
    SEALED_TOKENS = sealed_tokens(m)
    R7.install_guard_ext()
    if not hasattr(C4, "_check_r04"): C4._check_r04 = C4._check
    C4._check = _check_r09                      # r04_common's guarded readers (and the R07 extension) look _check up at call time
    _io_open = io.open
    def g_io_open(file, *a, **k): _chk(file); return _io_open(file, *a, **k)
    io.open = g_io_open
    _ld, _sd, _st, _lst = os.listdir, os.scandir, os.stat, os.lstat
    def g_listdir(path="."): _chk(path); return [n for n in _ld(path) if not _is_sealed(os.path.join(str(os.fspath(path)), n))]
    class _Scan:
        """os.scandir result filtered of sealed entries; iterator and context manager like the original."""
        def __init__(self, it): self._it = it
        def __iter__(self): return (e for e in self._it if not _is_sealed(e.path))
        def __enter__(self): return self
        def __exit__(self, *a): self._it.close(); return False
        def close(self): self._it.close()
    def g_scandir(path="."): _chk(path); return _Scan(_sd(path))
    def g_stat(path, *a, **k): _chk(path); return _st(path, *a, **k)
    def g_lstat(path, *a, **k): _chk(path); return _lst(path, *a, **k)
    os.listdir, os.scandir, os.stat, os.lstat = g_listdir, g_scandir, g_stat, g_lstat
    _walk = os.walk
    def g_walk(top, *a, **k): _chk(top); return _walk(top, *a, **k)
    os.walk = g_walk
    _gg, _gi = _glob.glob, _glob.iglob
    def g_glob(pathname, *a, **k): _chk(pathname); return [x for x in _gg(pathname, *a, **k) if not _is_sealed(x)]
    def g_iglob(pathname, *a, **k): _chk(pathname); return (x for x in _gi(pathname, *a, **k) if not _is_sealed(x))
    _glob.glob, _glob.iglob = g_glob, g_iglob
    _popen = pathlib.Path.open
    def g_popen(self, *a, **k): _chk(str(self)); return _popen(self, *a, **k)
    pathlib.Path.open = g_popen
    _ext = True

def sealed_paths(m):
    """Every manifest-resolved path of the five sealed pairs (never opened): each path column that is set (harmonized hr /
    lr / masks / directory, raw, qc), the derived hr / lr / mask / products paths under the pair's harmonized directory
    (the confirmatory rows carry only `harmonized_path`), and the bare ids."""
    out = {}
    for pid in C4.SEALED:
        row = m[m.pair_id == pid]; assert len(row) == 1, pid; row = row.iloc[0]; paths = []
        for c in ("harmonized_path_hr", "harmonized_path_lr", "hr_valid", "lr_valid", "joint_valid", "harmonized_path", "raw_path_hr", "raw_path_lr", "qc_artifact_path", "hr_local_path"):
            v = row.get(c)
            if isinstance(v, str) and v: paths.append(os.path.join(C4.DATA, v))
        base = os.path.join(C4.DATA, str(row["harmonized_path"]))
        paths += [os.path.join(base, f) for f in ("hr.tif", "lr.tif", "hr_valid.tif", "lr_valid.tif", "joint_valid.tif", "ship_products_v2_1", "context_v1/ctx_lr.tif")]
        paths += [os.path.join(C4.DATA, "harmonized", pid), pid]
        out[pid] = sorted(set(paths))
    return out

def guard_selftest_r09(m):
    import rasterio, numpy as np, pandas as pd, pyarrow.parquet as pq
    readers = {"open": open, "io.open": io.open, "rasterio.open": rasterio.open, "numpy.load": np.load, "pandas.read_parquet": pd.read_parquet, "pyarrow.read_table": pq.read_table,
               "os.listdir": os.listdir, "os.scandir": os.scandir, "os.stat": os.stat, "os.lstat": os.lstat, "glob.glob": _glob.glob, "Path.open": lambda p: pathlib.Path(p).open(), "os.walk": lambda p: next(iter(os.walk(p)))}
    n = 0; probes = 0; rows = []
    for pid, paths in sealed_paths(m).items():
        for p in paths:
            for name, fn in readers.items():
                probes += 1
                try: fn(p)
                except PermissionError as e:
                    if "GUARD" not in str(e): raise AssertionError(f"PermissionError not from the guard: {name}({p}): {e}")
                    n += 1; continue
                except Exception as e: raise AssertionError(f"guard did not block {name}({p}): {type(e).__name__}: {e}")
                raise AssertionError(f"guard did not block {name}({p})")
            rows.append(dict(pair=pid, path=p))
    assert n == probes
    return n, probes, sorted(readers), rows

def start(name, check=True):
    m0 = C4.load_manifest(); install_guard_r09(m0)
    m = RB.start(name) if check else m0          # R08 chain (R04 guard self-test, R05 §6, R06/R07/R08 sums) + R08b prediction sources
    n, probes, readers, rows = guard_selftest_r09(m)
    fixed = check_fixed_r09() if check else {}
    for d in (ROOT, OUT_INT, OUT_SCORES, OUT_ART, PROV, LOGS, TABLES): os.makedirs(d, exist_ok=True)
    rec = dict(script=name, time=datetime.datetime.now().isoformat(timespec="seconds"), host=os.uname().nodename, slurm_job=os.environ.get("SLURM_JOB_ID"), git_head=R8.git_head(),
               sealed_guard_probes_blocked=n, sealed_guard_probes=probes, sealed_guard_readers=readers, sealed_paths_probed=rows, sealed_tokens=SEALED_TOKENS, fixed_inputs_r09=fixed)
    json.dump(rec, open(f"{PROV}/{name}_{rec['time'].replace(':', '')}_{os.getpid()}.json", "w"), indent=1, default=str)
    print(f"[r09_common.start] {name}: sealed-path guard self-test blocked {n}/{probes} probes ({len(readers)} entry points x {len(rows)} paths of {len(C4.SEALED)} sealed pairs); R08b/R07 artifacts {'verified' if fixed else 'not checked'}", flush=True)
    return m
