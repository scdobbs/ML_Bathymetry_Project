"""R06 (Phase 2, round 1) shared definitions. Importing this module installs the R04 guard (lockbox + confirmatory, through
builtins.open / rasterio.open / numpy.load / pandas.read_parquet) by importing r04_common. start() self-tests the guard,
checks the manifest sha256 and verifies the sha256 of every frozen input listed in directive §0.1 (R05 §6 hashes for the
code and tables; OAK SHA256SUMS for the R05 bar prediction tables and the R04 feature table) and refuses to run if any
differs. Variant definitions, the fixed training schedule and the sampler constants live here so that every script and
the report quote the same values."""
from __future__ import annotations
import datetime, hashlib, json, os, subprocess, sys
REPO = "/home/users/scdobbs/GAN_Bathymetry_Project"
DR = f"{REPO}/grl_reviews-and-prompt-reports/directives_and_reports"
R04_SCRIPTS, R05_SCRIPTS, R06_SCRIPTS = f"{DR}/R04/scripts", f"{DR}/R05/scripts", f"{DR}/R06/scripts"
for p in (REPO, R04_SCRIPTS, R05_SCRIPTS, R06_SCRIPTS):
    if p not in sys.path: sys.path.insert(0, p)
import r04_common as C4            # installs the guard on import
import r05_common as C5
ROOT = "/scratch/users/scdobbs/grl_review/R06"
R04, R05 = C4.ROOT, C5.ROOT
OAK_ARCHIVE = f"{C4.ARCHIVE}"      # .../auv_ship_colocated_bathy/grl_review_archive
INPUTS = f"{ROOT}/inputs"; MODELS = f"{ROOT}/models"; SCORES = f"{ROOT}/scores"; LOGS = f"{ROOT}/logs"
TILES = f"{R05}/tiles/tiles_v1_1.parquet"; BLOCKS = f"{R05}/tiles/blocks_v1_1.parquet"
BAR = f"{R05}/bar"; FEATURES = f"{R04}/features/features_v1.parquet"; FIELDS = f"{R04}/fields"

# ---------------- frozen inputs (directive §0.1): sha256 as recorded in REPORT_R05 §6 ----------------
FROZEN_CODE = {
    f"{R04_SCRIPTS}/targets_v1.py": "51b6a0f0c5136bfeb91162543d571517f3eec911db7bae1d163441d9d57b039b",
    f"{R04_SCRIPTS}/r04_common.py": "6dfe324a46120d3126eeace494781a48612c83fcc4f73610f9073879590adc3e",
    f"{R05_SCRIPTS}/eval_v1_1.py": "647e58909f5981022b86e4191bd4ec344d2851ec46d9f65424a0dc719a1a5307",
    f"{R05_SCRIPTS}/features_v1_1.py": "a19be345458c9be229879a38f01ab6d44ea7b87cd6ab8b91b03510885b0ed840",
    f"{R05_SCRIPTS}/r05_common.py": "0a458b62a16d310571851f30a46b3cfc75c8717eb395a1241575cd501b5189fe",
    f"{R05_SCRIPTS}/bar_v1_1.py": "0699cd454df81b998c1f89c915ded89ef60f55b1f3392753e7f84275d5d73911",
    f"{R05_SCRIPTS}/run_bar_v1_1.py": "6492ac5ad2ed75183797429bf438d42d3ab4a7789aad68132e54628ac9b467ab",
    TILES: "8ef139867613b6c9f2eeb2e13073ee5562ecda51b3cf6f5bb7d17fd73df023dc",
    BLOCKS: "22fd6ee0ab5ecb2367d64126c3d3154eb3cb4e5dd3fa65962f3e85d4718183cc",
}
# the R05 bar prediction tables and the R04 feature table carry no hash in R05 §6; they are checked against the OAK
# archive SHA256SUMS written when they were preserved (R05 §5, R04 §8.1).
OAK_CHECKED = {f"{BAR}/{f}": (f"{OAK_ARCHIVE}/R05/bar/SHA256SUMS", f"./{f}") for f in
               ("pred_tile.parquet", "pred_block.parquet", "pred_tile_EW0207.parquet", "pred_block_EW0207.parquet", "pair_rules.json")}
OAK_CHECKED[FEATURES] = (f"{OAK_ARCHIVE}/R04/features/SHA256SUMS", "./features_v1.parquet")
PHASE0_FREEZE_COMMIT = "866320b7e0917460b63060e86b4a8be8b409c64e"

# ---------------- round-1 design (directive §1-2) ----------------
WIN, TILE, MARGIN, BLOCK, NB = 256, 128, 64, 32, 4          # window, tile, context ring, block, blocks per side
PRODUCTS = ("ship_rsd", "ship_count", "ship_xtrack_frac", "ship_beam_angle")   # V2 branch, in this channel order
VARIANTS = {   # id -> (loss, ship-product branch, central-128 crop)
    "V0": dict(loss="absolute", products=False, crop=False, desc="shared design, absolute MSE on log10 A_fine over kept blocks"),
    "V1": dict(loss="centred", products=False, crop=False, desc="V0 with the within-pair (group) centred MSE"),
    "V2": dict(loss="centred", products=True, crop=False, desc="V1 plus the ship-product branch (channel dropout p = 0.3 per product)"),
    "V3": dict(loss="centred", products=False, crop=True, desc="V1 with the input cropped to the central 128 px"),
}
SEED_MAIN = 20261002; SEEDS_V1 = (20261002, 20261003, 20261004)
# training schedule = R02 recipe (het_louo.py / het_clean.py): 8,000 optimizer steps, batch 32, AdamW 2e-4 wd 1e-4,
# cosine to 1e-6, grad-clip 1.0, bf16 autocast. Under the §1.4 sampler a batch is 4 groups x 8 windows = 32.
STEPS, GROUPS_PER_BATCH, WINDOWS_PER_GROUP = 8000, 4, 8
LR, WEIGHT_DECAY, ETA_MIN, GRAD_CLIP, BASE_CH, CH_MULT = 2e-4, 1e-4, 1e-6, 1.0, 32, (1, 2, 4, 4)
PRODUCT_DROPOUT = 0.3
LOG10_FLOOR_M = 1e-4           # floor inside log10 of A_fine (kept-block minimum in blocks_v1.1 is 5.6e-4 m; never fires)
LOW_AMP_UNITS = ("lu02", "lu06", "lu10", "lu11", "lu12", "vu02")   # §4: median A_fine below 3 cm
BUDGET_GPU_H = 400.0
PRED_COLS = ["V0", "V1_s20261002", "V1_s20261003", "V1_s20261004", "V1_seedavg", "V2", "V3"]

def sha256_file(path): return C4.sha256_file(path)

def oak_expected(sums_path, rel):
    for line in open(sums_path):
        h, _, name = line.rstrip("\n").partition("  ")
        if name == rel: return h
    raise KeyError(f"{rel} not in {sums_path}")

def check_frozen(verbose=True):
    """§0.1: record the sha256 of every frozen input; stop if any differs."""
    rec = {}; bad = []
    for path, exp in FROZEN_CODE.items():
        h = sha256_file(path); rec[path] = dict(sha256=h, expected=exp, ok=(h == exp))
        if h != exp: bad.append(path)
    for path, (sums, rel) in OAK_CHECKED.items():
        exp = oak_expected(sums, rel); h = sha256_file(path); rec[path] = dict(sha256=h, expected=exp, ok=(h == exp), source=sums)
        if h != exp: bad.append(path)
    if bad: raise SystemExit(f"STOP (directive §0.1): frozen inputs differ from R05 §6 / OAK archive: {bad}")
    if verbose: print(f"[r06_common] frozen inputs verified: {len(rec)} files match R05 §6 / OAK SHA256SUMS", flush=True)
    return rec

def git_head():
    try: return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    except Exception: return None

def start(name, check=True):
    """Every R06 script calls this first: guard self-test, manifest hash, frozen-input hashes, provenance record."""
    C4.install_guard(); n = C4.guard_selftest()
    m = C4.load_manifest()
    frozen = check_frozen() if check else {}
    rec = dict(script=name, time=datetime.datetime.now().isoformat(timespec="seconds"), host=os.uname().nodename,
               slurm_job=os.environ.get("SLURM_JOB_ID"), slurm_array_task=os.environ.get("SLURM_ARRAY_TASK_ID"), manifest=C4.MANIFEST,
               manifest_sha256=m.attrs["sha256"], guard_probes_blocked=n, contract_sha256=sha256_file(C4.CONTRACT), units_version=C4.UNITS_VERSION,
               e1m_threshold_m=C4.E1M_THRESHOLD_M, git_head=git_head(), frozen=frozen)
    os.makedirs(f"{ROOT}/provenance", exist_ok=True)
    with open(f"{ROOT}/provenance/{name}_{rec['time'].replace(':', '')}_{os.getpid()}.json", "w") as f: json.dump(rec, f, indent=1)
    print(f"[r06_common.start] {name}: guard self-test blocked {n}/{n} probes; manifest sha256 OK ({m.attrs['sha256'][:12]}…); contract {rec['contract_sha256'][:12]}…; git {rec['git_head'][:8] if rec['git_head'] else '?'}", flush=True)
    return m

def code_hash(files=None):
    """sha256 over the round-1 input-pipeline / model / training / inference sources (directive §6.1); the scoring code is hashed separately."""
    files = files or ["r06_common.py", "build_inputs_r06.py", "r06_model.py", "train_r06.py"]
    h = hashlib.sha256()
    for f in files: h.update(open(f"{R06_SCRIPTS}/{f}", "rb").read())
    return h.hexdigest()

def fold_units(fold):
    """Training units of a fold: the 16 development units minus the held-out one ('ALL' = all 16, scores EW0207 only)."""
    units = list(C4.UNITS)
    if fold == "ALL": return units
    assert fold in units, fold
    return [u for u in units if u != fold]
