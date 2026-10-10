"""R08b shared definitions: imports r08_common (guard + R07 extension, R05 §6 checks, R06/R07 score-table OAK checks) and
adds the R08b fixed-input checks (R08 prediction tables against OAK R08 sums, each pair's joint_v1.tif against the OAK R04
fields sums), the unit sets of the 2026-10-09 ruling and the interior-distance constants of §2."""
from __future__ import annotations
import datetime, json, os, sys
R08_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "R08", "scripts")
if R08_SCRIPTS not in sys.path: sys.path.insert(0, R08_SCRIPTS)
import r08_common as R8, r04_common as C4, r07_common as R7
ROOT = "/scratch/users/scdobbs/grl_review/R08b"; REP = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); TABLES = f"{REP}/tables"
OUT_INT, OUT_SCORES, PROV, LOGS = (f"{ROOT}/{d}" for d in ("interior", "scores", "provenance", "logs"))
# ---- unit sets (ruling of 2026-10-09)
TEN = ("lu00", "lu04", "lu05", "lu07", "lu08", "lu09", "lu13", "lu14", "lu15", "nu00")
SIX = R8.LOW_AMP; NINE = tuple(u for u in TEN if u != "lu08"); ALL16 = R8.UNITS16; NATIVE7 = tuple(u for u in ALL16 if u in C4.NATIVE7)
UNIT_SETS = {"ten": list(TEN), "nine": list(NINE), "six": list(SIX), "all16": list(ALL16)}
assert set(TEN) | set(SIX) == set(ALL16) and not set(TEN) & set(SIX)
# ---- §2
D_LEVELS = {"D1": 1, "D2": 2}            # multiples of ceil(lr_native / hr_res) px
INTERIOR_FRAC = 0.95                      # a tile is interior if >= 95 % of its joint-valid px have distance >= D
MATERIAL_DELTA, MATERIAL_CI_HI = -0.10, -0.05   # §4 reading guide
# ---- prediction sources (file -> (OAK sums dir, rel name))
PRED_SOURCES = {
    "R05_tile": (R8.PRED_TILE_R05, "R05/bar", "pred_tile.parquet"), "R05_tile_EW": (R8.PRED_TILE_EW_R05, "R05/bar", "pred_tile_EW0207.parquet"),
    "R06_tile": (f"{R8.R06_SCORES}/pred_tile_r06.parquet", "R06/scores", "pred_tile_r06.parquet"), "R06_tile_EW": (f"{R8.R06_SCORES}/pred_tile_EW0207_r06.parquet", "R06/scores", "pred_tile_EW0207_r06.parquet"),
    "R07_tile": (R8.PRED_TILE_R07, "R07/scores", "pred_tile_r07.parquet"), "R07_tile_EW": (f"{R8.R07_SCORES}/pred_tile_EW0207_r07.parquet", "R07/scores", "pred_tile_EW0207_r07.parquet"),
    "R08_within": (f"{R8.OUT_WITHIN}/pred_tile_within_site.parquet", "R08/within_site", "pred_tile_within_site.parquet"),
    "R08_dropgroup": (f"{R8.OUT_IMP}/pred_tile_dropgroup.parquet", "R08/importance", "pred_tile_dropgroup.parquet"),
    "R08_within_feasibility": (f"{R8.OUT_WITHIN}/feasibility.json", "R08/within_site", "feasibility.json"),
}
def check_pred_sources(): return {k: R8.oak_verify(sub, rel, path) for k, (path, sub, rel) in PRED_SOURCES.items()}
def verify_field(pid, name="joint_v1.tif"): return R8.oak_verify("R04/fields", f"{pid}/{name}", f"{R8.R04_FIELDS}/{pid}/{name}")

def start(name):
    m = R8.start(name)           # guard self-test, manifest, R05 §6 / R06 / R07 / context checks, provenance under R08
    preds = check_pred_sources()
    for d in (ROOT, OUT_INT, OUT_SCORES, PROV, LOGS, TABLES): os.makedirs(d, exist_ok=True)
    rec = dict(script=name, time=datetime.datetime.now().isoformat(timespec="seconds"), host=os.uname().nodename, slurm_job=os.environ.get("SLURM_JOB_ID"), git_head=R8.git_head(), prediction_sources=preds, unit_sets=UNIT_SETS)
    json.dump(rec, open(f"{PROV}/{name}_{rec['time'].replace(':', '')}_{os.getpid()}.json", "w"), indent=1, default=str)
    print(f"[r08b_common.start] {name}: {len(preds)} prediction sources verified against OAK SHA256SUMS", flush=True); return m
