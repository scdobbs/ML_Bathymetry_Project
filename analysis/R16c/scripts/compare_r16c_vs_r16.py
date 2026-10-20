"""R16c must-hold 3.2 — the dress rehearsal against freeze v2 (R16c root) must reproduce the R16 rehearsal (R16b root rehearsal_b,
job 47690627) on the three eligible mock sites: the ACQ-R11b rasters are bit-identical to ACQ-R09's (REPORT_ACQ-R11b bullet 4).
Compared (development data only; values are compared, not interpreted):
  - the copied mock rasters (sha256 per file, R16 mock vs R16c mock);
  - build: tiles_v1_1 / tiles_int / features parquet frames (exact, NaN pattern equal);
  - predictions/<pair>.parquet frames (exact);
  - scores: inputs_<tag>.parquet frames, and results_<tag>.json for D1 and D1_artifact with every key named path / sha256 / time /
    seconds / key_path / host / node / acq_record (k provenance: the run record's path) removed recursively (the paths and the acquisition record differ by construction).
Writes R16c/tables/r16c_vs_r16.json; exit 1 on any difference."""
import glob, hashlib, json, os, sys
import numpy as np, pandas as pd
import r16c_common as R
A = R.R16_ROOT_B; B = R.ROOT; MA = "/scratch/users/scdobbs/grl_review/R16/mock/auv_ship_colocated_bathy/phase5/harmonized"; MB = R.MOCK_HARM
DROP = {"path", "sha256", "time", "seconds", "key_path", "host", "node", "inputs", "calls", "inputs_sha256", "calls_sha256", "acq_run_record", "inputs_assembly", "acq_record"}   # acq_record: the path of the mock run record (R16: ..._r09.json; R16c: ..._r11b.json), added after job 47710277
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
def strip(o):
    if isinstance(o, dict): return {k: strip(v) for k, v in o.items() if k not in DROP}
    if isinstance(o, list): return [strip(v) for v in o]
    return o
def frames_equal(pa, pb):
    a, b = pd.read_parquet(pa), pd.read_parquet(pb)
    if list(a.columns) != list(b.columns) or len(a) != len(b): return dict(equal=False, why=f"shape/columns {a.shape} {b.shape}")
    try: pd.testing.assert_frame_equal(a, b, check_exact=True); return dict(equal=True, n=len(a))
    except AssertionError as e: return dict(equal=False, why=str(e)[:300])
def diff_paths(a, b, path=""):
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)): out += diff_paths(a.get(k, "<missing>"), b.get(k, "<missing>"), f"{path}/{k}")
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)): out += diff_paths(x, y, f"{path}[{i}]")
    elif not (a == b or (isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b))): out.append(dict(at=path, r16=str(a)[:80], r16c=str(b)[:80]))
    return out
if __name__ == "__main__":
    out = dict(r16_root=A, r16c_root=B, rasters={}, build={}, predictions={}, scores={})
    for pid in R.MOCK_SITES:
        fa = {os.path.relpath(p, f"{MA}/{pid}"): sha(p) for p in glob.glob(f"{MA}/{pid}/**/*.tif", recursive=True)}; fb = {os.path.relpath(p, f"{MB}/{pid}"): sha(p) for p in glob.glob(f"{MB}/{pid}/**/*.tif", recursive=True)}
        out["rasters"][pid] = dict(n=len(fa), identical=(fa == fb and len(fa) > 0))
    for pid in R.ELIGIBLE_MOCK:
        out["build"][pid] = {f: frames_equal(f"{A}/build/{pid}/{f}.parquet", f"{B}/build/{pid}/{f}.parquet") for f in ("tiles_v1_1", "tiles_int", "features")}
        out["predictions"][pid] = frames_equal(f"{A}/predictions/{pid}.parquet", f"{B}/predictions/{pid}.parquet")
    for tag in ("D1", "D1_artifact"):
        ra, rb = strip(json.load(open(f"{A}/scores/results_{tag}.json"))), strip(json.load(open(f"{B}/scores/results_{tag}.json"))); d = diff_paths(ra, rb)
        out["scores"][tag] = dict(inputs=frames_equal(f"{A}/scores/inputs_{tag}.parquet", f"{B}/scores/inputs_{tag}.parquet"), results_identical=(not d), differences=d[:20],
                                  C1_primary=(rb.get("C1_primary") or {}).get("reading"), C1_secondary=(rb.get("C1_per_site_secondary") or {}).get("reading"))
    ok = (all(v["identical"] for v in out["rasters"].values()) and all(x["equal"] for v in out["build"].values() for x in v.values()) and all(v["equal"] for v in out["predictions"].values())
          and all(v["inputs"]["equal"] and v["results_identical"] for v in out["scores"].values()))
    out["all_identical"] = bool(ok); os.makedirs(R.TABLES, exist_ok=True); json.dump(out, open(f"{R.TABLES}/r16c_vs_r16.json", "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in out.items() if k != "build"}, indent=1, default=str)[:4000]); print("R16c vs R16 all identical:", ok); sys.exit(0 if ok else 1)
