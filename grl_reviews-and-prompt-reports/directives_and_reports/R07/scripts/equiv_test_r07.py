"""R07 §1.4 equivalence test (required; §0.1 item 5). The pretraining window builder (r07_windows: extract_lr_tile +
assemble) is run on 200 windows drawn from development pairs' own lr.tif, with positions and hr_res taken from R06's
input records (inputs/<pair>/meta.parquet). Its output must be byte-identical to (a) the R06 input arrays stored on
scratch (x.npy, lrv.npy, cond in meta.parquet, prod.npy, prodm.npy) and (b) build_inputs_r06.window_inputs recomputed
from the same geometry, for x, the validity channel, the FiLM vector and the V2 channels + masks. Any difference stops
the directive. Writes checks/equiv_windows.json."""
from __future__ import annotations
import hashlib, json, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r07_common as R7, r06_common as R6, r04_common as C4, r07_windows as W
import build_inputs_r06 as B6
from preprocessing import extract_clean_patches as ecp

def hb(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    m = R7.start("equiv_test_r07"); os.makedirs(R7.CHECKS, exist_ok=True); t0 = time.time()
    metas = []
    for pid in C4.DEV_PAIRS:
        d = pd.read_parquet(f"{R6.INPUTS}/{pid}/meta.parquet"); d["row"] = np.arange(len(d)); metas.append(d)
    T = pd.concat(metas, ignore_index=True)
    # stratified draw: at least 2 windows per pair (76), the rest uniform over all windows -> 200
    rng = np.random.default_rng(R7.HOLDOUT_SEED)
    pick = pd.concat([g.sample(n=min(2, len(g)), random_state=int(rng.integers(1 << 31))) for _, g in T.groupby("pair")])
    rest = T.drop(pick.index).sample(n=n - len(pick), random_state=int(rng.integers(1 << 31)))
    S = pd.concat([pick, rest]).sort_values(["pair", "row"])
    rows = []; n_same_stored = n_same_recomputed = 0; n_usable = 0
    for pid, sub in S.groupby("pair"):
        row = C4.pair_row(m, pid); g = ecp.pair_setup(C4.DATA, row, R6.WIN); P = B6.load_products(row, g); usable = P is not None
        d = f"{R6.INPUTS}/{pid}"; X = np.load(f"{d}/x.npy", mmap_mode="r"); LV = np.load(f"{d}/lrv.npy", mmap_mode="r"); meta = pd.read_parquet(f"{d}/meta.parquet")
        PR = np.load(f"{d}/prod.npy", mmap_mode="r") if usable else None; PM = np.load(f"{d}/prodm.npy", mmap_mode="r") if usable else None
        assert abs(float(meta.hr_res.iloc[0]) - g["hr_res"]) < 1e-9 and abs(float(meta.lr_native.iloc[0]) - g["lr_native"]) < 1e-9
        for r in sub.itertuples():
            i = int(r.row); r0, c0 = int(r.r0), int(r.c0)
            new = W.window_inputs(g, P, r0, c0, products=True); ref = B6.window_inputs(g, P, r0, c0)
            cond_stored = meta[["cond0", "cond1", "cond2", "cond3"]].values[i].astype(np.float32)
            same_stored = hb(new["x"]) == hb(X[i]) and hb(new["lrv"]) == hb(LV[i]) and hb(new["cond"]) == hb(cond_stored)
            same_recomp = hb(new["x"]) == hb(ref["x"]) and hb(new["lrv"]) == hb(ref["lrv"]) and hb(new["cond"]) == hb(ref["cond"])
            if usable:
                same_stored = same_stored and hb(new["prod"]) == hb(PR[i]) and hb(new["prodm"]) == hb(PM[i])
                same_recomp = same_recomp and hb(new["prod"]) == hb(ref["prod"]) and hb(new["prodm"]) == hb(ref["prodm"])
            else:   # R06 stores no product arrays for non-usable pairs; its loader feeds zero channels and zero masks
                same_stored = same_stored and not new["prod"].any() and not new["prodm"].any(); same_recomp = same_recomp and ref["prod"] is None
            rows.append(dict(pair=pid, tile_id=r.tile_id, row=i, r0=r0, c0=c0, hr_res=g["hr_res"], lr_native=g["lr_native"], lr_px=int(g["lr_px"]), usable=usable,
                             identical_to_stored=bool(same_stored), identical_to_recomputed=bool(same_recomp)))
            n_same_stored += same_stored; n_same_recomputed += same_recomp; n_usable += usable
        del g
    res = dict(n=len(rows), n_identical_to_stored_r06_arrays=int(n_same_stored), n_identical_to_recomputed_r06_builder=int(n_same_recomputed), n_usable_windows=int(n_usable),
               n_pairs=int(S.pair.nunique()), seconds=time.time() - t0, rows=rows, verdict=("PASS" if n_same_stored == n_same_recomputed == len(rows) else "FAIL -> STOP"))
    json.dump(res, open(f"{R7.CHECKS}/equiv_windows.json", "w"), indent=1)
    print(f"§1.4 EQUIVALENCE: {n_same_stored}/{len(rows)} identical to the stored R06 arrays, {n_same_recomputed}/{len(rows)} identical to the recomputed R06 builder "
          f"({n_usable} windows on product-usable pairs, {S.pair.nunique()} pairs, {time.time()-t0:.0f} s) -> {res['verdict']}", flush=True)
    if res["verdict"] != "PASS": raise SystemExit(2)
