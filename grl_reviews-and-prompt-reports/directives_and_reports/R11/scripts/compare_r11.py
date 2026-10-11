"""R11 comparisons (every number from persisted files). Usage: python compare_r11.py A1 <pairs> | A1pred <pairs> | A2 | D3
A1: the rehearsal build of each pair against the FROZEN artifacts as in R10 §A2 (tiles_v1.1 keys / r0 / c0 / s_lr / A_fine; F_hr / joint_v1;
    features: every bar_v1_1.ALL column within 1e-6 and the NaN pattern of the 14 contract (product) columns identical; interior flag at D1;
    A_fine_int at D1; the pair_stats product-feature decision).
A1pred: the rehearsal predictions of each pair: finite on every tile for the three candidates (not scored).
A2: the harness build (rows from the acquisition rehearsal supplement) against the build from the manifest v2.1 rows: every item identical;
    EX1202L2 not scorable (no k) with nothing built.
D3: the R11 scoring re-run (per_pair_D1 / D125) against R10b's on OAK: every retained column identical; the §A3 columns listed."""
import glob, json, os, sys
import numpy as np, pandas as pd, rasterio
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "phase4", "scripts"))
import phase4_common as P4, bar_v1_1 as B
R8, R9, RB = P4.R8, P4.R9, P4.R9.RB; ROOT = P4.ROOT["rehearsal"]; HROOT = f"{ROOT}/harness"; OAK10B = f"{R8.OAK_ARCHIVE}/R10b/rehearsal"
def cpu(): return open("/proc/cpuinfo").read().split("model name")[1].split("\n")[0].split(":")[1].strip()
def same_arr(a, b): return bool(a.shape == b.shape and np.array_equal(a, b, equal_nan=True)) if np.issubdtype(np.asarray(a).dtype, np.number) else bool(np.array_equal(a, b))
def frame_identical(a, b, keys):
    a = a.set_index(keys).sort_index(); b = b.set_index(keys).sort_index(); r = dict(keys_identical=bool(a.index.equals(b.index)), n=[len(a), len(b)], columns_identical=bool(list(a.columns) == list(b.columns)))
    if not r["keys_identical"] or not r["columns_identical"]: r["identical"] = False; return r
    bad = [c for c in a.columns if not same_arr(a[c].values, b[c].values)]; r["columns_differing"] = bad; r["identical"] = not bad; return r
def a1(pairs):
    m, root, rec = P4.start("compare_r11.py", "rehearsal", pairs); tiles = R8.load_tiles(); feats = R8.load_features(); TI = pd.read_parquet(f"{R9.OUT_INT}/tiles_int.parquet").set_index("tile_id"); out = {}
    for pid in pairs:
        d = f"{root}/build/{pid}"; t = pd.read_parquet(f"{d}/tiles_v1_1.parquet").set_index("tile_id"); ref = tiles[tiles.pair == pid].set_index("tile_id"); r = {}
        r["tile_keys_identical"] = bool(set(t.index) == set(ref.index)); r["n_tiles"] = [len(t), len(ref)]; ref = ref.reindex(t.index)
        for c in ("r0", "c0", "s_lr", "A_fine"): r[f"{c}_identical"] = bool(np.array_equal(t[c].values, ref[c].values))
        for name in ("F_hr", "joint_v1"):
            with rasterio.open(f"{root}/fields/{pid}/{name}.tif") as a, rasterio.open(f"{R8.R04_FIELDS}/{pid}/{name}.tif") as b: A, Bm = a.read(1), b.read(1); r[f"{name}_identical"] = bool(A.shape == Bm.shape and np.array_equal(A, Bm, equal_nan=True))
        f = pd.read_parquet(f"{d}/features.parquet"); fr = feats[feats.pair == pid]; key = ["tile_id", "block_id"]; f = f.set_index(key); fr = fr.set_index(key).reindex(f.index); cols = [c for c in B.ALL if c in f.columns]
        a = f[cols].values.astype(float); b = fr[cols].values.astype(float); ok = ~np.isnan(a) & ~np.isnan(b); pc = f[B.CONTRACT].values.astype(float); pcr = fr[B.CONTRACT].values.astype(float)
        r["features"] = dict(n_rows=len(f), n_rows_unmatched_in_frozen=int(np.isnan(b).all(axis=1).sum()), n_cols=len(cols), nan_pattern_identical=bool(np.array_equal(np.isnan(a), np.isnan(b))), max_abs_diff=float(np.max(np.abs(a[ok] - b[ok]))) if ok.any() else 0.0,
                             within_1e6=bool((np.max(np.abs(a[ok] - b[ok])) if ok.any() else 0.0) <= 1e-6), product_columns=dict(n=len(B.CONTRACT), nan_pattern_identical=bool(np.array_equal(np.isnan(pc), np.isnan(pcr))), frac_nan_build=float(np.isnan(pc).mean()), frac_nan_frozen=float(np.isnan(pcr).mean())),
                             ship_usable_build=sorted({bool(v) for v in f.ship_usable}), ship_usable_frozen=sorted({bool(v) for v in fr.ship_usable.dropna()}))
        ti = pd.read_parquet(f"{d}/tiles_int.parquet").set_index("tile_id"); l1 = set(pd.read_csv(f"{RB.OUT_INT}/lists/{pid}_D1.csv").tile_id)
        r["interior_D1"] = dict(identical=bool(set(ti[ti.interior_D1].index) == l1), n_build=int(ti.interior_D1.sum()), n_frozen=len(l1))
        ref_int = TI.A_fine_int_D1.reindex(ti.index).values; a2 = ti.A_fine_int_D1.values; both = np.isfinite(a2) & np.isfinite(ref_int)
        r["A_fine_int_D1"] = dict(nan_pattern_identical=bool(np.array_equal(np.isfinite(a2), np.isfinite(ref_int))), max_abs_diff=float(np.max(np.abs(a2[both] - ref_int[both]))) if both.any() else 0.0)
        s = json.load(open(f"{d}/pair_stats.json")); r["pair_stats"] = {k: s.get(k) for k in ("status", "ship_products_usable", "product_features", "contract_columns_frac_nan", "cells_across", "k")}; r["views"] = s["views"]
        row = P4.C4.pair_row(m, pid); r["manifest"] = dict(ship_products_usable=P4.C4.ship_usable(row), reason=str(row.get("ship_products_usable_reason")), flags=str(row.get("products_v2_1_flags")), ruling_source=str(row.get("ruling_source")))
        r["all_identical"] = bool(r["tile_keys_identical"] and all(r[f"{c}_identical"] for c in ("r0", "c0", "s_lr", "A_fine", "F_hr", "joint_v1")) and r["features"]["nan_pattern_identical"] and r["features"]["within_1e6"] and r["features"]["product_columns"]["nan_pattern_identical"] and r["interior_D1"]["identical"] and r["A_fine_int_D1"]["nan_pattern_identical"] and r["A_fine_int_D1"]["max_abs_diff"] == 0.0)
        out[pid] = r; print(pid, json.dumps(r, default=float)[:600], flush=True)
    return out
def a1pred(pairs):
    m, root, rec = P4.start("compare_r11.py", "rehearsal", pairs); out = {}
    for pid in pairs:
        P = pd.read_parquet(f"{root}/predictions/{pid}.parquet"); t = pd.read_parquet(f"{root}/build/{pid}/tiles_v1_1.parquet"); f = pd.read_parquet(f"{root}/build/{pid}/features.parquet"); ft = f[f.block_id < 0]
        r = dict(n_tiles=int(len(P)), n_tiles_build=int(len(t)), tiles_match=bool(set(P.tile_id) == set(t.tile_id)), product_columns_all_nan=bool(np.isnan(ft[B.CONTRACT].values.astype(float)).all()))
        for c in P4.CANDIDATES: v = P[c].values.astype(float); r[c] = dict(all_finite=bool(np.isfinite(v).all()), all_positive=bool((v > 0).all()), min=float(v.min()), median=float(np.median(v)), max=float(v.max()))
        r["all_finite"] = all(r[c]["all_finite"] for c in P4.CANDIDATES); out[pid] = r; print(pid, json.dumps(r), flush=True)
    return out
def a2():
    m, root, rec = P4.start("compare_r11.py", "rehearsal", ["TN399__MGDS_33068", "FK151121__MGDS_22383"]); out = {}
    for pid in ("TN399__MGDS_33068", "FK151121__MGDS_22383"):
        a, b = f"{root}/build/{pid}", f"{HROOT}/build/{pid}"; r = {}
        for name, keys in (("tiles_v1_1.parquet", ["tile_id"]), ("tiles_v1.parquet", ["window_id"] if "window_id" in pd.read_parquet(f"{a}/tiles_v1.parquet").columns else ["tile_id"]), ("blocks_v1_1.parquet", ["tile_id", "br0", "bc0"]), ("features.parquet", ["tile_id", "block_id"]), ("tiles_int.parquet", ["tile_id"])):
            r[name] = frame_identical(pd.read_parquet(f"{a}/{name}"), pd.read_parquet(f"{b}/{name}"), keys)
        for name in ("F_hr", "F_res", "bicubic_m", "joint_v1"):
            with rasterio.open(f"{root}/fields/{pid}/{name}.tif") as x, rasterio.open(f"{HROOT}/fields/{pid}/{name}.tif") as y: r[f"{name}.tif"] = dict(identical=bool(x.shape == y.shape and x.transform == y.transform and np.array_equal(x.read(1), y.read(1), equal_nan=True)))
        for name in ("x.npy", "lrv.npy"): r[f"inputs/{name}"] = dict(identical=same_arr(np.load(f"{root}/inputs/{pid}/{name}", mmap_mode="r")[:], np.load(f"{HROOT}/inputs/{pid}/{name}", mmap_mode="r")[:]))
        r["inputs/meta.parquet"] = frame_identical(pd.read_parquet(f"{root}/inputs/{pid}/meta.parquet"), pd.read_parquet(f"{HROOT}/inputs/{pid}/meta.parquet"), ["tile_id"])
        def strip(o):                                              # wall-clock timings (seconds) are not build items
            return {k: strip(v) for k, v in o.items() if k not in ("seconds", "k_provenance")} if isinstance(o, dict) else o
        s, sh = strip(json.load(open(f"{a}/pair_stats.json"))), strip(json.load(open(f"{b}/pair_stats.json"))); r["pair_stats"] = dict(identical=bool(s == sh), differing=[k for k in s if s.get(k) != sh.get(k)], excluded="seconds (timings), k_provenance (rehearsal: no acquisition record in the plain build)")
        r["all_identical"] = all(v["identical"] for v in r.values()); out[pid] = r; print(pid, "all identical" if r["all_identical"] else json.dumps({k: v for k, v in r.items() if isinstance(v, dict) and not v["identical"]}, default=str)[:1200], flush=True)
    ex = "EX1202L2__MGDS_20815"; st = json.load(open(f"{HROOT}/build/{ex}/pair_stats.json")); files = sorted(os.listdir(f"{HROOT}/build/{ex}"))
    out[ex] = dict(status=st.get("status"), files_built=files, fields_dir=os.path.exists(f"{HROOT}/fields/{ex}"), inputs_dir=os.path.exists(f"{HROOT}/inputs/{ex}"), passed=bool(st.get("status") == P4.NO_K and files == ["pair_stats.json"] and not os.path.exists(f"{HROOT}/fields/{ex}") and not os.path.exists(f"{HROOT}/inputs/{ex}")), k_provenance=st.get("k_provenance"))
    print(ex, out[ex]); return out
def d3():
    pairs = list(P4.C4.DEV_PAIRS) + [R8.EW_PAIR]; m, root, rec = P4.start("compare_r11.py", "rehearsal", pairs); out = {}
    for tag in ("D1", "D125"):
        rel = f"scores/per_pair_{tag}.csv"; R8.oak_verify("R10b/rehearsal", rel, f"{OAK10B}/{rel}"); R8.oak_verify("R10b/rehearsal", f"score_inputs/inputs_{tag}.parquet", f"/scratch/users/scdobbs/grl_review/R10b/rehearsal/score_inputs/inputs_{tag}.parquet"); a = pd.read_csv(f"{root}/scores/per_pair_{tag}.csv").set_index("pair"); b = pd.read_csv(f"{OAK10B}/{rel}").set_index("pair").reindex(a.index); cols = [c for c in b.columns if c in a.columns]; d = {}
        for c in cols:
            x, y = a[c], b[c]
            if np.issubdtype(x.dtype, np.number) and np.issubdtype(y.dtype, np.number): xx, yy = x.values.astype(float), y.values.astype(float); same_nan = np.array_equal(np.isnan(xx), np.isnan(yy)); ok = np.isfinite(xx) & np.isfinite(yy); d[c] = float(np.max(np.abs(xx[ok] - yy[ok]))) if ok.any() and same_nan else (0.0 if same_nan else float("nan"))
            else: d[c] = 0.0 if bool((x.fillna("").astype(str) == y.fillna("").astype(str)).all()) else float("nan")
        out[tag] = dict(n_columns=len(cols), max_abs_diff_per_column=d, all_identical=bool(all(np.isfinite(v) and v == 0 for v in d.values())), columns_dropped=[c for c in b.columns if c not in a.columns], columns_added=[c for c in a.columns if c not in b.columns], rows=[len(a), len(b)], reference=dict(path=f"{OAK10B}/{rel}"))
        print(tag, out[tag]["n_columns"], out[tag]["all_identical"], out[tag]["columns_added"], flush=True)
    return out
if __name__ == "__main__":
    what = sys.argv[1]; res = dict(node=os.uname().nodename, cpu=cpu(), what=what)
    res["result"] = {"A1": lambda: a1(P4.split_pairs(sys.argv[2])), "A1pred": lambda: a1pred(P4.split_pairs(sys.argv[2])), "A2": a2, "D3": d3}[what]()
    os.makedirs(f"{ROOT}/compare", exist_ok=True); json.dump(res, open(f"{ROOT}/compare/compare_r11_{what}.json", "w"), indent=1, default=float); print("wrote", f"{ROOT}/compare/compare_r11_{what}.json")
