"""R10b §B2 and §B4 comparisons against the R10 outputs on OAK (sha256-verified).
B2: the R10b rehearsal build / figures of KM0923, AR26, EW0207 against R10's: tiles_v1_1 (keys, r0, c0, s_lr, A_fine), F_hr / joint_v1
    mosaics, tiles_int (interior flag, A_fine_int), features (<= 1e-6 absolute, NaN pattern), pair_stats views, set-A pages pixel-identical.
B4: per_pair_D1.csv / per_pair_D125.csv of the R10b scoring against R10's for every column R10 reported that still exists: max abs difference per column.
Writes compare_r10b.json."""
import json, os, sys
import numpy as np, pandas as pd, rasterio
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "phase4", "scripts"))
import phase4_common as P4, bar_v1_1 as B
R8 = P4.R8; ROOT = P4.ROOT["rehearsal"]; OAK = f"{P4.R10_OAK}/rehearsal"; PAIRS = ["KM0923__MGDS_31429", "AR26__MGDS_31838", "EW0207__MGDS_32556"]
def v(rel): R8.oak_verify("R10/rehearsal", rel, f"{OAK}/{rel}"); return f"{OAK}/{rel}"
def cpu(): return open("/proc/cpuinfo").read().split("model name")[1].split("\n")[0].split(":")[1].strip()
if __name__ == "__main__":
    what = sys.argv[1]; out = {"node": os.uname().nodename, "cpu": cpu()}
    if what in ("B2", "both"):
        b2 = {}
        for pid in PAIRS:
            r = {}; t = pd.read_parquet(f"{ROOT}/build/{pid}/tiles_v1_1.parquet").set_index("tile_id"); t0 = pd.read_parquet(v(f"build/{pid}/tiles_v1_1.parquet")).set_index("tile_id")
            r["tile_keys_identical"] = bool(set(t.index) == set(t0.index)); t0 = t0.reindex(t.index)
            for c in ("r0", "c0", "s_lr", "A_fine"): r[f"{c}_identical"] = bool(np.array_equal(t[c].values, t0[c].values))
            for name in ("F_hr", "joint_v1"):
                with rasterio.open(f"{ROOT}/fields/{pid}/{name}.tif") as a, rasterio.open(v(f"fields/{pid}/{name}.tif")) as b: A, Bm = a.read(1), b.read(1); r[f"{name}_identical"] = bool(np.array_equal(A, Bm, equal_nan=True))
            ti = pd.read_parquet(f"{ROOT}/build/{pid}/tiles_int.parquet").set_index("tile_id"); ti0 = pd.read_parquet(v(f"build/{pid}/tiles_int.parquet")).set_index("tile_id").reindex(ti.index)
            r["interior_D1_identical"] = bool(np.array_equal(ti.interior_D1.values, ti0.interior_D1.values)); r["A_fine_int_D1_identical"] = bool(np.array_equal(ti.A_fine_int_D1.values, ti0.A_fine_int_D1.values, equal_nan=True))
            f = pd.read_parquet(f"{ROOT}/build/{pid}/features.parquet").set_index(["tile_id", "block_id"]); f0 = pd.read_parquet(v(f"build/{pid}/features.parquet")).set_index(["tile_id", "block_id"]).reindex(f.index); a, b = f[B.ALL].values.astype(float), f0[B.ALL].values.astype(float); ok = np.isfinite(a) & np.isfinite(b)
            r["features"] = dict(nan_pattern_identical=bool(np.array_equal(np.isnan(a), np.isnan(b))), max_abs_diff=float(np.max(np.abs(a[ok] - b[ok]))), within_tol=bool(np.max(np.abs(a[ok] - b[ok])) <= 1e-6))
            s, s0 = json.load(open(f"{ROOT}/build/{pid}/pair_stats.json")), json.load(open(v(f"build/{pid}/pair_stats.json"))); r["views_identical"] = bool(s["views"] == s0["views"]); r["cells_across"] = s.get("cells_across"); r["thinner_than_development"] = s.get("thinner_than_development")
            b2[pid] = r
        k = {x["pair"]: x["code"] for x in json.load(open(f"{ROOT}/figures/key/figset_A_key.json"))["order"]}; k0 = {x["pair"]: x["code"] for x in json.load(open(v("figures/key/figset_A_key.json")))["order"]}
        for pid in PAIRS:
            a = plt.imread(f"{ROOT}/figures/work/pages/A_{k[pid]}.png"); b = plt.imread(v(f"figures/work/pages/A_{k0[pid]}.png")); b2[pid]["page_identical"] = bool(a.shape == b.shape and np.array_equal(a, b)); b2[pid]["codes"] = [k[pid], k0[pid]]
        out["B2"] = b2
    if what in ("B4", "both"):
        b4 = {}
        for tag in ("D1", "D125"):
            a = pd.read_csv(f"{ROOT}/scores/per_pair_{tag}.csv").set_index("pair"); b = pd.read_csv(v(f"scores/per_pair_{tag}.csv")).set_index("pair").reindex(a.index); cols = [c for c in b.columns if c in a.columns and c != "rho_s_lr_W64"]; d = {}
            for c in cols:
                x, y = a[c], b[c]
                if np.issubdtype(x.dtype, np.number) and np.issubdtype(y.dtype, np.number): xx, yy = x.values.astype(float), y.values.astype(float); same_nan = np.array_equal(np.isnan(xx), np.isnan(yy)); ok = np.isfinite(xx) & np.isfinite(yy); d[c] = float(np.max(np.abs(xx[ok] - yy[ok]))) if ok.any() and same_nan else (0.0 if same_nan else float("nan"))
                else: d[c] = 0.0 if bool((x.fillna("").astype(str) == y.fillna("").astype(str)).all()) else float("nan")
            b4[tag] = dict(n_columns=len(cols), max_abs_diff_per_column=d, all_identical=bool(all(np.isfinite(v) and v == 0 for v in d.values())), columns_dropped=[c for c in b.columns if c not in a.columns], columns_added=[c for c in a.columns if c not in b.columns])
        out["B4"] = b4
    json.dump(out, open(f"{ROOT}/compare_r10b_{what}.json", "w"), indent=1, default=float); print(json.dumps(out, indent=1, default=float)[:2500])
