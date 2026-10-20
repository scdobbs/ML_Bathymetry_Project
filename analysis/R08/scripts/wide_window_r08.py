"""R08 §6 — wide-window reference rows (no fitting). For every tiles_v1.1 tile the 64 x 64-cell window of the ACQ-R05
ctx_lr.tif centred on the tile centre (cells outside the context grid invalid; windows with < 50 % valid cells -> NaN),
passed to the FROZEN features_v1.native_features (R04, hash-checked) exactly as the 128-px ship window is: s_lr (SD of
valid cells floored at 0.5 m), C (roughness on the x8 bicubic), shat_fine (spectral_extrap), beta. Each pair's context
files are verified against the pair's SHA256SUMS and the context_v1.parquet hashes first. Leak test: 100 tiles recomputed
through a ShipOnlyView whose HR keys hold noise; the rows must be byte-identical. Scored with eval_v1_1 on 16 and 15 units
next to the frozen 128-px rows (s_lr_128, C_128, shat_fine_128, beta_128), ref_s_lr (s_lr_256), ref_C (C_hr_128), GBT_ship, V2."""
from __future__ import annotations
import json, os, sys, time
import numpy as np, pandas as pd, rasterio
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r08_common as R8, r04_common as C4, eval_v1_1 as E
import features_v1 as F          # R04 frozen feature code (sha256 checked in r08_common.start)

ROWS = ["s_lr", "C", "shat_fine", "beta"]; EXTRA = ["std_ship", "logA", "fit_r2", "noise_excess", "aniso_spec", "n_fit_bins", "curv", "aniso", "slope", "mu_lr"]
W = R8.W_CELLS

def load_ctx(row):
    with rasterio.open(row["ctx_lr_path"]) as s:
        lr = s.read(1).astype(np.float32); tr = s.transform; crs = s.crs; nod = s.nodata
        if nod is not None and np.isfinite(nod): lr = np.where(lr == nod, np.nan, lr)
    with rasterio.open(row["ctx_lr_valid_path"]) as s:
        v = s.read(1); assert s.transform == tr and s.shape == lr.shape
    return dict(ctx_lr=lr, ctx_valid=(v == 1) & np.isfinite(lr), ctx_tr=tr, ctx_crs=crs, cell_m=float(row["cell_m"]), k=None)

def window_rows(ctx, x, y):
    """The four rows (plus extras) for the W x W window centred on map point (x, y) in the context CRS."""
    r, c = rasterio.transform.rowcol(ctx["ctx_tr"], x, y); r1 = int(round(r - W / 2.0)); c1 = int(round(c - W / 2.0))
    H, Wd = ctx["ctx_lr"].shape; lr = np.full((W, W), np.nan, np.float32); lv = np.zeros((W, W), bool)
    rs, re_, cs, ce = max(0, r1), min(H, r1 + W), max(0, c1), min(Wd, c1 + W)
    if re_ > rs and ce > cs:
        lr[rs - r1:re_ - r1, cs - c1:ce - c1] = ctx["ctx_lr"][rs:re_, cs:ce]; lv[rs - r1:re_ - r1, cs - c1:ce - c1] = ctx["ctx_valid"][rs:re_, cs:ce]
    lv &= np.isfinite(lr); vf = float(lv.sum() / (W * W))
    out = {k: np.nan for k in ROWS + EXTRA}; out["valid_frac"] = vf; out["n_valid"] = int(lv.sum())
    if vf < R8.W_MIN_VALID: return out
    f = F.native_features(np.where(lv, lr, np.nan), lv, W, vf, ctx["k"], ctx["cell_m"])
    for k in ROWS + EXTRA: out[k] = f.get(k, np.nan)
    return out

def one_pair(args):
    pid, crow = args; crow = dict(crow); ver = R8.verify_context_pair(crow); ctx = load_ctx(crow)
    t = D["tiles"][D["tiles"].pair == pid]; ctx["k"] = int(t.k.iloc[0]); assert t.k.nunique() == 1
    xs = ((t.left + t.right) / 2.0).values; ys = ((t.top + t.bottom) / 2.0).values; tcrs = rasterio.crs.CRS.from_string(t.crs.iloc[0])
    same = (tcrs == ctx["ctx_crs"])
    if not same:
        from pyproj import Transformer; xs, ys = Transformer.from_crs(tcrs, ctx["ctx_crs"], always_xy=True).transform(xs, ys)
    view = F.ShipOnlyView(**ctx, hr_arr=None, hrv_arr=None, joint=None, paths=None)
    rows = []
    for tid, x, y in zip(t.tile_id.values, xs, ys): rows.append(dict(tile_id=tid, pair=pid, unit=C4.UNIT_OF[pid], **window_rows(view, x, y)))
    return pd.DataFrame(rows), dict(pair=pid, unit=C4.UNIT_OF[pid], crs_equal=bool(same), tile_crs=str(tcrs), ctx_crs=str(ctx["ctx_crs"]), cell_m=ctx["cell_m"], k=ctx["k"], ctx_shape=list(ctx["ctx_lr"].shape), ctx_disagrees=bool(crow["ctx_disagrees"]), verification=ver)

def leak_test(ctxinfo, tiles, n=R8.LEAK_N, seed=0):
    rng = np.random.default_rng(seed); st = tiles.sample(n=n, random_state=seed); res = []
    for pid, g in st.groupby("pair"):
        crow = dict(ctxinfo[pid]); ctx = load_ctx(crow); ctx["k"] = int(g.k.iloc[0]); tcrs = rasterio.crs.CRS.from_string(g.crs.iloc[0])
        xs = ((g.left + g.right) / 2.0).values; ys = ((g.top + g.bottom) / 2.0).values
        if tcrs != ctx["ctx_crs"]:
            from pyproj import Transformer; xs, ys = Transformer.from_crs(tcrs, ctx["ctx_crs"], always_xy=True).transform(xs, ys)
        v1 = F.ShipOnlyView(**ctx, hr_arr=None, hrv_arr=None, joint=None, paths=None)
        noise = rng.standard_normal(ctx["ctx_lr"].shape).astype(np.float32) * 100 - 2000
        v2 = F.ShipOnlyView(**ctx, hr_arr=noise, hrv_arr=rng.random(ctx["ctx_lr"].shape) > 0.3, joint=(rng.random(ctx["ctx_lr"].shape) > 0.3).astype(np.uint8), paths="noise")
        for tid, x, y in zip(g.tile_id.values, xs, ys):
            a = window_rows(v1, x, y); b = window_rows(v2, x, y)
            ident = F.feature_hash(a) == F.feature_hash(b) and all((np.isnan(a[k]) and np.isnan(b[k])) or a[k] == b[k] for k in a)
            res.append(dict(pair=pid, tile_id=tid, identical=bool(ident)))
    return res

D = {}
if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1"); workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 8)); t0 = time.time()
    m = R8.start("wide_window_r08"); os.makedirs(R8.OUT_WIDE, exist_ok=True); units = list(C4.UNITS)
    tiles = R8.load_tiles(); D["tiles"] = tiles; ctxdf = R8.load_context(); ctxinfo = {r["pair_id"]: r for r in ctxdf.to_dict("records")}
    pairs = list(tiles.pair.unique()); C4.assert_no_sealed(pairs); assert set(pairs) <= set(ctxinfo), set(pairs) - set(ctxinfo)
    with Pool(min(workers, len(pairs))) as pool: outs = pool.map(one_pair, [(p, ctxinfo[p]) for p in pairs])
    rows = pd.concat([o[0] for o in outs], ignore_index=True); info = [o[1] for o in outs]
    rows = rows.rename(columns={k: f"{k}_W64" for k in ROWS + EXTRA + ["valid_frac", "n_valid"]}); rows.to_parquet(f"{R8.OUT_WIDE}/rows_W64.parquet")
    R8.dump(dict(W_cells=W, min_valid_frac=R8.W_MIN_VALID, pairs=info, seconds=time.time() - t0), f"{R8.OUT_WIDE}/pairs_info.json")
    print(f"rows computed for {len(rows)} tiles of {len(pairs)} pairs; crs equal on {sum(i['crs_equal'] for i in info)}/{len(info)} ({time.time()-t0:.0f} s)", flush=True)
    # ---- leak test
    lt = leak_test(ctxinfo, tiles); n_ok = sum(r["identical"] for r in lt)
    R8.dump(dict(n=len(lt), n_identical=n_ok, rows=lt), f"{R8.OUT_WIDE}/leak_test.json"); print(f"LEAK TEST: {n_ok}/{len(lt)} identical with the HR keys replaced by noise", flush=True)
    if n_ok != len(lt): raise SystemExit("LEAK TEST FAILED")
    # ---- defined fraction per unit (incl. EW0207)
    rows["unit"] = rows.pair.map(C4.UNIT_OF); defn = rows.groupby("unit").agg(n_tiles=("tile_id", "size"), **{f"{k}_W64": (f"{k}_W64", lambda s: float(np.isfinite(s).mean())) for k in ROWS}).reset_index()
    defn.to_csv(f"{R8.OUT_WIDE}/defined_fraction.csv", index=False)
    # ---- scoring on the development table
    P, rules = R8.load_pred_tile(); EP = rules["eligible"]["A_fine"]; feats = R8.load_features(); ft = feats[feats.block_id < 0].set_index("tile_id")
    P = P.merge(rows[["tile_id"] + [f"{k}_W64" for k in ROWS]], on="tile_id", how="left")
    for k in ("s_lr_128", "C_128", "shat_fine_128", "beta_128"): P[k] = ft[k].reindex(P.tile_id).values
    a, b = P.s_lr_128.values, P.ref_shipstd128.values; assert np.array_equal(np.isnan(a), np.isnan(b)) and np.allclose(a[~np.isnan(a)], b[~np.isnan(b)], rtol=0, atol=0), "s_lr_128 != ref_shipstd128"
    P.to_parquet(f"{R8.OUT_WIDE}/pred_tile_wide.parquet")
    PAIRS = [("s_lr_W64", "s_lr_128"), ("C_W64", "C_128"), ("shat_fine_W64", "shat_fine_128"), ("beta_W64", "beta_128")]
    CTX = ["ref_s_lr", "ref_C", "GBT_ship_A_fine", "V2_seedavg"]; cols = [c for pr in PAIRS for c in pr] + CTX
    S = {c: E.score_one(P, c, "A_fine", "tile", units, elig_pairs=EP) for c in cols}
    out = dict(rows=ROWS, pairs=PAIRS, context_columns=CTX, scores_per_unit={c: {u: S[c]["per_unit"][u].get("rho", np.nan) for u in units} for c in cols}, median_ci={}, comparisons={})
    for name, uu in (("all16", list(R8.UNITS16)), ("fifteen", list(R8.UNITS15))):
        out["median_ci"][name] = {c: E.median_ci(P, c, "A_fine", "tile", uu, metric="rho", elig_pairs=EP) for c in cols}
        out["comparisons"][name] = {f"{w}-{n}": E.compare(P, w, n, "A_fine", "tile", uu, metric="rho", elig_pairs=EP) for w, n in PAIRS}
        out["comparisons"][name].update({f"{w}-{c}": E.compare(P, w, c, "A_fine", "tile", uu, metric="rho", elig_pairs=EP) for w, _ in PAIRS for c in ("ref_s_lr", "GBT_ship_A_fine", "V2_seedavg")})
        print(f"  [{name}] " + "; ".join(f"{c} {out['median_ci'][name][c]['value']:.3f}" for c in cols) + " | " + "; ".join(f"{k} {v['diff_median']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] {v['units_a_better']}/{v['units_b_better']} p={v['sign_test_p_one_sided']:.3f}" for k, v in out["comparisons"][name].items() if k.split("-")[1].endswith("_128")), flush=True)
    R8.dump(out, f"{R8.OUT_WIDE}/scores.json"); print(f"done ({time.time()-t0:.0f} s)")
