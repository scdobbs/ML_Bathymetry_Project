"""R05 §2.4 driver: refit the frozen bar (bar_v1_1 = bar_v1 code with the eval_v1_1 inner criterion) on tiles_v1.1 /
blocks_v1.1 with features_v1 (R04 table, unchanged; the EX1402L2 xtrack override is already NaN there), evaluate with
eval_v1_1, write BAR_v1_1.md, prediction tables and chosen configurations; EW0207 separately."""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r05_common as C5, r04_common as C, bar_v1_1 as B, eval_v1_1 as E

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tiles_dir", default=f"{C5.ROOT}/tiles"); ap.add_argument("--features", default=f"{C5.R04}/features/features_v1.parquet")
    ap.add_argument("--out", default=f"{C5.ROOT}/bar"); ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4))); ap.add_argument("--grid", default="full"); a = ap.parse_args()
    m = C5.start("run_bar_v1_1"); os.makedirs(a.out, exist_ok=True); units = list(C.UNITS); gi = None if a.grid == "full" else list(range(int(a.grid)))
    tiles = pd.read_parquet(f"{a.tiles_dir}/tiles_v1_1.parquet"); blocks = pd.read_parquet(f"{a.tiles_dir}/blocks_v1_1.parquet"); feats = pd.read_parquet(a.features)
    Bt, Tt = B.assemble(tiles, blocks, feats); E.assert_units(Bt, allow_ew0207=True); E.assert_units(Tt, allow_ew0207=True)
    dev_B, dev_T = Bt[Bt.unit.isin(units)].reset_index(drop=True), Tt[Tt.unit.isin(units)].reset_index(drop=True)
    # §2.2.4 eligible pairs from the TILE table (>= 10 tiles with finite target); same set at both scales. §2.2.5 detection pairs from blocks.
    elig = {tgt: sorted(p for p, v in E.eligible_pairs(dev_T.assign(_p=1.0), "_p", tgt).items() if v["eligible"]) for tgt in ("A_fine", "T_fine", "E_1m")}
    detp = E.detection_pairs(dev_B); det_pairs = sorted(p for p, v in detp.items() if v["detection"])
    print(f"block rows {len(Bt)} (dev {len(dev_B)}), tile rows {len(Tt)} (dev {len(dev_T)}); eligible pairs " + ", ".join(f"{k} {len(v)}/38" for k, v in elig.items()) + f"; detection pairs {len(det_pairs)}/38", flush=True)
    json.dump(dict(eligible=elig, detection_pairs=det_pairs, detection_prevalence={p: v["prevalence"] for p, v in detp.items()}), open(f"{a.out}/pair_rules.json", "w"), indent=1)
    out = {}; md = [f"# BAR_v1_1 — strongest free predictor under eval_v1.1 (R05 §2.4)\n", f"Units: {C.UNITS_VERSION} ({len(units)} units); tiles_v1.1 / blocks_v1.1; training cap {B.CAP} rows per unit per fold (seed {B.SEED}); grid {len(B.GRID)} configurations; inner criterion = eval_v1.1 statistic; E_1m threshold {C.E1M_THRESHOLD_M} m; pair eligibility >= {E.MIN_TILES_PER_PAIR} tiles; detection rule per pair [{E.DETECTION_BOUNDS[0]}, {E.DETECTION_BOUNDS[1]}].\n"]
    for scale, tab in (("tile", dev_T), ("block", dev_B)):
        t0 = time.time(); P, chosen = B.run_scale(tab, scale, units, a.workers, grid_idx=gi, elig_pairs=elig, det_pairs=det_pairs); P.to_parquet(f"{a.out}/pred_{scale}.parquet")
        json.dump(chosen, open(f"{a.out}/chosen_configs_{scale}.json", "w"), indent=1, default=float)
        res = B.evaluate_bar(P, scale, units, elig_pairs=elig, det_pairs=det_pairs, det_prevalence={p: v["prevalence"] for p, v in detp.items()}); out[scale] = res; json.dump(res, open(f"{a.out}/bar_results_{scale}.json", "w"), indent=1, default=float)
        for key in res: md.append(B.md_block(res, key, units))
        cm = B.cap_mask(tab.unit.values); cnt = pd.Series(tab.unit.values[cm]).value_counts().to_dict(); tot = tab.unit.value_counts().to_dict()
        md.append(f"\nTraining rows per unit after the {B.CAP}-row cap ({scale}): " + ", ".join(f"{u} {cnt.get(u, 0)}/{tot.get(u, 0)}" for u in units))
        print(f"[{scale}] done in {time.time()-t0:.0f} s", flush=True)
    ew = {}
    for scale, tab, full in (("tile", dev_T, Tt), ("block", dev_B, Bt)):
        rows = full[full.unit == "lu16"].reset_index(drop=True)
        if len(rows) == 0: continue
        Pe = rows[["unit", "pair", "tile_id", "block_id", "A_fine", "T_fine", "E_1m", "sigma_m_R02"]].copy()
        for k, c in B.REFS.items(): Pe[k] = rows[c].values.astype(float)
        chosen = json.load(open(f"{a.out}/chosen_configs_{scale}.json"))
        for tgt in ("A_fine", "T_fine"):
            for name, cols in (("GBT_ship", B.ALL), ("GBT_noship", B.NOSHIP)):
                cfgs = [v["cfg_idx"] for v in chosen[f"{scale}/{tgt}/{name}"].values()]; ci = max(set(cfgs), key=cfgs.count)
                ok = np.isfinite(tab[tgt].values) & (tab[tgt].values > 0); cmk = B.cap_mask(tab.unit.values[ok])
                mdl = B.make(B.GRID[ci], "reg").fit(tab.loc[ok, cols].values[cmk], np.log10(tab.loc[ok, tgt].values[cmk] + B.EPS))
                Pe[f"{name}_{tgt}"] = 10 ** mdl.predict(rows[cols].values)
        tbl = Pe.assign(unit="lu16"); S = {c: E.score_one(tbl, c, "A_fine", scale, ["lu16"], elig_pairs=["EW0207__MGDS_32556"]) for c in [c for c in Pe.columns if c.startswith(("GBT_", "ref_"))] if "T_fine" not in c}
        ew[scale] = dict(n=len(rows), A_fine=S); Pe.to_parquet(f"{a.out}/pred_{scale}_EW0207.parquet")
        md.append(f"\n### EW0207 (lu16), {scale} scale, A_fine — separate one-unit table ({len(rows)} rows; models refit on the 16 development units)\n\n| predictor | rho | log-log R² | capture20 |\n|---|---|---|---|")
        for c, s in S.items(): r = s["per_unit"]["lu16"]; md.append(f"| {c} | {r.get('rho', np.nan):.3f} | {r.get('r2_loglog', np.nan):.3f} | {r.get('capture20', np.nan):.3f} |")
    json.dump(ew, open(f"{a.out}/bar_results_EW0207.json", "w"), indent=1, default=float)
    open(f"{a.out}/BAR_v1_1.md", "w").write("\n".join(md) + "\n"); print("wrote", f"{a.out}/BAR_v1_1.md", flush=True)
