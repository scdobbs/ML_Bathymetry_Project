"""R04 §3 driver: assemble block / tile tables, run bar_v1 at both scales, evaluate with eval_v1, write BAR_v1.md,
prediction tables and chosen configurations. EW0207 (lu16) is scored separately through the same code where models exist
(the GBTs are trained on the 16 development units only; EW0207 rows are predicted by the all-16 refit per model)."""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r04_common as C, bar_v1 as B, eval_v1 as E

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tiles_dir", default=f"{C.ROOT}/tiles"); ap.add_argument("--features", default=f"{C.ROOT}/features/features_v1.parquet")
    ap.add_argument("--out", default=f"{C.ROOT}/bar"); ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4))); ap.add_argument("--grid", default="full"); a = ap.parse_args()
    m = C.start("run_bar"); os.makedirs(a.out, exist_ok=True); units = list(C.UNITS); gi = None if a.grid == "full" else list(range(int(a.grid)))
    tiles = pd.read_parquet(f"{a.tiles_dir}/tiles_v1.parquet"); blocks = pd.read_parquet(f"{a.tiles_dir}/blocks_v1.parquet"); feats = pd.read_parquet(a.features)
    Bt, Tt = B.assemble(tiles, blocks, feats); E.assert_units(Bt, allow_ew0207=True); E.assert_units(Tt, allow_ew0207=True)
    dev_B, dev_T = Bt[Bt.unit.isin(units)].reset_index(drop=True), Tt[Tt.unit.isin(units)].reset_index(drop=True)
    print(f"block rows {len(Bt)} (dev {len(dev_B)}), tile rows {len(Tt)} (dev {len(dev_T)}); features {len(B.ALL)} (contract {len(B.CONTRACT)})", flush=True)
    out = {}; md = [f"# BAR_v1 — strongest free predictor at block and tile scale (R04 §3)\n", f"Units: {C.UNITS_VERSION} ({len(units)} units); training cap {B.CAP} rows per unit per fold (seed {B.SEED}); grid {len(B.GRID)} configurations; E_1m threshold {C.E1M_THRESHOLD_M} m.\n"]
    for scale, tab in (("block", dev_B), ("tile", dev_T)):
        t0 = time.time(); P, chosen = B.run_scale(tab, scale, units, a.workers, grid_idx=gi); P.to_parquet(f"{a.out}/pred_{scale}.parquet")
        json.dump(chosen, open(f"{a.out}/chosen_configs_{scale}.json", "w"), indent=1, default=float)
        res = B.evaluate_bar(P, scale, units); out[scale] = res; json.dump(res, open(f"{a.out}/bar_results_{scale}.json", "w"), indent=1, default=float)
        for key in res: md.append(B.md_block(res, key, units))
        # per-unit training-row counts used (cap)
        cm = B.cap_mask(tab.unit.values); cnt = pd.Series(tab.unit.values[cm]).value_counts().to_dict(); tot = tab.unit.value_counts().to_dict()
        md.append(f"\nTraining rows per unit after the {B.CAP}-row cap ({scale}): " + ", ".join(f"{u} {cnt.get(u, 0)}/{tot.get(u, 0)}" for u in units))
        print(f"[{scale}] done in {time.time()-t0:.0f} s", flush=True)
    # EW0207 separately: refit each model on all 16 units (capped) with the config most often chosen, predict lu16 rows
    ew = {}
    for scale, tab, full in (("block", dev_B, Bt), ("tile", dev_T, Tt)):
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
        tbl = Pe.assign(unit="lu16"); S = {c: E.score_one(tbl, c, "A_fine", scale, ["lu16"]) for c in [c for c in Pe.columns if c.startswith(("GBT_", "ref_"))] if "T_fine" not in c}
        ew[scale] = dict(n=len(rows), A_fine=S); Pe.to_parquet(f"{a.out}/pred_{scale}_EW0207.parquet")
        md.append(f"\n### EW0207 (lu16), {scale} scale, A_fine — separate one-unit table ({len(rows)} rows; models refit on the 16 development units)\n\n| predictor | rho | log-log R² | capture20 |\n|---|---|---|---|")
        for c, s in S.items(): r = s["per_unit"]["lu16"]; md.append(f"| {c} | {r['rho']:.3f} | {r.get('r2_loglog', np.nan):.3f} | {r['capture20']:.3f} |")
    json.dump(ew, open(f"{a.out}/bar_results_EW0207.json", "w"), indent=1, default=float)
    open(f"{a.out}/BAR_v1.md", "w").write("\n".join(md) + "\n"); print("wrote", f"{a.out}/BAR_v1.md", flush=True)
