"""R06 §3 scoring and §4 diagnostic. Every variant's leave-one-unit-out predictions are added as columns to the R05 bar
prediction tables (tile and block; A_fine units, metres: block = 10^pred, tile = RMS over the 16 blocks) and scored ONLY
through the frozen eval_v1_1 (score_one / compare / median_ci) with the R05 pair rules (pair_rules.json).
  §3.1 every endpoint at tile and block scale;  §3.2 each variant vs GBT_ship, ref_s_lr, ref_C on A_fine/tile (and block);
  §3.3 V1 - V0, V2 - V1, V1 - V3;  §3.4 native 7 vs R02 sigma_m;  §3.5 EW0207 one-unit tables;  §3.6 per-unit table;
  §2 seed check (V1 x 3 seeds, separately and seed-averaged = mean of the three log10 block predictions);
  §4 dist_edge / frac_valid reference rows and within-pair partial Spearman rho given both, per unit, low-amplitude vs other.
Writes scores/R06_scores.{md,json}, per_unit_table.csv, diagnostic_lowamp.json, compute_summary.json."""
from __future__ import annotations
import glob, json, os, sys, time
import numpy as np, pandas as pd
from scipy.stats import rankdata, pearsonr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r06_common as R, r04_common as C4, eval_v1_1 as E

EW_PAIR = C4.EW0207["lu16"][0]
CTX = ["GBT_ship_A_fine", "GBT_noship_A_fine", "GBT_ship_T_fine", "GBT_noship_T_fine", "ref_C", "ref_s_lr", "ref_shipstd128", "ref_rsd128", "sigma_m_R02"]
NAT = [u for u in C4.UNITS if u in C4.NATIVE7]
MAIN = {"V0": "V0", "V1": f"V1_s{R.SEED_MAIN}", "V2": "V2", "V3": "V3"}

def load_preds(variant, seed, folds):
    dfs = []
    for f in folds:
        p = f"{R.MODELS}/{variant}/seed{seed}/fold_{f}/pred.parquet"
        if not os.path.exists(p): raise SystemExit(f"missing prediction: {p}")
        d = pd.read_parquet(p); d["fold"] = f
        if f != "ALL": assert set(d.unit) == {f}, (f, set(d.unit))
        dfs.append(d)
    d = pd.concat(dfs, ignore_index=True); assert not d.tile_id.duplicated().any(); return d

def blk_cols(): return [f"blk{j}" for j in range(R.NB * R.NB)]

def seed_average(dfs):
    base = dfs[0][["tile_id", "unit", "pair"]].copy(); M = np.mean([d.set_index("tile_id").loc[base.tile_id, blk_cols()].values for d in dfs], axis=0)
    for j, c in enumerate(blk_cols()): base[c] = M[:, j]
    base["tile_pred_m"] = np.sqrt(np.mean(10 ** (2 * M), axis=1)); return base

def add_pred(Pt, Pb, name, df):
    """Tile column = tile_pred_m; block column = 10^blk{block_id}. Every tile / kept block of the table must be covered."""
    t = df.set_index("tile_id"); Pt[name] = t.tile_pred_m.reindex(Pt.tile_id).values
    L = t[blk_cols()].stack().reset_index(); L.columns = ["tile_id", "blk", "v"]; L["block_id"] = L.blk.str[3:].astype(int)
    key = pd.MultiIndex.from_frame(Pb[["tile_id", "block_id"]]); Pb[name] = (10 ** L.set_index(["tile_id", "block_id"]).v).reindex(key).values
    assert np.isfinite(Pt[name]).all() and np.isfinite(Pb[name]).all(), name

def partial_rho(p, y, z1, z2):
    ok = np.isfinite(p) & np.isfinite(y) & np.isfinite(z1) & np.isfinite(z2)
    if ok.sum() < 5: return np.nan
    r = [rankdata(v[ok]) for v in (p, y, z1, z2)]; Z = np.c_[np.ones(ok.sum()), r[2], r[3]]
    res = [v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0] for v in (r[0], r[1])]
    if np.std(res[0]) == 0 or np.std(res[1]) == 0: return np.nan
    return float(pearsonr(res[0], res[1])[0])

def partial_by_unit(T, pred, elig):
    out = {}
    for u, du in T.groupby("unit"):
        vals = [partial_rho(s[pred].values.astype(float), s.A_fine.values.astype(float), s.dist_edge_128.values.astype(float), s.frac_valid_128.values.astype(float)) for pr, s in du.groupby("pair") if pr in elig]
        out[u] = float(np.nanmedian(vals)) if len(vals) and np.isfinite(vals).any() else np.nan
    return out

def f3(x): return "nan" if x is None or not np.isfinite(x) else f"{x:+.3f}"

if __name__ == "__main__":
    if os.environ.get("R06_MODELS"): R.MODELS = os.environ["R06_MODELS"]; R.SCORES = os.environ.get("R06_SCORES", R.SCORES)   # self-test on synthetic predictions only
    m = R.start("score_r06"); os.makedirs(R.SCORES, exist_ok=True); t0 = time.time(); units = list(C4.UNITS)
    Pt = pd.read_parquet(f"{R.BAR}/pred_tile.parquet"); Pb = pd.read_parquet(f"{R.BAR}/pred_block.parquet"); rules = json.load(open(f"{R.BAR}/pair_rules.json"))
    EP = rules["eligible"]; DET = rules["detection_pairs"]; E.assert_units(Pt); E.assert_units(Pb)
    preds = {"V0": load_preds("V0", R.SEED_MAIN, units), "V2": load_preds("V2", R.SEED_MAIN, units), "V3": load_preds("V3", R.SEED_MAIN, units)}
    for s in R.SEEDS_V1: preds[f"V1_s{s}"] = load_preds("V1", s, units)
    preds["V1_seedavg"] = seed_average([preds[f"V1_s{s}"] for s in R.SEEDS_V1])
    for name in R.PRED_COLS: add_pred(Pt, Pb, name, preds[name])
    Pt.to_parquet(f"{R.SCORES}/pred_tile_r06.parquet"); Pb.to_parquet(f"{R.SCORES}/pred_block_r06.parquet")
    OUT = {}; md = [f"# R06 round-1 scores (eval_v1.1, tiles_v1.1 / blocks_v1.1, R05 pair rules)\n", f"Units: {C4.UNITS_VERSION} ({len(units)}); eligible pairs A_fine {len(EP['A_fine'])}/38; detection pairs {len(DET)}/38; variant columns: {', '.join(R.PRED_COLS)} (A_fine in metres: block 10^pred, tile RMS over 16 blocks).\n"]
    # ---- §3.1 every endpoint
    S = {}
    for tgt, scales in E.TARGETS.items():
        for sc in scales:
            tab = Pt if sc == "tile" else Pb; cols = R.PRED_COLS + [c for c in CTX if c in tab.columns and (c.startswith("ref_") or c == "sigma_m_R02" or tgt in c)]
            if tgt == "E_1m": cols = R.PRED_COLS + [c for c in ("ref_C", "ref_s_lr", "ref_shipstd128", "ref_rsd128", "sigma_m_R02", "GBT_ship_E_1m", "GBT_noship_E_1m") if c in tab.columns]
            S[f"{tgt}/{sc}"] = {c: E.score_one(tab, c, tgt, sc, units, EP[tgt], DET) for c in cols}
            md.append(f"\n## {tgt}/{sc}\n\n" + E.md_scores(S[f"{tgt}/{sc}"], units))
    OUT["scores"] = S
    # ---- §3.2-3.4 comparisons on A_fine at tile (primary) and block (secondary)
    CI = {}; CMP = {}
    for sc, tab in (("tile", Pt), ("block", Pb)):
        CI[sc] = {c: E.median_ci(tab, c, "A_fine", sc, units, metric="rho", elig_pairs=EP["A_fine"]) for c in R.PRED_COLS + ["GBT_ship_A_fine", "ref_s_lr", "ref_C"]}
        cmps = []
        for c in R.PRED_COLS:
            for ref in ("GBT_ship_A_fine", "ref_s_lr", "ref_C"):
                for mtr in ("rho", "capture20", "r2_loglog"): cmps.append(E.compare(tab, c, ref, "A_fine", sc, units, metric=mtr, elig_pairs=EP["A_fine"]))
        for a, b in (("V1", "V0"), ("V2", "V1"), ("V1", "V3")):
            for mtr in ("rho", "capture20", "r2_loglog"):
                cmps.append(E.compare(tab, MAIN[a], MAIN[b], "A_fine", sc, units, metric=mtr, elig_pairs=EP["A_fine"]))
                cmps.append(E.compare(tab, "V1_seedavg" if a == "V1" else MAIN[a], "V1_seedavg" if b == "V1" else MAIN[b], "A_fine", sc, units, metric=mtr, elig_pairs=EP["A_fine"]))
        for c in R.PRED_COLS + ["GBT_ship_A_fine"]:
            for mtr in ("rho", "capture20", "r2_loglog"): cmps.append(E.compare(tab, c, "sigma_m_R02", "A_fine", sc, NAT, metric=mtr, elig_pairs=EP["A_fine"]))
        for sa in R.SEEDS_V1:
            for sb in R.SEEDS_V1:
                if sa < sb: cmps.append(E.compare(tab, f"V1_s{sa}", f"V1_s{sb}", "A_fine", sc, units, metric="rho", elig_pairs=EP["A_fine"]))
        CMP[sc] = cmps
        md.append(f"\n## A_fine/{sc}: median rho [95 % CI] per predictor\n\n| predictor | median rho | 95 % CI | n units |\n|---|---|---|---|")
        for c, r in CI[sc].items(): md.append(f"| {c} | {r['value']:.3f} | [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] | {r['n_units']} |")
        md.append(f"\n## A_fine/{sc}: comparisons (bootstrap over units, exact one-sided sign test)\n\n" + E.md_compare(cmps))
    OUT["median_ci"] = CI; OUT["comparisons"] = CMP
    # ---- §2 seed spread (V1)
    vals = {s: CI["tile"][f"V1_s{s}"]["value"] for s in R.SEEDS_V1}; pu = {s: S["A_fine/tile"][f"V1_s{s}"]["per_unit"] for s in R.SEEDS_V1}
    per_unit_sd = {u: float(np.std([pu[s][u].get("rho", np.nan) for s in R.SEEDS_V1], ddof=1)) for u in units}
    per_unit_range = {u: float(np.ptp([pu[s][u].get("rho", np.nan) for s in R.SEEDS_V1])) for u in units}
    seed = dict(median_rho_per_seed=vals, spread_max_minus_min=float(max(vals.values()) - min(vals.values())), sd_of_medians=float(np.std(list(vals.values()), ddof=1)),
                seedavg_median_rho=CI["tile"]["V1_seedavg"]["value"], per_unit_sd=per_unit_sd, per_unit_range=per_unit_range, median_per_unit_sd=float(np.nanmedian(list(per_unit_sd.values()))), median_per_unit_range=float(np.nanmedian(list(per_unit_range.values()))))
    OUT["seed_check"] = seed
    md.append("\n## Seed check (V1, A_fine/tile)\n\n| seed | median rho | 95 % CI |\n|---|---|---|" + "".join(f"\n| {s} | {CI['tile'][f'V1_s{s}']['value']:.3f} | [{CI['tile'][f'V1_s{s}']['ci95'][0]:.3f}, {CI['tile'][f'V1_s{s}']['ci95'][1]:.3f}] |" for s in R.SEEDS_V1)
              + f"\n| seed-averaged prediction | {seed['seedavg_median_rho']:.3f} | [{CI['tile']['V1_seedavg']['ci95'][0]:.3f}, {CI['tile']['V1_seedavg']['ci95'][1]:.3f}] |"
              + f"\n\nSpread of the median over the three seeds: {seed['spread_max_minus_min']:.3f} (max - min), SD {seed['sd_of_medians']:.3f}; per-unit rho: median SD across seeds {seed['median_per_unit_sd']:.3f}, median range {seed['median_per_unit_range']:.3f}.")
    # ---- §3.5 EW0207
    EW = {}
    t_all = {v: load_preds(v, R.SEED_MAIN, ["ALL"]) for v in ("V0", "V1", "V2", "V3")}
    for v, d in t_all.items(): assert set(d.pair) == {EW_PAIR} and set(d.unit) == {"lu16"}
    for sc, base in (("tile", pd.read_parquet(f"{R.BAR}/pred_tile_EW0207.parquet")), ("block", pd.read_parquet(f"{R.BAR}/pred_block_EW0207.parquet"))):
        Pe = base.copy()
        if sc == "tile":
            for v, d in t_all.items(): Pe[v] = d.set_index("tile_id").tile_pred_m.reindex(Pe.tile_id).values
        else:
            key = pd.MultiIndex.from_frame(Pe[["tile_id", "block_id"]])
            for v, d in t_all.items():
                L = d.set_index("tile_id")[blk_cols()].stack().reset_index(); L.columns = ["tile_id", "blk", "val"]; L["block_id"] = L.blk.str[3:].astype(int)
                Pe[v] = (10 ** L.set_index(["tile_id", "block_id"]).val).reindex(key).values
        for v in ("V0", "V1", "V2", "V3"): assert np.isfinite(Pe[v]).all()
        tbl = Pe.assign(unit="lu16"); cols = ["V0", "V1", "V2", "V3", "GBT_ship_A_fine", "GBT_noship_A_fine", "ref_C", "ref_s_lr", "ref_shipstd128"]
        EW[sc] = {c: E.score_one(tbl, c, "A_fine", sc, ["lu16"], elig_pairs=[EW_PAIR]) for c in cols}
        Pe.to_parquet(f"{R.SCORES}/pred_{sc}_EW0207_r06.parquet")
        md.append(f"\n## EW0207 (lu16), {sc} scale, A_fine — separate one-unit table ({len(Pe)} rows; variants trained on all 16 development units, seed {R.SEED_MAIN})\n\n| predictor | rho | log-log R² | capture20 |\n|---|---|---|---|")
        for c, s in EW[sc].items(): r = s["per_unit"]["lu16"]; md.append(f"| {c} | {r.get('rho', np.nan):.3f} | {r.get('r2_loglog', np.nan):.3f} | {r.get('capture20', np.nan):.3f} |")
    OUT["EW0207"] = EW
    # ---- §3.6 per-unit table (A_fine/tile rho) with each unit's median A_fine
    medA = Pt.groupby("unit").A_fine.median(); cols = R.PRED_COLS + ["GBT_ship_A_fine", "ref_s_lr", "ref_C", "sigma_m_R02"]
    rows = []
    for u in units:
        r = dict(unit=u, n_pairs=len(C4.UNITS[u]), n_tiles=int((Pt.unit == u).sum()), median_A_fine_tile_m=float(medA[u]), low_amplitude=(u in R.LOW_AMP_UNITS))
        for c in cols: r[c] = S["A_fine/tile"][c]["per_unit"][u].get("rho", np.nan)
        rows.append(r)
    PU = pd.DataFrame(rows); PU.to_csv(f"{R.SCORES}/per_unit_table.csv", index=False)
    md.append("\n## Per-unit A_fine/tile rho (unit value = median over eligible pairs)\n\n| unit | pairs | tiles | median A_fine (m) | " + " | ".join(cols) + " |\n|---|---|---|---|" + "---|" * len(cols))
    for r in rows: md.append(f"| {r['unit']}{' *' if r['low_amplitude'] else ''} | {r['n_pairs']} | {r['n_tiles']} | {r['median_A_fine_tile_m']:.4f} | " + " | ".join(f3(r[c]) for c in cols) + " |")
    md.append("\n`*` low-amplitude unit (median A_fine < 3 cm, §4).")
    # ---- §4 diagnostic
    F = pd.read_parquet(R.FEATURES, columns=["tile_id", "block_id", "dist_edge_128", "frac_valid_128"]); F = F[F.block_id < 0].drop(columns="block_id")
    T = Pt.merge(F, on="tile_id", how="left"); assert len(T) == len(Pt)
    diag = dict(reference_rows={c: E.score_one(T, c, "A_fine", "tile", units, EP["A_fine"]) for c in ("dist_edge_128", "frac_valid_128")}, partial={})
    pcols = ["GBT_ship_A_fine"] + R.PRED_COLS
    for c in pcols: diag["partial"][c] = partial_by_unit(T, c, set(EP["A_fine"]))
    low = [u for u in units if u in R.LOW_AMP_UNITS]; other = [u for u in units if u not in R.LOW_AMP_UNITS]
    diag["partial_medians"] = {c: dict(low_amplitude=float(np.nanmedian([diag["partial"][c][u] for u in low])), other=float(np.nanmedian([diag["partial"][c][u] for u in other]))) for c in pcols}
    diag["plain_medians"] = {c: dict(low_amplitude=float(np.nanmedian([S["A_fine/tile"][c]["per_unit"][u].get("rho", np.nan) for u in low])), other=float(np.nanmedian([S["A_fine/tile"][c]["per_unit"][u].get("rho", np.nan) for u in other]))) for c in pcols}
    diag["low_amplitude_units"] = low; json.dump(diag, open(f"{R.SCORES}/diagnostic_lowamp.json", "w"), indent=1, default=float)
    md.append("\n## §4 diagnostic — low-amplitude units\n\n### Reference rows (A_fine/tile, eval_v1.1)\n\n" + E.md_scores(diag["reference_rows"], units))
    md.append("\n### Within-pair partial Spearman rho (prediction vs A_fine | dist_edge_128, frac_valid_128), unit = median over eligible pairs\n\n| predictor | " + " | ".join(units) + " | median low-amp (6) | median other (10) | plain rho median low-amp / other |\n|---|" + "---|" * (len(units) + 3))
    for c in pcols: md.append(f"| {c} | " + " | ".join(f3(diag["partial"][c][u]) for u in units) + f" | {f3(diag['partial_medians'][c]['low_amplitude'])} | {f3(diag['partial_medians'][c]['other'])} | {f3(diag['plain_medians'][c]['low_amplitude'])} / {f3(diag['plain_medians'][c]['other'])} |")
    # ---- reading rule (§3)
    med = {v: CI["tile"][MAIN[v]]["value"] for v in ("V0", "V1", "V2", "V3")}; best = max(med, key=med.get); spread = seed["spread_max_minus_min"]
    within = [v for v in med if med[best] - med[v] < spread]; simplicity = {"V3": 0, "V0": 1, "V1": 1, "V2": 2}     # fewer inputs first (V2 last), then smaller window (V3 first); V0 / V1 tie
    named = sorted(within, key=lambda v: (simplicity[v], v))
    OUT["reading_rule"] = dict(median_rho=med, highest=best, seed_spread=spread, within_spread_of_highest=within, named_candidate=named[0], tie_between=[v for v in within if simplicity[v] == simplicity[named[0]]],
                               note="choosing among four variants on the same 16 units biases the named candidate's development score upward; this is a round-2 candidate, not adoption")
    md.append(f"\n## Reading rule\n\nMedian rho A_fine/tile: " + ", ".join(f"{v} {med[v]:.3f}" for v in med) + f". Highest: {best}. Seed spread (V1, max - min of the three medians): {spread:.3f}. Within the spread of the highest: {within}. Named candidate under the rule (fewer inputs, then smaller window): **{named[0]}**" + (f" (tie by the rule with {[v for v in within if simplicity[v] == simplicity[named[0]] and v != named[0]]})" if len([v for v in within if simplicity[v] == simplicity[named[0]]]) > 1 else "") + ". Choosing among four variants on the same 16 units biases the winner's development score upward; this names a round-2 candidate, not an adopted model.")
    # ---- compute summary
    logs = [json.load(open(f)) for f in glob.glob(f"{R.MODELS}/*/seed*/fold_*/train_log.json")]
    comp = dict(n_trainings=len(logs), gpu_h_total=sum(l["wall_total_s"] for l in logs) / 3600, gpu_h_train=sum(l["wall_train_s"] for l in logs) / 3600, by_gpu={}, by_variant={}, budget_gpu_h=R.BUDGET_GPU_H)
    for l in logs:
        comp["by_gpu"].setdefault(l["gpu"], dict(n=0, gpu_h=0.0)); comp["by_gpu"][l["gpu"]]["n"] += 1; comp["by_gpu"][l["gpu"]]["gpu_h"] += l["wall_total_s"] / 3600
        comp["by_variant"].setdefault(l["variant"], dict(n=0, gpu_h=0.0, s_per_step=[])); comp["by_variant"][l["variant"]]["n"] += 1; comp["by_variant"][l["variant"]]["gpu_h"] += l["wall_total_s"] / 3600; comp["by_variant"][l["variant"]]["s_per_step"].append(l["s_per_step"])
    for v in comp["by_variant"]: comp["by_variant"][v]["s_per_step"] = float(np.median(comp["by_variant"][v]["s_per_step"]))
    comp["params"] = {l["variant"]: l["params"] for l in logs}; comp["schedule"] = {k: logs[0][k] for k in ("steps", "batch", "groups_per_batch", "windows_per_group", "lr", "weight_decay", "eta_min", "grad_clip", "precision", "optimizer", "schedule")} if logs else {}
    json.dump(comp, open(f"{R.SCORES}/compute_summary.json", "w"), indent=1, default=float); OUT["compute"] = comp
    md.append(f"\n## Compute\n\n{comp['n_trainings']} trainings, {comp['gpu_h_total']:.1f} GPU-h in total (training {comp['gpu_h_train']:.1f} GPU-h) against the {R.BUDGET_GPU_H:.0f} GPU-h cap; by GPU: " + "; ".join(f"{g}: {d['n']} trainings, {d['gpu_h']:.1f} GPU-h" for g, d in comp["by_gpu"].items()) + "; by variant: " + "; ".join(f"{v}: {d['n']} x median {d['s_per_step']:.3f} s/step, {d['gpu_h']:.1f} GPU-h" for v, d in comp["by_variant"].items()) + ".")
    OUT["scoring_code_sha256"] = R.sha256_file(__file__); OUT["training_code_sha256"] = R.code_hash()
    json.dump(OUT, open(f"{R.SCORES}/R06_scores.json", "w"), indent=1, default=float); open(f"{R.SCORES}/R06_scores.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md)); print(f"\nwrote {R.SCORES}/R06_scores.md ({time.time()-t0:.0f} s)", flush=True)
