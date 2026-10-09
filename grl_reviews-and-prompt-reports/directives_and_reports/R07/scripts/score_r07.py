"""R07 §4 scoring: every arm's leave-one-unit-out predictions are added as columns to the R05 bar prediction tables
(A_fine, metres: block = 10^pred, tile = RMS over the 16 blocks; seed-averaged = mean of the three seeds' log10 block
predictions) and scored ONLY through the frozen eval_v1_1 (score_one / compare / median_ci) with the R05 pair rules.
  §4.1 primary: P-V1 (seed-avg) - V1 (seed-avg, R06): median d-rho, 95 % bootstrap CI over units, units better of 16,
       one-sided sign test; reading rule fixed by the directive (CI lower > 0 improves / includes 0 none / upper < 0 worsens).
  §4.2 secondary: P-V2 - V2 (both seed-avg); each of P-V1, P-V2, V1, V2 vs the bar (GBT_ship) and the references;
       seed spread of P-V1, P-V2, V2 next to R06 V1's; EW0207 one-unit table (all-unit models); native 7 vs R02 sigma_m;
       block scale; per-unit table.
R06 arms are read from the R06 scratch outputs after verifying every file against the OAK R06 SHA256SUMS.
Writes scores/R07_scores.{md,json}, pred_*_r07.parquet, per_unit_table.csv, compute_summary.json."""
from __future__ import annotations
import glob, json, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r07_common as R7, r06_common as R6, r04_common as C4, eval_v1_1 as E

EW_PAIR = C4.EW0207["lu16"][0]; NAT = [u for u in C4.UNITS if u in C4.NATIVE7]; SEEDS = R7.SEEDS
ARMS4 = ("P-V1", "V1", "P-V2", "V2")
REFS = ("GBT_ship_A_fine", "ref_s_lr", "ref_C")

def src_dir(arm, seed, fold):
    """Where an arm's fold lives: R06 scratch for V1 (all seeds) and V2 seed 20261002 (verified against OAK), R07 otherwise."""
    if arm == "V1" or (arm == "V2" and seed == R7.SEED_MAIN):
        rel = f"{arm}/seed{seed}/fold_{fold}"; R7.verify_r06_file(f"{rel}/pred.parquet"); return f"{R6.MODELS}/{rel}", "R06"
    return f"{R7.MODELS}/{arm}/seed{seed}/fold_{fold}", "R07"

def load_preds(arm, seed, folds):
    dfs = []; srcs = set()
    for f in folds:
        d, src = src_dir(arm, seed, f); p = f"{d}/pred.parquet"
        if not os.path.exists(p): raise SystemExit(f"missing prediction: {p}")
        x = pd.read_parquet(p); x["fold"] = f
        if f != "ALL": assert set(x.unit) == {f}, (f, set(x.unit))
        dfs.append(x); srcs.add(src)
    d = pd.concat(dfs, ignore_index=True); assert not d.tile_id.duplicated().any(); return d, sorted(srcs)

def blk_cols(): return [f"blk{j}" for j in range(R6.NB * R6.NB)]

def seed_average(dfs):
    base = dfs[0][["tile_id", "unit", "pair"]].copy(); M = np.mean([d.set_index("tile_id").loc[base.tile_id, blk_cols()].values for d in dfs], axis=0)
    for j, c in enumerate(blk_cols()): base[c] = M[:, j]
    base["tile_pred_m"] = np.sqrt(np.mean(10 ** (2 * M), axis=1)); return base

def add_pred(Pt, Pb, name, df):
    t = df.set_index("tile_id"); Pt[name] = t.tile_pred_m.reindex(Pt.tile_id).values
    L = t[blk_cols()].stack().reset_index(); L.columns = ["tile_id", "blk", "v"]; L["block_id"] = L.blk.str[3:].astype(int)
    key = pd.MultiIndex.from_frame(Pb[["tile_id", "block_id"]]); Pb[name] = (10 ** L.set_index(["tile_id", "block_id"]).v).reindex(key).values
    assert np.isfinite(Pt[name]).all() and np.isfinite(Pb[name]).all(), name

def reading(c):
    lo, hi = c["ci95"]
    if lo > 0: return "pretraining improves ranking (CI lower bound above 0)"
    if hi < 0: return "pretraining worsens ranking (CI upper bound below 0)"
    return "no detectable effect (CI includes 0)"

def f3(x): return "nan" if x is None or not np.isfinite(x) else f"{x:+.3f}"
def fmt_cmp(c): return f"{c['diff_median']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] | {c['units_a_better']} / {c['units_b_better']} of {c['n_units']} | {c['sign_test_p_one_sided']:.3f}"

if __name__ == "__main__":
    if os.environ.get("R07_MODELS"): R7.MODELS = os.environ["R07_MODELS"]; R7.SCORES = os.environ.get("R07_SCORES", R7.SCORES)   # synthetic self-test only
    m = R7.start("score_r07", check_corpus_files=False); os.makedirs(R7.SCORES, exist_ok=True); t0 = time.time(); units = list(C4.UNITS)
    Pt = pd.read_parquet(f"{R6.BAR}/pred_tile.parquet"); Pb = pd.read_parquet(f"{R6.BAR}/pred_block.parquet"); rules = json.load(open(f"{R6.BAR}/pair_rules.json"))
    EP = rules["eligible"]; DET = rules["detection_pairs"]; E.assert_units(Pt); E.assert_units(Pb)
    preds = {}; sources = {}; cols = []
    for arm in ARMS4:
        for s in SEEDS:
            preds[f"{arm}_s{s}"], sources[f"{arm}_s{s}"] = load_preds(arm, s, units); cols.append(f"{arm}_s{s}")
        preds[f"{arm}_seedavg"] = seed_average([preds[f"{arm}_s{s}"] for s in SEEDS]); cols.append(f"{arm}_seedavg")
    for c in cols: add_pred(Pt, Pb, c, preds[c])
    Pt.to_parquet(f"{R7.SCORES}/pred_tile_r07.parquet"); Pb.to_parquet(f"{R7.SCORES}/pred_block_r07.parquet")
    OUT = dict(prediction_columns=cols, sources=sources); md = [f"# R07 Phase 3 scores (eval_v1.1, tiles_v1.1 / blocks_v1.1, R05 pair rules)\n",
          f"Units: {C4.UNITS_VERSION} ({len(units)}); eligible pairs A_fine {len(EP['A_fine'])}/38; detection pairs {len(DET)}/38; columns: {', '.join(cols)} (A_fine in metres: block 10^pred, tile RMS over 16 blocks; seedavg = mean of the three seeds' log10 block predictions). V1 (3 seeds) and V2 seed {R7.SEED_MAIN} are the R06 outputs (verified against OAK); V2 seeds {SEEDS[1]} / {SEEDS[2]} and both P arms are R07.\n"]
    AVG = [f"{a}_seedavg" for a in ARMS4]
    # ---- every endpoint (context)
    S = {}
    for tgt, scales in E.TARGETS.items():
        for sc in scales:
            tab = Pt if sc == "tile" else Pb
            ctx = [c for c in ("GBT_ship_A_fine", "GBT_noship_A_fine", "ref_C", "ref_s_lr", "sigma_m_R02") if c in tab.columns] if tgt == "A_fine" else [c for c in (f"GBT_ship_{tgt}", f"GBT_noship_{tgt}", "ref_C", "ref_s_lr", "sigma_m_R02") if c in tab.columns]
            S[f"{tgt}/{sc}"] = {c: E.score_one(tab, c, tgt, sc, units, EP[tgt], DET) for c in cols + ctx}
            md.append(f"\n## {tgt}/{sc}\n\n" + E.md_scores(S[f"{tgt}/{sc}"], units))
    OUT["scores"] = S
    # ---- A_fine at tile (primary) and block (secondary): median CI, comparisons
    CI = {}; CMP = {}
    for sc, tab in (("tile", Pt), ("block", Pb)):
        CI[sc] = {c: E.median_ci(tab, c, "A_fine", sc, units, metric="rho", elig_pairs=EP["A_fine"]) for c in cols + list(REFS)}
        cm = {}
        for mtr in ("rho", "capture20", "r2_loglog"):
            cm[f"primary P-V1-V1/{mtr}"] = E.compare(tab, "P-V1_seedavg", "V1_seedavg", "A_fine", sc, units, metric=mtr, elig_pairs=EP["A_fine"])
            cm[f"secondary P-V2-V2/{mtr}"] = E.compare(tab, "P-V2_seedavg", "V2_seedavg", "A_fine", sc, units, metric=mtr, elig_pairs=EP["A_fine"])
            for s in SEEDS:   # same-seed pairs, for information
                cm[f"P-V1-V1 seed {s}/{mtr}"] = E.compare(tab, f"P-V1_s{s}", f"V1_s{s}", "A_fine", sc, units, metric=mtr, elig_pairs=EP["A_fine"])
                cm[f"P-V2-V2 seed {s}/{mtr}"] = E.compare(tab, f"P-V2_s{s}", f"V2_s{s}", "A_fine", sc, units, metric=mtr, elig_pairs=EP["A_fine"])
            for c in cols:
                for ref in REFS: cm[f"{c}-{ref}/{mtr}"] = E.compare(tab, c, ref, "A_fine", sc, units, metric=mtr, elig_pairs=EP["A_fine"])
            for c in AVG + ["GBT_ship_A_fine"]: cm[f"native7 {c}-sigma_m_R02/{mtr}"] = E.compare(tab, c, "sigma_m_R02", "A_fine", sc, NAT, metric=mtr, elig_pairs=EP["A_fine"])
        CMP[sc] = cm
        md.append(f"\n## A_fine/{sc}: median rho [95 % CI] per predictor\n\n| predictor | median rho | 95 % CI | n units |\n|---|---|---|---|")
        for c, r in CI[sc].items(): md.append(f"| {c} | {r['value']:.3f} | [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] | {r['n_units']} |")
        md.append(f"\n## A_fine/{sc}: comparisons (bootstrap over units, exact one-sided sign test)\n\n" + E.md_compare(list(cm.values())))
    OUT["median_ci"] = CI; OUT["comparisons"] = CMP
    # ---- §4.1 primary reading
    prim = CMP["tile"]["primary P-V1-V1/rho"]; OUT["primary"] = dict(comparison="P-V1_seedavg - V1_seedavg, A_fine/tile, rho", result=prim, reading=reading(prim))
    md.append(f"\n## §4.1 Primary comparison: P-V1 (seed-averaged) − V1 (seed-averaged, R06), A_fine/tile\n\n| Δ median ρ [95 % CI] | units better / worse of 16 | sign-test p |\n|---|---|---|\n| {fmt_cmp(prim)} |\n\n**Reading (fixed rule): {OUT['primary']['reading']}.**")
    sec = CMP["tile"]["secondary P-V2-V2/rho"]; OUT["secondary_pv2_v2"] = dict(result=sec, reading=reading(sec))
    md.append(f"\n## §4.2.1 P-V2 − V2 (both seed-averaged), A_fine/tile\n\n| Δ median ρ [95 % CI] | units better / worse of 16 | sign-test p |\n|---|---|---|\n| {fmt_cmp(sec)} |\n\n(Reported, not used to reverse the primary reading; under the same rule it would read: {OUT['secondary_pv2_v2']['reading']}.)")
    # ---- §4.2.2 each arm vs the bar
    md.append("\n## §4.2.2 Each arm (seed-averaged, and per seed) against the bar and the references, A_fine/tile\n\n| arm | median ρ [95 % CI] | − GBT_ship Δ [CI] | better / worse | p | − ref_s_lr Δ [CI] | better / worse | p | − ref_C Δ [CI] | better / worse | p |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for c in cols:
        r = CI["tile"][c]; row = f"| {c} | {r['value']:.3f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] |"
        for ref in REFS: row += " " + fmt_cmp(CMP["tile"][f"{c}-{ref}/rho"]) + " |"
        md.append(row)
    for ref in REFS: r = CI["tile"][ref]; md.append(f"| ({ref}) | {r['value']:.3f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] | | | | | | | | | |")
    # ---- §4.2.3 seed spread
    SP = {}
    for arm in ARMS4:
        vals = {s: CI["tile"][f"{arm}_s{s}"]["value"] for s in SEEDS}; pu = {s: S["A_fine/tile"][f"{arm}_s{s}"]["per_unit"] for s in SEEDS}
        per_unit_sd = {u: float(np.std([pu[s][u].get("rho", np.nan) for s in SEEDS], ddof=1)) for u in units}
        SP[arm] = dict(median_rho_per_seed=vals, spread_max_minus_min=float(max(vals.values()) - min(vals.values())), sd_of_medians=float(np.std(list(vals.values()), ddof=1)),
                       seedavg_median_rho=CI["tile"][f"{arm}_seedavg"]["value"], median_per_unit_sd=float(np.nanmedian(list(per_unit_sd.values()))), per_unit_sd=per_unit_sd)
    OUT["seed_spread"] = SP
    md.append("\n## §4.2.3 Seed spread (A_fine/tile; R06 V1's spread was 0.023)\n\n| arm | median ρ seed 20261002 | 20261003 | 20261004 | spread (max − min) | SD of medians | seed-averaged median ρ | median per-unit SD across seeds |\n|---|---|---|---|---|---|---|---|")
    for arm in ARMS4: sp = SP[arm]; md.append(f"| {arm} | " + " | ".join(f"{sp['median_rho_per_seed'][s]:.3f}" for s in SEEDS) + f" | {sp['spread_max_minus_min']:.3f} | {sp['sd_of_medians']:.3f} | {sp['seedavg_median_rho']:.3f} | {sp['median_per_unit_sd']:.3f} |")
    # ---- §4.2.4 EW0207 (all-unit models)
    EW = {}; ew_cols = []; ew_preds = {}
    for arm in ARMS4:
        have = []
        for s in SEEDS:
            d, _ = src_dir(arm, s, "ALL") if not (arm == "V1" and s != R7.SEED_MAIN) else (None, None)
            if d and os.path.exists(f"{d}/pred.parquet"):
                x, _ = load_preds(arm, s, ["ALL"]); assert set(x.pair) == {EW_PAIR}; ew_preds[f"{arm}_s{s}"] = x; ew_cols.append(f"{arm}_s{s}"); have.append(x)
        if len(have) == len(SEEDS): ew_preds[f"{arm}_seedavg"] = seed_average(have); ew_cols.append(f"{arm}_seedavg")
    for sc, base in (("tile", pd.read_parquet(f"{R6.BAR}/pred_tile_EW0207.parquet")), ("block", pd.read_parquet(f"{R6.BAR}/pred_block_EW0207.parquet"))):
        Pe = base.copy()
        if sc == "tile":
            for c, d in ew_preds.items(): Pe[c] = d.set_index("tile_id").tile_pred_m.reindex(Pe.tile_id).values
        else:
            key = pd.MultiIndex.from_frame(Pe[["tile_id", "block_id"]])
            for c, d in ew_preds.items():
                L = d.set_index("tile_id")[blk_cols()].stack().reset_index(); L.columns = ["tile_id", "blk", "val"]; L["block_id"] = L.blk.str[3:].astype(int)
                Pe[c] = (10 ** L.set_index(["tile_id", "block_id"]).val).reindex(key).values
        for c in ew_cols: assert np.isfinite(Pe[c]).all(), c
        tbl = Pe.assign(unit="lu16"); ccols = ew_cols + ["GBT_ship_A_fine", "GBT_noship_A_fine", "ref_C", "ref_s_lr", "ref_shipstd128"]
        EW[sc] = {c: E.score_one(tbl, c, "A_fine", sc, ["lu16"], elig_pairs=[EW_PAIR]) for c in ccols}; Pe.to_parquet(f"{R7.SCORES}/pred_{sc}_EW0207_r07.parquet")
        md.append(f"\n## §4.2.4 EW0207 (lu16), {sc} scale, A_fine — one-unit table ({len(Pe)} rows; all-unit models; V1 has only R06's seed {R7.SEED_MAIN} all-unit model)\n\n| predictor | ρ | log-log R² | capture20 |\n|---|---|---|---|")
        for c, s in EW[sc].items(): r = s["per_unit"]["lu16"]; md.append(f"| {c} | {r.get('rho', np.nan):.3f} | {r.get('r2_loglog', np.nan):.3f} | {r.get('capture20', np.nan):.3f} |")
    OUT["EW0207"] = EW
    # ---- §4.2.5 native 7 vs sigma_m
    md.append("\n## §4.2.5 Native 7 units against R02 σ_m (LOUO), A_fine/tile and block\n\n| predictor − sigma_m_R02 | scale | Δ median ρ [95 % CI] | better / worse of 7 | p |\n|---|---|---|---|---|")
    for sc in ("tile", "block"):
        for c in AVG + ["GBT_ship_A_fine"]: md.append(f"| {c} | {sc} | {fmt_cmp(CMP[sc][f'native7 {c}-sigma_m_R02/rho'])} |")
    # ---- §4.2.6 block scale summary
    md.append("\n## §4.2.6 Block scale (A_fine/block): arms against the bar; primary and secondary differences\n\n| comparison | Δ median ρ [95 % CI] | better / worse of 16 | p |\n|---|---|---|---|")
    for k in ("primary P-V1-V1/rho", "secondary P-V2-V2/rho") + tuple(f"{c}-GBT_ship_A_fine/rho" for c in AVG): md.append(f"| {k.replace('/rho', '')} | {fmt_cmp(CMP['block'][k])} |")
    # ---- §4.2.7 per-unit table
    medA = Pt.groupby("unit").A_fine.median(); pcols = AVG + ["GBT_ship_A_fine", "ref_s_lr", "ref_C"]; rows = []
    for u in units:
        r = dict(unit=u, n_pairs=len(C4.UNITS[u]), n_tiles=int((Pt.unit == u).sum()), median_A_fine_tile_m=float(medA[u]), low_amplitude=(u in R6.LOW_AMP_UNITS))
        for c in pcols + [c for c in cols if c not in AVG]: r[c] = S["A_fine/tile"][c]["per_unit"][u].get("rho", np.nan)
        rows.append(r)
    PU = pd.DataFrame(rows); PU.to_csv(f"{R7.SCORES}/per_unit_table.csv", index=False)
    md.append("\n## §4.2.7 Per-unit A_fine/tile ρ (unit value = median over eligible pairs; * = low-amplitude unit, median A_fine < 3 cm)\n\n| unit | pairs | tiles | median A_fine (m) | " + " | ".join(pcols) + " |\n|---|---|---|---|" + "---|" * len(pcols))
    for r in rows: md.append(f"| {r['unit']}{' *' if r['low_amplitude'] else ''} | {r['n_pairs']} | {r['n_tiles']} | {r['median_A_fine_tile_m']:.4f} | " + " | ".join(f3(r[c]) for c in pcols) + " |")
    OUT["per_unit"] = rows
    # ---- compute summary (R07 trainings + pretraining + checks)
    logs = [json.load(open(f)) for f in glob.glob(f"{R7.MODELS}/*/seed*/fold_*/train_log.json")]
    pre = [json.load(open(f)) for f in glob.glob(f"{R7.PRETRAIN}/*/pretrain_log.json")]
    chk = [json.load(open(f)) for f in glob.glob(f"{R7.CHECKS}/scratch_equiv/*/seed*/fold_*/train_log.json")]
    comp = dict(n_finetunings=len(logs), gpu_h_finetune=sum(l["wall_total_s"] for l in logs) / 3600, n_pretrain_runs=len(pre), gpu_h_pretrain=sum(p["wall_train_s"] for p in pre) / 3600,
                gpu_h_checks=sum(l["wall_total_s"] for l in chk) / 3600, by_gpu={}, by_arm={}, pretrain=[dict(body=p["body"], pilot=p["pilot"], steps=p["steps"], gpu=p["gpu"], gpu_h=p["wall_train_s"] / 3600, s_per_step=p["s_per_step"]) for p in pre], budget_gpu_h=R7.BUDGET_GPU_H)
    for l in logs:
        comp["by_gpu"].setdefault(l["gpu"], dict(n=0, gpu_h=0.0)); comp["by_gpu"][l["gpu"]]["n"] += 1; comp["by_gpu"][l["gpu"]]["gpu_h"] += l["wall_total_s"] / 3600
        comp["by_arm"].setdefault(l["arm"], dict(n=0, gpu_h=0.0, s_per_step=[])); comp["by_arm"][l["arm"]]["n"] += 1; comp["by_arm"][l["arm"]]["gpu_h"] += l["wall_total_s"] / 3600; comp["by_arm"][l["arm"]]["s_per_step"].append(l["s_per_step"])
    for v in comp["by_arm"]: comp["by_arm"][v]["s_per_step"] = float(np.median(comp["by_arm"][v]["s_per_step"]))
    comp["gpu_h_total"] = comp["gpu_h_finetune"] + comp["gpu_h_pretrain"] + comp["gpu_h_checks"]
    json.dump(comp, open(f"{R7.SCORES}/compute_summary.json", "w"), indent=1, default=float); OUT["compute"] = comp
    md.append(f"\n## Compute\n\n{comp['n_finetunings']} fine-tunings ({comp['gpu_h_finetune']:.1f} GPU-h), {comp['n_pretrain_runs']} pretraining runs incl. pilot ({comp['gpu_h_pretrain']:.1f} GPU-h), checks {comp['gpu_h_checks']:.1f} GPU-h: {comp['gpu_h_total']:.1f} GPU-h against the {R7.BUDGET_GPU_H:.0f} GPU-h cap; by GPU: " + "; ".join(f"{g}: {d['n']} trainings, {d['gpu_h']:.1f} GPU-h" for g, d in comp["by_gpu"].items()) + "; by arm: " + "; ".join(f"{v}: {d['n']} x median {d['s_per_step']:.3f} s/step, {d['gpu_h']:.1f} GPU-h" for v, d in comp["by_arm"].items()) + ".")
    md.append("\nNote (§4.3): this round adopts no model and sets no adoption criteria; selecting among variants on the same 16 units biases the winner's development score upward.")
    OUT["scoring_code_sha256"] = R7.sha256_file(__file__); OUT["training_code_sha256"] = R7.code_hash()
    json.dump(OUT, open(f"{R7.SCORES}/R07_scores.json", "w"), indent=1, default=float); open(f"{R7.SCORES}/R07_scores.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md)); print(f"\nwrote {R7.SCORES}/R07_scores.md ({time.time()-t0:.0f} s)", flush=True)
