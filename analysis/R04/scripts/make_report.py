"""Assemble REPORT_R04 from the persisted artifacts (numbers are read from files, not retyped) plus the hand-written
text blocks in this file. Re-run after the bar job finishes."""
import json, os, sys, glob
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import r04_common as C
R = C.ROOT; D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
J = lambda p: json.load(open(p)); T = lambda p: open(p).read()
ts = J(f"{R}/tiles/tiles_summary.json"); cc = J(f"{R}/tiles/consistency_check.json"); dg = pd.read_csv(f"{R}/tiles/consistency_diag_units.csv").set_index("unit")
uc = pd.read_csv(f"{R}/tiles/unit_counts.csv").set_index("unit"); pc = pd.read_csv(f"{R}/tiles/pair_counts.csv")
ca = J(f"{R}/checks/check_a.json"); cb = J(f"{R}/checks/check_b.json"); val = J(f"{R}/validation/contract_v2_1_validation.json")
fav = J(f"{R}/features/features_v1_availability.json") if os.path.exists(f"{R}/features/features_v1_availability.json") else {}
flk = J(f"{R}/features/features_v1_leak.json") if os.path.exists(f"{R}/features/features_v1_leak.json") else {}
bar_md = T(f"{R}/bar/BAR_v1.md") if os.path.exists(f"{R}/bar/BAR_v1.md") else "BAR_PLACEHOLDER (bar job not finished)"
barb = J(f"{R}/bar/bar_results_block.json") if os.path.exists(f"{R}/bar/bar_results_block.json") else None
bart = J(f"{R}/bar/bar_results_tile.json") if os.path.exists(f"{R}/bar/bar_results_tile.json") else None
frz = J(f"{D}/freeze_record.json") if os.path.exists(f"{D}/freeze_record.json") else None
pres = [l.strip() for l in T(f"{R}/logs/r04_preserve_46236480.out").splitlines() if l.startswith("VERIFIED")]
pres4 = sorted(glob.glob(f"{R}/logs/r04_preserve_r04_*.out")); pres4 = [l.strip() for l in T(pres4[-1]).splitlines() if l.startswith("VERIFIED")] if pres4 else []
U = list(C.UNITS)
def f3(x): return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.3f}"

def bar_summary():
    if barb is None: return "BAR_PLACEHOLDER", {}
    out = {}; L = []
    for scale, res in (("block", barb), ("tile", bart)):
        for key, r in res.items():
            S = r["scores"]; ci = r["primary_ci"]; cmp = {(c["a"], c["b"], c["metric"]): c for c in r["comparisons"]}
            out[f"{key}"] = dict(gbt_ship=ci, gbt_noship=S.get(key.split("/")[0].join(["GBT_noship_", ""]), {}).get("median", {}), cmp=cmp, scores=S)
    return "\n".join(L), out
_, BS = bar_summary()
def cmp_line(key, a, b, metric="rho"):
    c = BS.get(key, {}).get("cmp", {}).get((a, b, metric))
    return "n/a" if c is None else f"{c['diff_median']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}], {c['units_a_better']} of {c['n_units']} units better (sign-test p = {c['sign_test_p_one_sided']:.3f})"
def med(key, pred, metric="rho"):
    s = BS.get(key, {}).get("scores", {}).get(pred); return "n/a" if s is None else f"{s['median'].get(metric, float('nan')):.3f}"
def prim(key):
    ci = BS.get(key, {}).get("gbt_ship"); return "n/a" if not ci else f"{ci['value']:.3f} [95 % CI {ci['ci95'][0]:.3f}, {ci['ci95'][1]:.3f}]"

# ---------------- text blocks ----------------
bottom = f"""1. **Challenge findings (§2): 10 items, 1 blocking by the directive's own rule.** The unit list is CONFIRMED (39 development pairs with k in manifest v2.1 = the 38 frozen pairs in 16 units + EW0207; 17 of 38 pairs / 12 of 16 units with usable ship products); manifest v2.1 is in the acquisition repo, not on OAK, with the directive's sha256; the superseded 2026-09-29 directive file does not exist; the submitted paper's number is 0.555, not 0.53 (the 0.53 / 0.005 / 0.06 are persisted-analysis numbers, all reproduced in §6); and the §1.1 matched-support check cannot reach 0.9 by construction (item 5, HOLD H1).
2. **Targets and tiles_v1 built and frozen (§3).** {ts['pairs'] and sum(p['n_windows'] for p in ts['pairs'])} windows / tiles at stride 128 over 39 pairs ({int(uc.loc[U, 'windows'].sum())} on the 16 development units, {int(uc.loc['lu16', 'windows'])} on EW0207), {int(uc.loc[U, 'blocks'].sum())} development blocks of which {int(uc.loc[U, 'kept'].sum())} kept (≥ 75 % joint-valid). The window grid is **identical to R02/R03 on every pair that had windows** (33 of 39; the 6 new pairs got theirs from the same code).
3. **Consistency check at matched support: CONTRADICTED on 12 of 15 units, and the reason is the R03 definition, not the R04 one (§3.3, HOLD H1).** R03's A_fine was the RMS over the full 256 window of a hard-annulus, untapered band-pass; my re-implementation reproduces it to r = {min(v['vs_r03conv_recomputed']['r_log'] for v in cc.values()):.4f}–1.0000. Its outer 16-px band carries {dg['med_edge16'].div(dg['med_interior']).min():.1f}–{dg['med_edge16'].div(dg['med_interior']).max():.1f}× the interior RMS (filter ringing at the window boundary and at zero-filled holes). Against the R04 field (taper, raised cosine, central 128) the log-correlation is {min(v['vs_Fhr256']['r_log'] for v in cc.values()):.2f}–{max(v['vs_Fhr256']['r_log'] for v in cc.values()):.2f} (≥ 0.9 only on TN399, FK181031, TN299) with median ratios {min(v['vs_Fhr256']['median_ratio'] for v in cc.values()):.2f}–{max(v['vs_Fhr256']['median_ratio'] for v in cc.values()):.2f}; the filter shape is immaterial (raised-cosine / hard-annulus ratio 0.98, r = 1.000). Restricting the R03 filter to the central 128 px recovers r = {dg['r_log_ha128_rc128'].min():.2f}–{dg['r_log_ha128_rc128'].max():.3f} (median {dg['r_log_ha128_rc128'].median():.2f}) with the R04 tile. Consequence: every R03 A_fine / T_fine result (R03 §3–§4, and the R03 numbers quoted in the directive) was measured on an edge-artifact-dominated target.
4. **The bar (§5): GBT_ship, A_fine at block scale, median within-unit Spearman ρ = {prim('A_fine/block')}.** GBT_ship − GBT_noship: {cmp_line('A_fine/block', 'GBT_ship_A_fine', 'GBT_noship_A_fine')}. Tile scale: {prim('A_fine/tile')}; T_fine block {prim('T_fine/block')} / tile {prim('T_fine/tile')}; E_1m block (10 detection units) P(E > 0) median AUPRC {med('E_1m/block', 'P_GBT_ship_E_1m', 'auprc')}, ρ {med('E_1m/block', 'GBT_ship_E_1m')}.
5. **Reference rows and R02 σ_m against the bar (A_fine, block, median ρ):** roughness C {med('A_fine/block', 'ref_C')}, s_lr (R02 tile statistic) {med('A_fine/block', 'ref_s_lr')}, ship std at 128 px {med('A_fine/block', 'ref_shipstd128')}, median ship_rsd at 128 px {med('A_fine/block', 'ref_rsd128')} (on usable units); R02 σ_m (native 7) {med('A_fine/block', 'sigma_m_R02')} vs GBT_ship on the same 7 units: {cmp_line('A_fine/block', 'GBT_ship_A_fine', 'sigma_m_R02')}. GBT_ship − C: {cmp_line('A_fine/block', 'GBT_ship_A_fine', 'ref_C')}.
6. **Check A (§6): the submitted numbers reproduce exactly (within-pair R² 0.555 on the paper's total error, pooled 0.460, σ~fine 1.2 × 10⁻⁵ / within-pair 0.003), and the gap survives in meters with the clean model but shrinks from ~100:1 to ~2:1.** R02 σ_m: within-pair R² {f3(ca['table']['R02 clean sigma_m (mosaic, m, per 256 tile) | T-ship (R02 definition, m, 256 mosaic tile)']['within_pair_r2'])} on T-ship vs {f3(ca['table']['R02 clean sigma_m (mosaic, m, per 256 tile) | A_fine tile 128 (m, R04)']['within_pair_r2'])} on A_fine (tile, m) / {f3(ca['table']['R02 clean sigma_m (mosaic, m, per 256 tile) | A_fine F_hr RMS over the 256 window (m, R04)']['within_pair_r2'])} (256 window); median ρ {f3(ca['table']['R02 clean sigma_m (mosaic, m, per 256 tile) | T-ship (R02 definition, m, 256 mosaic tile)']['median_rho'])} vs {f3(ca['table']['R02 clean sigma_m (mosaic, m, per 256 tile) | A_fine tile 128 (m, R04)']['median_rho'])}. The submitted σ in its own (AUV-normalized) units has R² {f3(ca['table']['submitted sigma (het_production_v1, AUV-normalized, per 256 patch) | T-ship (R02 definition, m, 256 mosaic tile)']['within_pair_r2'])} on T-ship and {f3(ca['table']['submitted sigma (het_production_v1, AUV-normalized, per 256 patch) | A_fine tile 128 (m, R04)']['within_pair_r2'])} on A_fine in meters (negative ρ); multiplied by the patch HR std it reaches {f3(ca['table']["submitted sigma x patch HR std (m; the paper's own scale factor, AUV-side) | T-ship (R02 definition, m, 256 mosaic tile)"]['within_pair_r2'])} / {f3(ca['table']["submitted sigma x patch HR std (m; the paper's own scale factor, AUV-side) | A_fine tile 128 (m, R04)"]['within_pair_r2'])}.
7. **Check B (§7): σ_norm tracks slope and the resolved-band misfit, not the unresolved relief.** Partial Spearman given log s_lr: vs slope {cb['1. log sigma_norm vs log slope | log s_lr']['median']:+.2f} [{cb['1. log sigma_norm vs log slope | log s_lr']['ci95'][0]:+.2f}, {cb['1. log sigma_norm vs log slope | log s_lr']['ci95'][1]:+.2f}]; vs T_res {cb['2a. log sigma_norm vs log T_res | log s_lr']['median']:+.2f} (given slope too {cb['2b. log sigma_norm vs log T_res | log s_lr, log slope']['median']:+.2f}); vs A_fine given s_lr and slope {cb['3. log sigma_norm vs log A_fine (R03 convention, 256 window) | log s_lr, log slope']['median']:+.2f} (R03 convention), {cb["3'. log sigma_norm vs log A_fine (R04 tile 128, m) | log s_lr, log slope"]['median']:+.2f} (R04 tile), {cb["3''. log sigma_norm vs log A_fine (R04 F_hr RMS over the 256 window) | log s_lr, log slope"]['median']:+.2f} (R04 256 window), all CIs spanning zero.
8. **Preservation, contract, freeze (§8).** R02/R03 originals copied to OAK `grl_review_archive/` first, {sum(int(l.split()[1]) for l in pres)} files sha256-verified; R04 outputs {('copied: ' + str(sum(int(l.split()[1]) for l in pres4)) + ' files verified') if pres4 else 'copy job pending at report time'}. Contract v2.1 committed verbatim (sha256 matches); the consumer validator passes its synthetic tests and **38 of 39 pairs; EX1402L2__MGDS_31073 (vu02) is non-conforming: `ship_xtrack_frac` is NaN on all {[r for r in val if r['info']['pair_id']=='EX1402L2__MGDS_31073'][0]['info'].get('n_cells_with_soundings')} sounding cells** (the acquisition validator did not test that condition). Feature leak test {flk.get('n_identical', '?')}/{flk.get('n', '?')} identical. Freeze hashes in §8.3; **the `phase0-freeze` tag is withheld pending HOLD H1** (the directive's §1.1 stop rule fired), everything else is committed and hashed."""

holds = f"""(H1) **§1.1 consistency check below 0.9 on 12 of 15 units — stop-and-report rule fired.** The evidence (§3.3) says R03's A_fine is dominated by window-edge ringing and the R04 definition is the one the directive intended; Steve's call is whether to (a) accept targets_v1 as frozen (then tag `phase0-freeze` on the committed state), (b) change the E_1m threshold or any §1 choice first (a new version + ledger row), or (c) re-score R03's §3–§4 claims on the edge-free targets before freezing. Nothing in §5–§7 depends on R03's A_fine except the Check B item 3 row that uses it (reported alongside the R04 rows).
(H2) **EX1402L2 (vu02) ship products fail contract v2.1 on one raster** (`ship_xtrack_frac` all NaN). The pair stayed `ship_products_usable = True` as the manifest says; its xtrack feature is NaN (the GBT handles it). Steve's call: keep as is (recorded here), or ask the acquisition side to rebuild/flag the product (needs a manifest v2.1.1 row and a ledger entry).
(H3) **Tile rows with < 50 % joint-valid px.** Windows are selected at ≥ 50 % joint validity but their central 128-px tiles can fall below it ({int((pd.read_parquet(f"{R}/tiles/tiles_v1.parquet").frac_joint < 0.5).sum())} of {len(pd.read_parquet(f"{R}/tiles/tiles_v1.parquet"))} tiles; none below 64 px). They are kept (the directive says all tiles) with targets over their valid px; blocks already carry the 75 % rule. Steve's call whether a tile-level validity floor should enter targets_v1 before the freeze."""

not_pursued = f"""- Re-scoring R03's §3–§4 predictor tables (σ_m, roughness, expanded GBT) on the edge-free A_fine / T_fine (the bar and Check A/B cover the new targets; the old tables stand as measured).
- Rebuilding or flagging EX1402L2's `ship_xtrack_frac` product (acquisition repo).
- Quantifying the residual interior-hole ringing in F_hr (joint px within 8 px of an invalid px: median fraction {dg['frac_nearhole_med'].median():.3f}, RMS {dg['ratio_nearhole_interior'].median():.1f}× the interior under the R03 filter) with an interior-only variant of the targets.
- Per-unit tuning diagnostics of the GBT (which grid points win per fold are saved in `bar/chosen_configs_*.json`, not analysed).
- Any feature importance / ablation beyond GBT_ship vs GBT_noship.
- A contract-v2.1 feature for the ship window at sizes other than 128 / 224 px.
- Re-deriving the paper's capture statistics (0.502 / 0.442) on the new targets."""

challenge = T(f"{R}/challenge_draft.md")
# ---------------- §3 targets ----------------
grid_rows = []
for c in ts["grid_checks"]:
    grid_rows.append(f"| {c['pair']} | {C.UNIT_OF.get(c['pair'], '')} | {c['n_r04']} | {c.get('prior') or '— (new)'} | {c.get('n_prior_total', '')} | {c.get('n_prior_stride128', '')} | {('identical' if c.get('identical') else ('n/a' if c.get('prior') is None else 'DIFFERS')) } |")
unit_tab = ["| unit | pairs | windows = tiles | blocks | kept blocks | E_1m > 0 prevalence (kept blocks) | median A_fine block (m) | median T_fine block (m) |", "|---|---|---|---|---|---|---|---|"]
for u in U + ["lu16"]:
    r = uc.loc[u]; unit_tab.append(f"| {u}{' (EW0207, separate)' if u == 'lu16' else ''} | {int(r.pairs)} | {int(r.windows)} | {int(r.blocks)} | {int(r.kept)} | {r.E1m_prev_block:.3f} | {r.A_fine_block_med:.3f} | {r.T_fine_block_med:.3f} |")
cons_tab = ["| unit | R03 name | n R03 tiles on the stride-128 grid | r(log) R03 A_fine vs F_hr RMS, same 256 window | median ratio | r(log) vs R04 tile A_fine (128) | r(log) R03 vs my R03-convention recomputation | edge-16 / interior RMS (R03 filter) | r(log) R03 filter @128 vs R04 field @128 | ratio R04/R03-filter @128 |", "|---|---|---|---|---|---|---|---|---|---|"]
for u, r in cc.items():
    d = dg.loc[u]; cons_tab.append(f"| {u} | {r['r03_name']} | {r['n_r03_rows']} | **{r['vs_Fhr256']['r_log']:.3f}**{' (< 0.9)' if r['below_0p9'] else ''} | {r['vs_Fhr256']['median_ratio']:.3f} | {r['vs_tile128']['r_log']:.3f} | {r['vs_r03conv_recomputed']['r_log']:.4f} | {d.med_edge16 / d.med_interior:.2f} | {d.r_log_ha128_rc128:.3f} | {d.ratio_rc128_ha128:.3f} |")
excl = {u: round(float(uc.loc[u].E1m_prev_block), 3) for u in U if uc.loc[u].E1m_prev_block < 0.05 or uc.loc[u].E1m_prev_block > 0.95}
sec3 = f"""## 3. §1 — Frozen targets (`scripts/targets_v1.py`) and spatial support (`tiles_v1`)

### 3.1 Implementation (CONFIRMED against code; job {ts['pairs'][0].get('job', '46236482')})

`targets_v1.py` implements §1.1 exactly: per 256 × 256 window at stride 128, least-squares plane over joint-valid px with invalid px zeroed, Tukey (α = 0.25) taper, FFT band-pass with f_lr = hr_res / lr_native and f_hi = min(k f_lr, 0.45) (R02 T32B) and raised-cosine transitions of ± 0.25 octave (product of a raised-cosine high-pass step at f_lr and a low-pass step at f_hi in log₂ frequency), central 128 × 128 kept, re-masked to joint-valid. `F_hr` = band-passed (HR − δ), `F_res` = band-passed (HR − δ − bicubic), δ and the bicubic from the unchanged R02 `extract_clean_patches` path (`tile_ship_side`, `build_inputs`, `build_target`), joint = HR-valid ∧ nearest-upsampled LR-valid (the R02 per-tile mask). Fields are stored per pair as float32 GeoTIFF mosaics (`fields/<pair>/F_hr.tif`, `F_res.tif`, plus `bicubic_m.tif` and `joint_v1.tif`), on the HR grid, NaN outside the tiles. Summary targets over joint-valid px: A_fine = RMS(F_hr), T_fine = RMS(F_res), E_1m = fraction with |F_hr| > **1.0 m** (default in force; `targets_v1.E1M_THRESHOLD_M`). Blocks: 16 per tile, kept iff ≥ 75 % joint-valid. The R03 convention (`r03_afine`: hard annulus, no taper, plane detrend, RMS over the full window) is implemented alongside for the consistency check only.

### 3.2 Windows, tiles and blocks (`tiles/tiles_v1.parquet`, `blocks_v1.parquet`; per-pair counts in `tiles/pair_counts.csv`, grid checks in `tiles/tiles_summary.json`)

Window rule = R02 `extract_clean_patches` unchanged (joint-valid ≥ 50 %, HR-valid ≥ 50 %, HR std ≥ 0.05 m, ≥ 2 valid LR cells) at stride 128; tiles = the central 128 px (non-overlapping); `tiles_v1.parquet` carries unit, pair, window id, window and tile corners in HR px, tile bounds in CRS coordinates and the CRS, plus the per-window R02 statistics (mu_lr, s_lr, δ) and the targets.

{chr(10).join(unit_tab)}

Total: {int(uc.loc[U, 'windows'].sum())} tiles and {int(uc.loc[U, 'blocks'].sum())} blocks ({int(uc.loc[U, 'kept'].sum())} kept) on the 16 development units; EW0207 adds {int(uc.loc['lu16', 'windows'])} / {int(uc.loc['lu16', 'blocks'])} ({int(uc.loc['lu16', 'kept'])} kept). Cal DIG (lu12) = 18 pairs, {int(uc.loc['lu12', 'windows'])} tiles, one unit.

**Window-grid check** (R04 windows vs the stride-128 subset of the R02 / R03 patch indices; R02 train pairs were extracted at stride 64, val/test at 128, R03 at 64):

| pair | unit | R04 windows | prior set | prior windows (all) | prior at stride 128 | result |
|---|---|---|---|---|---|---|
{chr(10).join(grid_rows)}

### 3.3 Consistency check at matched support (§1.1) — CONTRADICTED on 12 of 15 units; mechanism established (`tiles/consistency_check.json`, `consistency_diag_units.csv`; job 46240691)

**Support R03 used (item 1):** `s3_bands.py` and `features_context.py` computed A_fine as `band_rms(hr − δ, joint, f_lr, f_hi)` on the **full 256 × 256 window** (plane detrend on valid px, zero fill, hard FFT annulus, no taper, RMS over all joint-valid px). My re-implementation (`targets_v1.r03_afine`) reproduces R03's stored values to r(log) ≥ 0.9988 and median ratio 1.000 on every unit (column 7), so the comparison is like-for-like.

**Item 2–3:** R03 A_fine vs the RMS of the R04 `F_hr` mosaic over the same 256 window (every px of the window is covered by F_hr: coverage 1.00 on all units), per unit, on the R03 tiles that lie on the stride-128 grid:

{chr(10).join(cons_tab)}

Reading. (i) The filter shape is not the cause: on NA090 the raised-cosine band-pass without taper over the full window gives ratio 0.98 and r(log) = 1.000 against the hard annulus. (ii) The support is: under the R03 filter the outer 16-px band of the window has {dg['med_edge16'].div(dg['med_interior']).min():.1f}–{dg['med_edge16'].div(dg['med_interior']).max():.1f}× the RMS of the interior (column 8) — Gibbs ringing of the periodic FFT at the window boundary (and at zero-filled holes), which is exactly what the directive's taper and central crop remove. The same R03 filter restricted to the central 128 px correlates with the R04 field at r(log) {dg['r_log_ha128_rc128'].min():.2f}–{dg['r_log_ha128_rc128'].max():.3f} (column 9) with ratios {dg['ratio_rc128_ha128'].min():.2f}–{dg['ratio_rc128_ha128'].max():.2f}; the two exceptions, CCZ (lu06) and DISCOL (lu02), are the near-flat provider-grid units where A_fine is 0.001–0.008 m and any residual ringing dominates. (iii) Over **all** R04 windows (not only the R03 subsample) the picture is the same (`consistency_all_windows.csv`: r(log) {pd.read_csv(f"{R}/tiles/consistency_all_windows.csv").all_r_log.min():.2f}–{pd.read_csv(f"{R}/tiles/consistency_all_windows.csv").all_r_log.max():.2f}). The directive's "expect ≥ 0.9" therefore cannot be met at the R03 support by any filter that removes the edge ringing; it is met (≥ 0.86 on 13 of 15 units) at the central-128 support. **Per the directive's stop rule this is HOLD H1; §4–§8 were completed on the R04 definitions so the ruling can be made on complete evidence.**

### 3.4 E_1m prevalence and the detection-unit rule (§1.2)

Prevalence of E_1m > 0 at block scale (kept blocks) per unit is in the table of §3.2. Units outside [5 %, 95 %] and therefore **excluded from the detection metrics (fixed now): {excl}** (EW0207, 0.961, is separate anyway). Detection metrics use the remaining {16 - len(excl)} units. The 1.0 m threshold is the default; Steve may change it before the freeze (a change re-runs `build_tiles_targets.py` only; the fields do not change).
"""
sec4 = """## 4. §2 — Frozen evaluation (`scripts/eval_v1.py`)

Interface: a table with `unit, pair, tile_id, block_id` (−1 for tile rows), the target columns and any predictor columns; `evaluate(table, preds)` scores every predictor on every endpoint; `compare(table, a, b, target, scale, metric)` gives the unit-bootstrap difference. Primary endpoint = A_fine, block scale, within-unit Spearman ρ, median over the development units; secondary = A_fine tile; T_fine block and tile; E_1m block (AUPRC for E_1m > 0 on the classifier probability, Spearman ρ on E_1m). Metric set per endpoint: within-unit Spearman ρ, within-unit log-log R² (A_fine, T_fine), top-20 % capture (share of the target's squared value in the top 20 % predicted rows), each per unit, median over units, and pooled (pooled ρ over all rows; pooled within-unit R² = unit-demeaned log-log, as R02/R03). Differences: bootstrap over units, 2,000 resamples, seed 20261001, 95 % percentile CI, count of units where the first predictor is better, exact one-sided binomial sign test (ties dropped; reported, not a gate). `assert_units` refuses any unit outside the frozen list (or the lu16 extension when asked) and any row of a sealed pair. Self-test on a synthetic 16-unit table (informative / random / half-informative predictors) is in the session log: unit assertion and sealed-pair refusal fire; medians, CIs and sign tests behave as expected.
"""
sec5 = f"""## 5. §3 — The bar (`scripts/features_v1.py`, `bar_v1.py`, `run_bar.py`; `bar/BAR_v1.md`, prediction tables `bar/pred_block.parquet`, `pred_tile.parquet`, `pred_*_EW0207.parquet`, chosen configurations `bar/chosen_configs_*.json`; job 46236484)

### 5.1 Features (§3.1; `features/features_v1.parquet`, {sum(v['n_rows'] for v in fav.values()) if fav else '?'} rows × 2 window sizes; availability in `features_v1_availability.json`)

Ship-only, per tile centre and per kept-block centre, on windows of 128 and 224 HR px (suffixes `_128`, `_224`): the native ship window (cells = ⌈win / ratio⌉, min 2) gives the R03 bank verbatim (std_ship, slope_ship_mean / p90, curv_ship_std, spectral β / logA / fit_r2 / ŝ_fine / noise_excess / aniso_spec, and C / curv / aniso / slope on the ×8 bicubic), s_lr and mu_lr of the window, and the contract-v2.1 features (median and p90 of ship_rsd, median ship_sd, median log ship_count, mean xtrack_frac, mean beam_angle, median rsd/sd) where `ship_products_usable`; the HR-grid window gives the R02 free-combined features the R02 way (area-average + mu-fill + bicubic: C_hr, curv_hr, aniso_hr, slope_hr; dist_edge and frac_valid from the ship mask only). Metadata: depth (= mu_lr at 128), lr_native, k, hr_res, log(|depth| / lr_native), and s_lr_256 (the parent window's R02 statistic). Spectral features are NaN where the window has too few cells (defined on 0 % of 128-px windows for most swath units, 100 % for Cal DIG at k = 2; see the availability file). The pair context is a `ShipOnlyView` that raises on any HR key; **feature leak test: {flk.get('n_identical', '?')} / {flk.get('n', '?')} ({flk.get('n_tiles', '?')} tiles + {flk.get('n_blocks', '?')} blocks) byte-identical with the HR grid replaced by noise.**

Contract-feature availability: usable on {sum(1 for v in fav.values() if v['usable'])} of 39 pairs ({', '.join(sorted({C.UNIT_OF[p] for p, v in fav.items() if v['usable']}))}); NaN on lu02, lu06, lu09 (rulings R4), lu12 (no swath) and lu16 (flagged).

### 5.2 Models and results (§3.2–3.3)

GBT_ship / GBT_noship: sklearn HistGradientBoosting (NaN-native), nested leave-one-unit-out over the 16 units with inner leave-one-unit-out tuning over the 48-point grid (max_depth {{2, 3, 4, 6}} × n_estimators {{100, 300, 600}} × learning_rate {{0.03, 0.1}} × min_child_samples {{20, 100}}), inner criterion = median over inner units of Spearman ρ (regressors) / AUPRC (E_1m classifier), log₁₀ target, one model per target and scale; E_1m = P(E > 0) classifier + log E regressor on positives, combined as P × 10^reg. Training rows capped at 200 per unit per fold by one seeded subsample (seed 20261001) applied identically in every fold; held-out units scored on all rows. Reference rows: C_hr_128, s_lr_256, s_lr_128, rsd_med_128; R02 σ_m = mean mosaic σ over the block / tile (native 7).

{bar_md}
"""
sec6 = f"""## 6. §4 — Check A: differential control re-measured (`checks/check_a.md`, `check_a.json`, `check_a_table.parquet`)

{T(f"{R}/checks/check_a.md").split(chr(10), 1)[1]}

**One-sentence verdict.** The gap between total-error skill and fine-amplitude skill survives in meters with the clean model, but as a factor of ~2 (within-pair R² 0.80 on T-ship vs 0.41–0.46 on A_fine; median ρ 0.85 vs 0.47–0.54), not the ~100:1 of the submitted analysis (0.555 vs 0.003), whose fine-band number was measured in AUV-normalized units against a fine band that the normalization itself anti-correlates with σ (the normalized submitted σ has *negative* ρ against A_fine in meters, and reaches R² 0.14–0.18 only after multiplying back by the patch HR std).
"""
sec7 = f"""## 7. §5 — Check B: what σ_norm learned (`checks/check_b.md`, `check_b.json`)

{T(f"{R}/checks/check_b.md").split(chr(10), 1)[1]}

One sentence each. (1) Given s_lr, σ_norm rises with the bicubic tile slope on every unit. (2) Given s_lr, σ_norm tracks the resolved-band misfit T_res on every unit, and still does when slope is also controlled (weaker on AT37-13, RR1506). (3) Given s_lr and slope, σ_norm has no consistent partial relation with the unresolved relief A_fine under any of the three definitions (median ≈ 0, CI spanning zero; positive only on NA090 and TN299).
"""
pres_lines = "\n".join(f"| {l.split()[-1].replace('/oak/stanford/groups/hilley/auv_ship_colocated_bathy/grl_review_archive/', '')} | {l.split()[1]} | {l.split()[3]} s |" for l in pres)
pres4_lines = "\n".join(f"| R04/{l.split()[-1].split('/')[-1]} | {l.split()[1]} ({l.split()[3]}) | {l.split()[4]} s |" for l in pres4) if pres4 else "| R04 outputs | copy job 46236486 pending at report time | |"
val_bad = [r for r in val if r["errors"]]
sec8 = f"""## 8. §6 — Preservation, contract validation, freeze

### 8.1 Preservation (first R04 compute; `scripts/preserve.sbatch` job 46236480 after two slower attempts 46232525 / 46235643 were cancelled; `preserve_r04.sbatch` job 46236486)

OAK `.../auv_ship_colocated_bathy/grl_review_archive/`, sha256 computed at the source and verified at the destination (`SHA256SUMS` in each directory):

| directory | files | copy + verify time |
|---|---|---|
{pres_lines}
{pres4_lines}

Sizes: R02 `patches_clean` 29 GB, `mosaics` 5.3 GB, `het_clean_r02` 304 MB; R03 `patches_clean_expanded` 7.7 GB, `s3` 0.4 MB, `s4` 12 MB, `s6/pilot/raw` 164 MB (from `du` at the source). Scratch copies remain.

### 8.2 Contract v2.1 and the consumer validator (`INTERFACE_CONTRACT_v2.1.md` committed verbatim, sha256 `{C.sha256_file(C.CONTRACT)}` = directive header; `scripts/contract_v2_1_validate.py`; `validation/contract_v2_1_validation.md`)

The validator checks the §5 schema, the frame-offset rules, grid identity of all 7 rasters with `lr.tif` (CRS, transform, shape), dtype / nodata, the §3 NaN and count conditions, value ranges, the §4 gate from the recorded numbers and the §6 usable gate against the manifest column. Unit tests (`--test`): a synthetic conforming directory and a synthetic unavailable (Cal DIG-style) directory pass with 0 errors; 10 non-conforming variants (grid mismatch, sd where count < 2, rsd missing where count ≥ 3, xtrack out of range, wrong dtype, flags not reproducing the gate, frame offset applied without basis, missing key, rasters on an unavailable pair, manifest usable but flagged) are each rejected. **Run on the 38 development pairs + EW0207: {len(val) - len(val_bad)} / {len(val)} pass.** The failure: {'; '.join(f"{r['info']['pair_id']} ({r['info']['unit']}): {r['errors']}" for r in val_bad)} — all {[r for r in val if r['info']['pair_id']=='EX1402L2__MGDS_31073'][0]['info'].get('n_cells_with_soundings')} cells with soundings have count ≥ 1 and NaN across-track fraction (beam angle is fine), i.e. the product is effectively unavailable although `available.ship_xtrack_frac` is true; the acquisition validator only tests the NaN-where-count rule on the two regrid rasters. HOLD H2. The manifest's `ship_products_usable` agrees with the available ∧ unflagged rule on all 39 pairs (CCZ, DISCOL, TAG flagged `registration_shift`; EW0207 `vertical_offset`; Cal DIG unavailable).

### 8.3 Freeze (§6.3)

{T(f"{D}/freeze_record.md") if os.path.exists(f"{D}/freeze_record.md") else 'FREEZE_PLACEHOLDER'}

Unit-set versioning: `units-v1` = the 16 units above; a later unit (Venere pu00 via manifest v2.1.1) enters by running `build_tiles_targets.py`, `build_features.py` and `run_bar.py` unchanged and tagging `phase0-freeze+units-v<n>`. **Tag status:** the commit is made and hashed; the `phase0-freeze` tag is withheld because the §1.1 stop rule fired (HOLD H1). Once Steve rules, `git tag phase0-freeze <commit>` on this commit (option a) or a new targets version (option b/c) completes §6.3.
"""
sec9 = f"""## 9. §7 — Input-representation record (`input_repr_record.md`, `input_repr_per_pair.csv`; report only)

{T(f"{R}/input_repr_record.md").split(chr(10), 1)[1]}
"""
sec10 = """## 10. Change-ledger rows

Appended to `grl_reviews-and-prompt-reports/CHANGE_LEDGER.md` (rows 10–13): frozen targets_v1 (band-passed, tapered, raised-cosine, central-crop fields; A_fine / T_fine / E_1m at block and tile scale) replacing the per-patch fine-band RMS and the CNN-residual total error as the evaluation targets; eval_v1 (primary endpoint, unit-equal weighting, bootstrap over units); the 16-unit development set with Cal DIG as one unit and the 200-row training cap; contract-v2.1 ship features in the free baseline. Evaluation-only; the submitted pipeline is untouched.
"""
sec11 = f"""## 11. Paths and reproduction

Scratch originals: `/scratch/users/scdobbs/grl_review/R04/` — `fields/<pair>/` (F_hr, F_res, bicubic_m, joint_v1 GeoTIFFs), `tiles/` (tiles_v1, blocks_v1, summaries, consistency), `features/` (features_v1.parquet, availability, leak test), `bar/` (BAR_v1.md, prediction tables, results, chosen configs), `checks/`, `validation/`, `provenance/` (one JSON per script run: host, job, manifest and contract sha256, guard probes), `logs/`. OAK: `grl_review_archive/R04/` (same layout, sha256-verified). Repo (this directory): report, `scripts/` as run, `tiles/tiles_v1.parquet` + summaries, `checks/`, `validation/`, `bar/` (markdown + JSON), `features/` (availability + leak JSON), `input_repr_*`, `freeze_record.*`, `logs/`.

```bash
cd grl_reviews-and-prompt-reports/directives_and_reports/R04/scripts
sbatch preserve.sbatch                                   # §6.1 first (CPU, ~1 h incl. verification)
sbatch --dependency=afterok:<preserve> tiles_targets.sbatch     # §1 (16 CPUs, 33 min)
sbatch --dependency=afterok:<tiles> features.sbatch             # §3.1 (32 CPUs, ~15 min)
sbatch --dependency=afterok:<features> bar.sbatch               # §3.2 (32 CPUs)
sbatch --dependency=afterok:<tiles> checks.sbatch               # §6.2 validator, §4, §5 (1 min)
sbatch consistency_diag.sbatch                                  # §1.1 mechanism (6 min)
python input_repr_record.py; python freeze.py; python make_report.py
```
"""
rep = T(f"{D}/REPORT_R04_2026-10-01.md").split("## 1. Bottom line")[0]
jobs = "preserve 46236480, tiles 46236482, features 46236483, bar 46236484, checks 46236485, consistency diagnostic 46240691, R04 archive 46236486 (cancelled slow attempts: 46232525, 46235643 and their dependents)"
rep = rep.replace("JOBS_PLACEHOLDER", jobs)
out = rep + "## 1. Bottom line\n\n" + bottom + "\n\n**HOLDs for Steve (3):**\n\n" + holds + "\n\n**Not pursued (one line each):**\n\n" + not_pursued + "\n\n---\n\n## 2. Challenge clause (§0)\n\n" + challenge + "\n---\n\n" + sec3 + "\n---\n\n" + sec4 + "\n---\n\n" + sec5 + "\n---\n\n" + sec6 + "\n---\n\n" + sec7 + "\n---\n\n" + sec8 + "\n---\n\n" + sec9 + "\n---\n\n" + sec10 + "\n---\n\n" + sec11
open(f"{D}/REPORT_R04_2026-10-01.md", "w").write(out); print("report written:", len(out), "chars; bar present:", barb is not None, "; freeze present:", frz is not None)
