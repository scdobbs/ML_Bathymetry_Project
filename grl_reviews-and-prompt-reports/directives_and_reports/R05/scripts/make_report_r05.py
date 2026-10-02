"""Assemble REPORT_R05 from the persisted artifacts plus the hand-written text blocks. Run after the bar and report tables."""
import json, os, sys, glob
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import r05_common as C5, r04_common as C
R = C5.ROOT; D = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); J = lambda p: json.load(open(p)); T = lambda p: open(p).read()
ts = J(f"{R}/tiles/tiles_v1_1_summary.json"); pu = pd.read_csv(f"{R}/tiles/tiles_v1_1_removed_per_unit.csv").set_index("unit"); pp = pd.read_csv(f"{R}/tiles/tiles_v1_1_removed_per_pair.csv")
st = J(f"{R}/eval/eval_v1_1_selftest.json"); av = J(f"{R}/features_v1_1_availability.json"); prov = J(f"{R}/sigma_m_provenance.json"); rules = J(f"{R}/bar/pair_rules.json")
bt, bb = J(f"{R}/bar/bar_results_tile.json"), J(f"{R}/bar/bar_results_block.json"); rt = J(f"{R}/bar/report_tables_v1_1.json"); frz = T(f"{D}/freeze_record.md")
pres = [l.strip() for f in sorted(glob.glob(f"{R}/logs/r05_preserve_*.out")) for l in T(f).splitlines() if l.startswith("VERIFIED")]
U = list(C.UNITS); multi = [u for u in U if len(C.UNITS[u]) > 1]
def prim(res, key):
    ci = res[key]["primary_ci"]; return f"{ci['value']:.3f} [95 % CI {ci['ci95'][0]:.3f}, {ci['ci95'][1]:.3f}] ({ci['n_units']} units)"
def cmpl(res, key, a, b, metric="rho"):
    c = next(c for c in res[key]["comparisons"] if c["a"] == a and c["b"] == b and c["metric"] == metric)
    return f"{c['diff_median']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}], {c['units_a_better']} of {c['n_units']} units better (sign-test p = {c['sign_test_p_one_sided']:.3f})"
def med(res, key, p, metric="rho"): return f"{res[key]['scores'][p]['median'].get(metric, float('nan')):.3f}"
sb = lambda sc, p, k: rt[f"{sc}/{p}"][k]; f3 = lambda x: f"{x['value']:.3f} [{x['ci95'][0]:.3f}, {x['ci95'][1]:.3f}]"
ineligible = bt["A_fine/tile"]["scores"]["GBT_ship_A_fine"]["ineligible_pairs"]; det_exc = bb["E_1m/block"]["detection_excluded"]; det_units_dropped = bb["E_1m/block"]["scores"]["GBT_ship_E_1m"]["units_dropped"]
bottom = f"""1. **Challenge findings (§2): none blocking; 3 notes.** (i) The directive's "≥ 10 tiles with a finite prediction" cannot be evaluated inside a single-scale prediction table, so the pair sets are fixed once from the tile table (eligibility) and the block table (detection) and passed to every scoring call; (ii) the eval_v1 self-test comparison in §2.2.7 is done with eval_v1 imported from the frozen R04 code; (iii) the bar refit was submitted three times (two cancelled within minutes: a smoke-test exit-code slip and the pass-through fix above) before the run that counts, job 46251812.
2. **tiles_v1.1 (§3):** {ts['tiles_v1']} → {ts['tiles_v1_1']} tiles (−{ts['tiles_removed']}), {ts['blocks_v1']} → {ts['blocks_v1_1']} blocks (−{ts['blocks_removed']}, of which {ts['kept_blocks_removed']} had been kept); smallest removed tile {ts['min_n_joint_removed']} joint px; no pair falls below 10 tiles. Pair rules in force: eligible pairs {len(rules['eligible']['A_fine'])} / 38 on A_fine (ineligible: {ineligible or 'none'}); detection pairs {len(rules['detection_pairs'])} / 38 (excluded: {len(det_exc)} pairs, listed in §4.1; units with no detection pair: {det_units_dropped}).
3. **The Phase 2 bar (§4.1): GBT_ship, A_fine at tile scale under eval_v1.1 = {prim(bt, 'A_fine/tile')}.** GBT_ship − GBT_noship: {cmpl(bt, 'A_fine/tile', 'GBT_ship_A_fine', 'GBT_noship_A_fine')}. Secondary: A_fine block {prim(bb, 'A_fine/block')}; T_fine tile {prim(bt, 'T_fine/tile')}, block {prim(bb, 'T_fine/block')}; E_1m block ρ {prim(bb, 'E_1m/block')}, classifier median AUPRC {med(bb, 'E_1m/block', 'P_GBT_ship_E_1m', 'auprc')}. References at tile scale (median ρ): C {med(bt, 'A_fine/tile', 'ref_C')}, s_lr {med(bt, 'A_fine/tile', 'ref_s_lr')}, ship std 128 {med(bt, 'A_fine/tile', 'ref_shipstd128')}, rsd 128 {med(bt, 'A_fine/tile', 'ref_rsd128')}; GBT_ship − C: {cmpl(bt, 'A_fine/tile', 'GBT_ship_A_fine', 'ref_C')}.
4. **What changed from v1 (§4.2).** A_fine tile, GBT_ship: R04 bar under eval_v1 {f3(sb('tile', 'GBT_ship_A_fine', 'v1'))} → the same R04 predictions under eval_v1.1 {f3(sb('tile', 'GBT_ship_A_fine', 'v1_1_on_r04_pred'))} → R05 refit {f3(sb('tile', 'GBT_ship_A_fine', 'v1_1_refit'))}; GBT_noship {f3(sb('tile', 'GBT_noship_A_fine', 'v1'))} → {f3(sb('tile', 'GBT_noship_A_fine', 'v1_1_on_r04_pred'))} → {f3(sb('tile', 'GBT_noship_A_fine', 'v1_1_refit'))}. On the mixed-k units the sign flips disappear: lu15 GBT_noship {bt['A_fine/tile']['scores']['GBT_noship_A_fine']['per_unit']['lu15']['rho']:+.2f} (v1 −0.40), lu07 {bt['A_fine/tile']['scores']['GBT_noship_A_fine']['per_unit']['lu07']['rho']:+.2f}; per-pair table in §4.2.
5. **R02 σ_m provenance (§4.2, last table): CONFIRMED leave-one-unit-out on all 7 native units** (het_clean.py fold rule; each fold's per-patch record lists only the held unit's pairs; fold n_train equals the capped file count of all nine native pairs minus the held unit's, exactly, on every fold; mosaic_infer.py maps each unit to its own fold). Under eval_v1.1 σ_m scores {med(bt, 'A_fine/tile', 'sigma_m_R02')} (tile) / {med(bb, 'A_fine/block', 'sigma_m_R02')} (block) median ρ on the 7 units vs GBT_ship on the same units: tile {cmpl(bt, 'A_fine/tile', 'GBT_ship_A_fine', 'sigma_m_R02')}.
6. **Freeze (§6).** `r04-as-run` tagged on 8b1d52d; R05 code and tables committed and tagged **`phase0-freeze`**; hashes in §6; R05 outputs on OAK `grl_review_archive/R05/` ({sum(int(l.split()[1]) for l in pres)} files verified); ledger rows 14–19 appended; Venere removed; `units-v1` is the Phase 2 unit set."""
holds = "None. (The three R04 HOLDs are closed by the §1 rulings; nothing in R05 needs a decision.)"
not_pursued = f"""- Scoring the R05 refit under eval_v1 (only the R04 predictions were re-scored under v1.1, which isolates the aggregation change).
- Any look at why the refit chose different grid points (chosen configurations are saved in `bar/chosen_configs_*.json`).
- A per-pair table for the single-pair units or for T_fine / E_1m.
- Re-scoring R03 or Check A / Check B under eval_v1.1.
- Any change to the 10-tile eligibility floor or the [5 %, 95 %] detection bounds (fixed by the directive)."""
challenge = """1. **§2.2.4 "≥ 10 tiles with a finite target and a finite prediction", applied "at tile and block scale".** The bar's prediction tables hold one scale each, so a block table carries no tile rows from which to count. Implemented as: the eligible-pair set is computed once per target from the tile table (finite target; GBT predictions are always finite, reference rows are NaN where undefined so they are checked per predictor on the tile rows) and the detection set once from the block table, then passed explicitly to every `score_one` / `compare` / `median_ci` call (`elig_pairs`, `det_pairs`); when not passed, `eval_v1_1` computes them from the table it is given. Both sets are saved in `bar/pair_rules.json`.
2. **§2.2.7 self-test "positive under v1".** Done with the frozen R04 `eval_v1` imported alongside (`eval/eval_v1_1_selftest.json`): a predictor carrying only the pair level scores median ρ −0.005 under v1.1 and +0.54 to +0.79 on each multi-pair unit under v1 (unit-pooled); an informative predictor keeps +0.83 under v1.1; noise ≈ 0.
3. **Job bookkeeping.** The refit was first submitted after a smoke test whose failure was hidden by a pipeline exit code (job 46250483, cancelled at once), then again before the pair-rule pass-through fix (46250876, cancelled within 3 min); the run that counts is 46251812. No results from the cancelled jobs were used.
4. **§3.4 provenance evidence.** The mosaic job log does not name the fold per unit; the mapping is hard-coded in `mosaic_infer.py` (`MODEL_FOR = fold_<unit>`), which with the per-fold held-pair records and the exact n_train arithmetic is taken as CONFIRMED rather than COULD NOT CHECK."""
sec3 = f"""## 3. §2.1 — tiles_v1.1 (`tiles/tiles_v1_1.parquet`, `blocks_v1_1.parquet`; counts in `tiles_v1_1_removed_per_pair.csv`, `_per_unit.csv`)

Selection on R04's tiles_v1 / blocks_v1 (fields and `targets_v1.py` untouched): tile kept iff ≥ 50 % of its central 128 × 128 px are joint-valid; blocks of dropped tiles dropped; the 75 % block rule unchanged.

| unit | tiles v1 | removed | tiles v1.1 | blocks v1 | removed (of which kept) | blocks v1.1 | kept blocks v1.1 |
|---|---|---|---|---|---|---|---|
""" + "\n".join(f"| {u}{' (EW0207)' if u == 'lu16' else ''} | {int(r.tiles_v1)} | {int(r.tiles_removed)} | {int(r.tiles_v1_1)} | {int(r.blocks_v1)} | {int(r.blocks_removed)} ({int(r.kept_blocks_removed)}) | {int(r.blocks_v1_1)} | {int(r.kept_blocks_v1_1)} |" for u, r in pu.iterrows()) + f"""

Totals: {ts['tiles_removed']} tiles and {ts['blocks_removed']} blocks removed (smallest removed tile {ts['min_n_joint_removed']} joint px); per pair in the CSV (largest: NA080__MGDS_31289 51, TN399__MGDS_30373 45, EW0207 22). No pair falls below the 10-tile eligibility floor.

## 3b. §2.2 — eval_v1_1 (`scripts/eval_v1_1.py`) and §2.3 — the availability override (`scripts/features_v1_1.py`, `features_v1_1_availability.json`)

`eval_v1_1.py` is `eval_v1.py` with the §2.2 changes only (docstring lists them): primary endpoint A_fine / tile; within-pair metrics → unit median over eligible pairs → median over units; bootstrap over unit values; pair eligibility ≥ {rules and 10} tiles; detection per pair; pooled metrics kept and labelled secondary. Self-test (§2.2.7): """ + "; ".join(f"{p}: v1.1 median ρ {r['v1_1_median_rho']:+.3f} vs v1 {r['v1_median_rho']:+.3f} (multi-pair units under v1: " + ", ".join(f"{u} {v:+.2f}" for u, v in r['v1_multi_units'].items()) + ")" for p, r in st.items()) + f""".

`features_v1_1.py` = R04 `features_v1.py` plus `OVERRIDES = {{(EX1402L2__MGDS_31073, ship_xtrack_frac): "all-NaN where count ≥ 1 (R04 H2)"}}`, applied in `contract_features` (the product is masked to NaN for that pair). The R04 feature table was not recomputed: the availability report verifies that `xtrack_mean_128/224` are already NaN on all {av['EX1402L2__MGDS_31073']['n_rows']} rows of the pair while its other contract features are defined on 100 %; usable pairs {sum(1 for v in av.values() if v['usable'])} / 39. Validator result for the pair: fails on one raster, overridden by R05 ruling H2.
"""
sec4 = f"""## 4. §3 — Reporting items

### 4.1 The bar under eval_v1.1 (`bar/BAR_v1_1.md`, prediction tables `bar/pred_tile.parquet`, `pred_block.parquet`, `pred_*_EW0207.parquet`, `chosen_configs_*.json`; job 46251812)

{T(f"{R}/bar/BAR_v1_1.md").split(chr(10), 2)[2]}

### 4.2–4.4 v1 vs v1.1 side by side, per-pair ρ on the multi-pair units, R02 σ_m provenance (`bar/report_tables_v1_1.md`)

{T(f"{R}/bar/report_tables_v1_1.md")}
"""
pres_lines = "\n".join(f"| {l.split()[-1].replace('/oak/stanford/groups/hilley/auv_ship_colocated_bathy/grl_review_archive/', '')} | {l.split()[1]} ({l.split()[3]}) | {l.split()[4]} s |" for l in pres)
sec6 = f"""## 5. §4 — Preservation

OAK `grl_review_archive/R05/`, sha256 computed at the source and verified at the destination (job {os.path.basename(sorted(glob.glob(f"{R}/logs/r05_preserve_*.out"))[-1]).split('_')[-1].split('.')[0] if glob.glob(f"{R}/logs/r05_preserve_*.out") else 'pending'}):

| directory | files (size) | time |
|---|---|---|
{pres_lines}

## 6. §5 — Freeze

{frz}
Tags: `r04-as-run` on 8b1d52d (R04 commit); `phase0-freeze` on the R05 commit (hash appended below after the commit). Ledger rows 14–19 (`CHANGE_LEDGER.md`): H1, H2, H3 rulings; eval_v1.1 with the §0 reasons; bar_v1.1; Venere removed / units-v1.

## 7. Paths and reproduction

Scratch `/scratch/users/scdobbs/grl_review/R05/` (tiles, bar, eval, provenance, logs); OAK `grl_review_archive/R05/`; repo R05 directory: report, `scripts/`, `tiles/` (tiles_v1_1.parquet + counts), `eval/`, `bar/` (markdown + JSON), `freeze_record.*`, `features_v1_1_availability.json`, `sigma_m_provenance.json`. Run order: `build_tiles_v1_1.py`, `eval_v1_1.py` (self-test), `availability_report.py`, `sigma_m_provenance.py`, `sbatch bar_v1_1.sbatch` (run_bar_v1_1 + report_tables_v1_1), `sbatch preserve_r05.sbatch`, `freeze_r05.py`, `make_report_r05.py`.
"""
head = f"""# REPORT R05 — Phase 0 close: rulings on the R04 HOLDs, eval_v1.1, bar_v1.1, and the freeze

**Directive:** `directives_and_reports/R05_directive_phase0_eval_v1_1_and_freeze_2026-10-01.md`
**Executor:** Claude Code on Sherlock, 2026-10-01 (the R04 session, continued).
**Compute:** CPU only, 0 GPU-h, no training. Jobs (serc): bar refit 46251812 (32 CPUs), preservation {os.path.basename(sorted(glob.glob(f"{R}/logs/r05_preserve_*.out"))[-1]).split('_')[-1].split('.')[0] if glob.glob(f"{R}/logs/r05_preserve_*.out") else 'pending'}; everything else on the login node (selection, self-tests, provenance; seconds each).
**Outputs:** `/scratch/users/scdobbs/grl_review/R05/` (copied to OAK `grl_review_archive/R05/`) and this directory.
**Certainty labels:** CONFIRMED / CONTRADICTED / COULD NOT CHECK. Meters unless marked.

---

## 1. Bottom line

{bottom}

**HOLDs for Steve:** {holds}

**Not pursued (one line each):**

{not_pursued}

---

## 2. Challenge clause

{challenge}

---

"""
out = head + sec3 + "\n---\n\n" + sec4 + "\n---\n\n" + sec6
open(f"{D}/REPORT_R05_2026-10-01.md", "w").write(out); print("report written:", len(out), "chars")
