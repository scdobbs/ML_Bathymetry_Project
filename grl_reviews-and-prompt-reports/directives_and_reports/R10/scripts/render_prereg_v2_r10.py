"""R10 Part C — render PREREG_phase4_v2.md from PREREG_phase4_v1.md (sha256-verified) by replacing the B2 item list tail, B3,
B4, B5 and B6 sections and adding the version note and the final 'changes from v1' section; the rulings header, B1, B2
items 1-3 and the lockbox history are carried over VERBATIM (string slices of v1). Bracketed fields come from the R10 results:
B_min (calibration.json), s_lr_W64 in force (cells.json), the development expectation (per_pair_D1.csv / per_pair_D125.csv),
the five scripts' sha256 (computed), the R08 wide_window_r08.py sha256 (R08 freeze record). --draft writes PREREG_phase4_v2_DRAFT.md
with the open fields marked [HOLD ...]."""
import argparse, json, os, re, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "phase4", "scripts"))
import phase4_common as P4
RB = P4.R9.RB; ROOT = P4.ROOT["rehearsal"]; OUTDIR = f"{P4.DR}/R10"
def section(text, head):
    i = text.index(head); j = text.find("\n## ", i + 1); return i, (j if j >= 0 else len(text))
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--draft", action="store_true"); ap.add_argument("--hold_a63", default=""); ap.add_argument("--hold_a65", default=""); a = ap.parse_args()
    v1 = open(P4.PREREG_V1).read(); assert P4.sha256_file(P4.PREREG_V1) == P4.PREREG_V1_SHA
    cal = json.load(open(f"{ROOT}/calibration/calibration.json")); cells = json.load(open(f"{ROOT}/cells/cells.json")); W8 = json.load(open(f"{P4.DR}/R08/freeze_record.json"))["scripts_sha256"]["wide_window_r08.py"]
    bmin = cal["B_min"]; bmin_txt = f"{bmin}" if bmin is not None else f"[HOLD A6.3: {a.hold_a63 or 'pooled coverage below 0.90 at every B'}]"
    inforce = cells["rule"]["in_force"]; hold65 = cells["rule"]["hold"]
    force_txt = "in force" if inforce else ("[HOLD A6.5: " + (a.hold_a65 or "confirmatory cells_across could not be checked from the manifest") + "]" if hold65 else "not in force")
    pp1 = pd.read_csv(f"{ROOT}/scores/per_pair_D1.csv"); pp2 = pd.read_csv(f"{ROOT}/scores/per_pair_D125.csv"); ten = [p for u in RB.TEN for p in P4.C4.UNITS[u]]
    def c1(pp): d = pp[pp.pair.isin(ten) & pp.status.str.startswith("in C1")]; return int(d.c1_condition.sum()), int(len(d))
    x1, y1 = c1(pp1); x2, y2 = c1(pp2); r = pp1[pp1.pair.isin(ten) & pp1.view.isin(["interior", "fallback"])].rho_s_lr
    scripts = {sc: P4.sha256_file(os.path.join(P4.HERE, sc)) for sc in P4.SCRIPTS}
    head = ("# PREREG_phase4_v2 — Phase 4 pre-registration, version 2 (frozen; tag `phase4-prereg-v2`)\n\n> **Version 2, written 2026-10-10 by directive R10. It supersedes `PREREG_phase4_v1.md` (sha256 `5c636f3d…4c1f`, tag `phase4-prereg`), which was never executed. "
            "No sealed file had been opened, listed or computed on when v2 was frozen (guard self-tests, REPORT_R09 and REPORT_R10). Changes from v1 are listed in the final section.**\n\n")
    # carry-over: everything from the rulings paragraph through the end of B2 item 3
    i0 = v1.index("**Written:**"); i_b2_end = v1.index("No other predictor is scored at Phase 4.")
    carried = v1[i0:i_b2_end]
    b2_4 = (f"4. **s_lr_W64 (secondary, labelled; {force_txt} by the rule of R10 §A6.5):** the frozen `features_v1.native_features` s_lr on the 64 × 64-cell window of the pair's context grid centred on the tile centre, as run in R08 §6 (`wide_window_r08.py`, sha256 `{W8}`); windows with < 50 % valid cells give NaN. It is not part of C1.\n\nNo other predictor is scored at Phase 4.\n\n")
    b3 = ("## B3. Order of operations at Phase 4 (each step a separate directive or job; nothing out of order)\n\n"
          f"1. **Acquisition repo:** build the sealed pairs' contract-v2.1 ship products with the frozen code. {'Also build their context grids with the frozen ACQ-R05 builder (s_lr_W64 is in force).' if inforce else ('[HOLD A6.5: whether the context grids are needed depends on the s_lr_W64 ruling.]' if hold65 else 'Context grids are not needed (s_lr_W64 is not in force).')} The manifest rows of the sealed pairs must carry `hr_native_res_m`, `lr_native_res_m`, `max_recoverable_k` and the harmonized paths before step 2 (`phase4_build.py` refuses a row without them).\n"
          "2. **CNN repo:** `phase4_build.py --mode phase4`. It also writes each pair's amplitude statistic (B4). **No prediction or score yet.**\n"
          "3. **Blind artifact judgement:** `phase4_figures.py --mode phase4` produces figure set A for the 5 sealed pairs, pages shuffled with seed 20261011 and coded Q01 to Q05, key in a separate file. Steve judges each page against the R08 §3.1 criteria and fills one row per page:\n\n"
          "   | field | values |\n   |---|---|\n   | `code` | Q01 to Q05 |\n   | `relief` | yes / weak / no |\n   | `criteria` | any of a, b, c, or none |\n   | `dominant` | relief / artifact / mixed |\n   | `ab_whole_survey` | yes / no / na: do stripes (a) or ribbing (b) extend across the whole survey? |\n   | `note` | free text |\n\n"
          "   The calls file is committed and its sha256 recorded in `CHANGE_LEDGER.md` **before** any prediction is computed.\n"
          "4. **Predict:** `phase4_predict.py --mode phase4`, on CPU.\n"
          "5. **Score once:** `phase4_score.py --mode phase4`. Report all results, whatever they are. No retuning, re-selection or re-scoring with changed rules afterwards.\n\n")
    b4 = ("## B4. Scoring rules\n\n- **Statistic:** the `eval_v1_1` within-pair Spearman ρ at tile scale. Each sealed pair is its own unit.\n"
          f"- **Blocks:** 4 × 4-tile blocks in tile-grid coordinates (r0 / 128, c0 / 128). A block is non-empty if it holds at least one scored tile. **B_min = {bmin_txt}.**\n"
          "- **Tiles:**\n  1. **Primary view:** interior tiles at D₁ (≥ 95 % of joint-valid px at ≥ D₁ = ⌈lr_native / hr_res⌉ px from any invalid px), scored against A_fine. Used if the pair has ≥ 10 interior tiles in ≥ B_min non-empty blocks.\n"
          "  2. **Fallback view:** tiles with a defined A_fine_int (≥ 2,048 interior px), scored against A_fine_int. Used if rule 1 fails and the pair has ≥ 10 such tiles in ≥ B_min non-empty blocks.\n"
          "  3. Otherwise the pair is **not scorable** and is outside C1. If either view has ≥ 10 tiles, its point ρ is reported, labelled \"too few blocks, not in C1\", with no CI.\n"
          "- **Uncertainty per pair:** a bootstrap over non-empty blocks, resampled with replacement to the same number of blocks, 2,000 resamples, a fresh `numpy.random.default_rng(20261009)` per pair, 95 % percentile CI. Resamples with an undefined ρ are counted and excluded; if they exceed 5 %, the CI is reported as not estimable and the pair is outside C1.\n"
          "- **Amplitude statistic:** the median of the scored target over the scored tiles of the view in use (m). It is computed at B3 step 2, before any prediction.\n"
          "- **Secondary rows, for every pair with ≥ 10 tiles on that view:** all-tile ρ against A_fine; A_fine_int ρ where the primary view is in use. They are reported beside the primary and do not enter C1.\n"
          "- **Artifact handling:**\n  - **artifact-dominated** (`dominant = artifact`): reported separately and outside C1;\n  - **low-amplitude artifact-affected** (`ab_whole_survey = yes` and amplitude statistic < 0.04 m): outside C1, and included in the second statement of C1;\n  - any other pair with criteria a, b or c is flagged and stays in.\n\n"
          "  The 0.04 m floor lies between the development groups: under interior scoring at D₁ the six artifact-affected units have unit medians of at most 0.029 m and the ten real-relief units at least 0.049 m (REPORT_R08b §5).\n\n")
    b5 = ("## B5. Criteria (fixed now)\n\n- **C1, primary: ranking transfers.** The **C1 set** is the confirmatory units that are scorable, not artifact-dominated and not low-amplitude artifact-affected. Its size is stated.\n"
          "  - **Confirmed** if s_lr's block-bootstrap CI lower bound is above zero on every unit of the C1 set.\n  - **Partially confirmed** if on some but not all.\n  - **Not confirmed** if on none.\n  - **Not testable** if the C1 set is empty.\n\n  The same reading is also stated with the low-amplitude artifact-affected units included.\n\n"
          f"  Expectation from development (R10 §A6.2, each pair as its own unit, held-out predictions): on the pairs of the ten real-relief units, s_lr met the C1 condition on {x1} of {y1} at D₁ and {x2} of {y2} at the 125 m distance; pair ρ ranged from {r.min():.2f} to {r.max():.2f}, median {r.median():.2f}, at D₁. Report each sealed unit's ρ and CI against these.\n"
          f"- **C2, secondary and descriptive: model complexity.** Per sealed unit, report GBT_ship − s_lr, P-V1 − s_lr and P-V1 − GBT_ship{' and s_lr_W64 − s_lr' if inforce else (' [and s_lr_W64 − s_lr, if the A6.5 HOLD puts it in force]' if hold65 else '')}. One sentence per comparison stating its direction across units. No superiority claim is made: with at most 3 units, even 3 of 3 gives a one-sided sign test of p = 0.125. State this in the result.\n"
          "- **C3, supporting:** C1 and C2 on the lockbox pairs, reported separately with their history label.\n\n")
    b6 = ("## B6. Provenance requirements at Phase 4\n\nEvery Phase 4 script runs in `phase4` mode only if `SIGNOFF_phase4_prereg_v2.md` exists and holds this document's sha256. Before running, it verifies that sha256, the candidate artifacts (B2), the frozen code (R05 §6; `targets_v1_int.py`; `bar_v1_1.py`; the R07 model code hash; `fields_r08.py`; `wide_window_r08.py` if in force) and the five Phase 4 scripts against this document:\n\n"
          "| script | sha256 |\n|---|---|\n" + "\n".join(f"| `{sc}` `{h}` |" for sc, h in scripts.items()) + "\n\nOutputs go to `grl_review_archive/phase4/` with `SHA256SUMS`. `phase4_score.py` refuses to run a second time.\n\n")
    final = ("## Changes from v1\n\n- **G1** — a pair counts in C1 only with ≥ B_min non-empty bootstrap blocks (B_min from R10 §A6.3); a pair with enough tiles but too few interior blocks uses the fallback target. Reason: a block bootstrap on a handful of blocks has no coverage.\n"
             "- **G2** — the interior-pixel score (A_fine_int) is reported beside the interior-tile score for every pair, as secondary. Reason: the two views agreed on the development units (R09 Part A) and the second row shows where they do not.\n"
             "- **G3** — a pair leaves C1 if Steve's blind call marks whole-survey stripes or ribbing and the median of its scored target is below 0.04 m; C1 is stated with and without such pairs. Reason: R08 / R08b showed the fine-band target on the six low-amplitude units is dominated by survey artifacts.\n"
             "- **G4** — if a sealed 256-px window holds fewer ship cells across than the development minimum, the 64-cell s_lr of R08 §6 enters as a labelled secondary row. Reason: the sealed ship cells (103–151 m) are coarser than any development unit.\n"
             "- the \"not testable\" outcome of C1 (v1 had no outcome for an empty set);\n- the calls-file fields (`code, relief, criteria, dominant, ab_whole_survey, note`) and the figure seed 20261011 with codes Q01–Q05;\n- the sign-off file `SIGNOFF_phase4_prereg_v2.md` and the single-run lock of `phase4_score.py`;\n- CPU inference for P-V1 (R10 §A4: CPU float32 against the R07 GPU bf16 values).\n")
    text = head + carried + b2_4 + b3 + b4 + b5 + b6 + final
    hold = (bmin is None) or hold65 or a.draft; name = "PREREG_phase4_v2_DRAFT.md" if hold else "PREREG_phase4_v2.md"; path = f"{OUTDIR}/{name}"
    open(path, "w").write(text); sha = P4.sha256_file(path)
    json.dump(dict(path=path, sha256=sha, draft=hold, B_min=bmin, s_lr_W64_in_force=inforce, hold_a65=hold65, expectation=dict(D1=[x1, y1], D125=[x2, y2], rho_min=float(r.min()), rho_max=float(r.max()), rho_median=float(r.median())), scripts_sha256=scripts), open(f"{ROOT}/prereg_v2_render.json", "w"), indent=1)
    print(f"{name} sha256 {sha}; B_min {bmin}; s_lr_W64 {force_txt}; draft={hold}")
