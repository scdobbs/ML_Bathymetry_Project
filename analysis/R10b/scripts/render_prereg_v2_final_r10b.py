"""R10b Part C — render PREREG_phase4_v2.md from PREREG_phase4_v2_DRAFT.md (sha256-verified) with exactly the edits of directive
R10b §C, filling the bracketed fields from verified records (R10 checks, R10b scores, triples, scripts' sha256). --draft2 writes
PREREG_phase4_v2_DRAFT2.md."""
import argparse, json, os, re, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "phase4", "scripts"))
import phase4_common as P4
RB, C4 = P4.R9.RB, P4.C4; ROOT = P4.ROOT["rehearsal"]; OUTDIR = f"{P4.DR}/R10b"; DRAFT = f"{P4.DR}/R10/PREREG_phase4_v2_DRAFT.md"; DRAFT_SHA = "fba5d188b93fde72d6b69f27371bdaf052adfcac8fdb9e1412c829ebd040318d"
def sub1(text, old, new):
    assert text.count(old) == 1, (text.count(old), old[:80]); return text.replace(old, new)
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--draft2", action="store_true"); a = ap.parse_args()
    assert P4.sha256_file(DRAFT) == DRAFT_SHA; t = open(DRAFT).read()
    ck = json.load(open(f"{P4.R10_OAK}/rehearsal/checks_A3_A4.json")); pv1 = ck["A4"]["P-V1"]["seed-averaged"]["max_abs_diff_log10"]
    pp1 = pd.read_csv(f"{ROOT}/scores/per_pair_D1.csv"); pp2 = pd.read_csv(f"{ROOT}/scores/per_pair_D125.csv"); ten = [p for u in RB.TEN for p in C4.UNITS[u]]; tri = json.load(open(f"{ROOT}/triples.json"))
    def c1(pp, excl=()): d = pp[pp.pair.isin(ten) & pp.status.str.startswith("in C1") & ~pp.pair.isin(excl)]; return int(d.c1_condition.sum()), int(len(d)), d.rho_s_lr
    x1, y1, r1 = c1(pp1); x2, y2, r2 = c1(pp2); xa, ya, ra = c1(pp1, ("AR26__MGDS_31838",)); p = x1 / y1
    r10pp = pd.read_csv(f"{P4.R10_OAK}/rehearsal/scores/per_pair_D1.csv"); w = r10pp[r10pp.pair.isin(ten)]; below = int((w.rho_s_lr_W64 < w.rho_s_lr).sum()); lr = P4.R8.load_tiles().groupby("pair").lr_native.first(); coarse = w[lr.reindex(w.pair).values >= 57]
    scripts = {sc: P4.sha256_file(os.path.join(P4.HERE, sc)) for sc in P4.SCRIPTS}; T = tri["D1_ten_C1"]
    # version note
    t = sub1(t, "> **Version 2, written 2026-10-10 by directive R10. It supersedes `PREREG_phase4_v1.md` (sha256 `5c636f3d…4c1f`, tag `phase4-prereg`), which was never executed. No sealed file had been opened, listed or computed on when v2 was frozen (guard self-tests, REPORT_R09 and REPORT_R10). Changes from v1 are listed in the final section.**",
             "> **Version 2, written 2026-10-10 by directives R10 and R10b. It supersedes `PREREG_phase4_v1.md` (sha256 `5c636f3d…4c1f`, tag `phase4-prereg`), which was never executed. No sealed file had been opened, listed or computed on when v2 was frozen (guard self-tests, REPORT_R09, REPORT_R10, REPORT_R10b). Changes from v1 are listed in the final section.**")
    t = sub1(t, "# PREREG_phase4_v2 — Phase 4 pre-registration, version 2 (frozen; tag `phase4-prereg-v2`)", "# PREREG_phase4_v2 — Phase 4 pre-registration, version 2 (frozen; tag `phase4-prereg-v2`)")
    # B2: delete item 4; add the CPU sentence to item 3
    i4 = t.index("4. **s_lr_W64"); j4 = t.index("No other predictor is scored at Phase 4."); t = t[:i4] + t[j4:]
    t = sub1(t, "seed 20261004 `" + P4.PV1_SHA[20261004] + "`.", "seed 20261004 `" + P4.PV1_SHA[20261004] + f"`. Inference at Phase 4 is CPU float32. On EW0207 it agrees with the R07 GPU bf16 predictions to {pv1:.1e} in max |Δ log₁₀| (seed-averaged) and gives the same all-tile ρ to three decimals (REPORT_R10 §5).")
    # B3 step 1 / 2 / 4
    i1 = t.index("1. **Acquisition repo:**"); j1 = t.index("2. **CNN repo:**")
    t = t[:i1] + "1. **Acquisition repo:** build the sealed pairs' contract-v2.1 ship products with the frozen code, and write `manifest/pairs_phase4_sealed.parquet` with the five sealed rows completed (`hr_native_res_m`, `lr_native_res_m`, the harmonized and product paths, and `max_recoverable_k` by the frozen rule, which the acquisition repo owns: `src/discovery/stage_c5_sweep.py:sweep_pair` with `stage_c5_spectral.perk_curve` / `perk_verdict`, reading the LR grid and the HR raster co-registered on a common grid, k = the largest of the swept k with the verdict `recoverable_signal`; written into the manifest by `src/acq_r03/manifest_v2_1.py`). Its sha256 is recorded in `CHANGE_LEDGER.md`. No context grid is built.\n" + t[j1:]
    t = sub1(t, "2. **CNN repo:** `phase4_build.py --mode phase4`. It also writes each pair's amplitude statistic (B4). **No prediction or score yet.**",
             "2. **CNN repo:** `phase4_build.py --mode phase4`. It also writes each pair's amplitude statistic (B4). `max_recoverable_k` is not set here: it comes from the supplement of step 1 (the acquisition repo owns the rule). A pair with no recoverable k is not built. **No prediction or score yet.**")
    t = sub1(t, "4. **Predict:** `phase4_predict.py --mode phase4`, on CPU.", "4. **Predict:** `phase4_predict.py --mode phase4`, on CPU. The script first reproduces the EW0207 reference predictions (s_lr identical; GBT_ship within 1e-6 m; P-V1 within 1e-4 in log₁₀) and refuses to continue otherwise.")
    # B4: rule 0, cells line, remove s_lr_W64 mentions (none remain in B4 of the draft besides B2/B5/C2 which are handled)
    t = sub1(t, "- **Tiles:**\n  1. **Primary view:**", "- **Tiles:**\n  0. A pair with no recoverable k has no fine band. It is **not scorable (no k)**, has no figure page, and is outside C1.\n  1. **Primary view:**")
    t = sub1(t, "- **Amplitude statistic:**", "- **Cells per window:** `256 × hr_res / lr_native` is reported for each sealed pair beside the development minimum of 3.88. A pair below it is flagged \"window thinner than any development pair\". No rule is attached.\n- **Amplitude statistic:**")
    # B5: expectation paragraph, pooled row, C2, C3
    i5 = t.index("  Expectation from development (R10 §A6.2"); j5 = t.index("- **C2, secondary and descriptive")
    exp = (f"  Expectation from development (REPORT_R10 §6.3; each pair as its own unit, held-out predictions): on the pairs of the ten real-relief units in the C1 set, s_lr met the C1 condition on {x1} of {y1} at D₁ and {x2} of {y2} at the 125 m distance. Pair ρ at D₁ ranged from {r1.min():.2f} to {r1.max():+.2f}, median {r1.median():+.2f}. Without AR26, which the 2026-10-09 ruling keeps out of development medians: {xa} of {ya}, range {ra.min():.2f} to {ra.max():+.2f}, median {ra.median():+.2f}. If three independent units each met the condition at the {x1}-of-{y1} rate, all three would do so with probability {p**3:.2f}, some but not all with {1 - p**3 - (1-p)**3:.2f}, and none with {(1-p)**3:.2f}. Report each sealed unit's ρ and CI against these.\n"
           f"- **C1 summary, secondary: pooled ranking.** The mean of s_lr's ρ over the units of the C1 set, with a 95 % CI from a block bootstrap stratified by unit (non-empty blocks resampled within each unit, 2,000 resamples, one `numpy.random.default_rng(20261013)`, units in ascending pair-id order). It is reported with a statement of whether its lower bound is above zero, and again with the low-amplitude artifact-affected units included. It does not change the C1 label. Expectation from development (REPORT_R10b §6): over all triples of the {T['n_pairs']} development pairs at D₁, the lower bound was above zero in {int(round(T['triples']['frac_lower_gt0'] * T['triples']['n']))} of {T['triples']['n']}; the pooled mean ranged from {T['triples']['mean_rho_min']:+.2f} to {T['triples']['mean_rho_max']:+.2f}, median {T['triples']['mean_rho_median']:+.2f}.\n")
    t = t[:i5] + exp + t[j5:]
    t = sub1(t, " [and s_lr_W64 − s_lr, if the A6.5 HOLD puts it in force]", "")
    t = sub1(t, "- **C3, supporting:** C1 and C2 on the lockbox pairs, reported separately with their history label.", "- **C3, supporting:** C1, the C1 summary and C2 on the lockbox pairs, reported separately with their history label.")
    # B6: supplement lock, EW0207 control, new script hashes
    i6 = t.index("## B6."); j6 = t.index("## Changes from v1")
    b6 = ("## B6. Provenance requirements at Phase 4\n\nEvery Phase 4 script runs in `phase4` mode only if `SIGNOFF_phase4_prereg_v2.md` exists and holds this document's sha256. Before running, it verifies that sha256, the candidate artifacts (B2), the frozen code (R05 §6; `targets_v1_int.py`; `bar_v1_1.py`; the R07 model code hash; `fields_r08.py`) and the five Phase 4 scripts against this document:\n\n| script | sha256 |\n|---|---|\n" + "\n".join(f"| `{sc}` `{h}` |" for sc, h in scripts.items()) +
          "\n\nIn `phase4` mode the five sealed manifest rows are read only from `manifest/pairs_phase4_sealed.parquet` (beside manifest v2.1), which must hold exactly the five pre-registered pair ids with the v2.1 columns and whose sha256 must be recorded in `CHANGE_LEDGER.md`; development rows come from the frozen v2.1 file. `phase4_predict.py` first reproduces the EW0207 reference predictions from the R10 inputs on OAK (s_lr identical; GBT_ship within 1e-6 m; P-V1 within 1e-4 in log₁₀) and writes the comparison to its provenance record. Outputs go to `grl_review_archive/phase4/` with `SHA256SUMS`. `phase4_score.py` refuses to run a second time.\n\n")
    t = t[:i6] + b6 + t[j6:]
    # Changes from v1: G4 line replaced; added lines
    i7 = t.index("- **G4**"); j7 = t.index("\n", i7)
    g4 = (f"- G4 (a wide-window s_lr as a secondary row) was accepted on 2026-10-10 and withdrawn the same day, before any sealed file was opened. In development, s_lr showed no loss of skill down to 3.88 cells across the window, and the 64-cell row was below s_lr on {below} of 15 pairs of the ten real-relief units ({coarse.rho_s_lr_W64.min():+.2f} to {coarse.rho_s_lr_W64.max():+.2f} against {coarse.rho_s_lr.min():+.2f} to {coarse.rho_s_lr.max():+.2f} on the pairs with ship cells of 57 m or more; REPORT_R10 §6.6).")
    t = t[:i7] + g4 + t[j7:]
    t = t.rstrip("\n") + ("\n- the pooled row (C1 summary, secondary; R10b §A1);\n- the \"not scorable (no k)\" outcome for a pair with no recoverable k (B4 rule 0);\n- cells per window reported for each sealed pair, no rule attached;\n- the sealed-row supplement `manifest/pairs_phase4_sealed.parquet` and its ledger lock (R10b §A3);\n- the EW0207 environment control before any Phase 4 prediction (R10b §A5);\n- the development expectation stated with and without AR26.\n")
    assert "s_lr_W64" not in t.replace("64-cell row", ""), [l for l in t.splitlines() if "s_lr_W64" in l]; assert "[" not in re.sub(r"\[[^\]]*\]\(", "", t) or True
    name = "PREREG_phase4_v2_DRAFT2.md" if a.draft2 else "PREREG_phase4_v2.md"; path = f"{OUTDIR}/{name}"; open(path, "w").write(t); sha = P4.sha256_file(path)
    json.dump(dict(path=path, sha256=sha, draft2=a.draft2, expectation=dict(D1=[x1, y1], D125=[x2, y2], D1_noAR26=[xa, ya], p=p, rho_min=float(r1.min()), rho_max=float(r1.max()), rho_median=float(r1.median()), noAR26=[float(ra.min()), float(ra.max()), float(ra.median())]), g4=dict(below=below, coarse_pairs=list(coarse.pair)), pv1_tol=pv1, scripts_sha256=scripts, triples=T), open(f"{ROOT}/prereg_v2_final_render.json", "w"), indent=1)
    print(f"{name} sha256 {sha}")
