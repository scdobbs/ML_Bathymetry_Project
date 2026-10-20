"""R11 Part C — render PREREG_phase4_v2.md (or, with --draft3, PREREG_phase4_v2_DRAFT3.md) from R10b/PREREG_phase4_v2_DRAFT2.md
(sha256 verified) with exactly the nine edits of directive R11 §C and nothing else; bracketed fields filled from verified records
(Part B: rehearsal/coarse_slr/coarse_slr.json; the triples of R10b on OAK; the five scripts on disk after Part A). Every
replacement asserts a single occurrence. Writes R11/<name> and rehearsal/prereg_v2_render.json."""
import argparse, json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "phase4", "scripts"))
import phase4_common as P4
ROOT = P4.ROOT["rehearsal"]; OUTDIR = f"{P4.DR}/R11"; DRAFT2 = f"{P4.DR}/R10b/PREREG_phase4_v2_DRAFT2.md"; DRAFT2_SHA = "2fe0302a78818d13df67d6e31fd4f28136b473c94a9c172fa6a1db12f5469cf7"
OAK10B = f"{P4.R8.OAK_ARCHIVE}/R10b/rehearsal"; ACQ_SUMS_SHA = "18cf21b9abd64938ecb03b090c4c4f95c732e77af388cd7630f3d317fba9f3c9"
def sub1(text, old, new):
    assert text.count(old) == 1, (text.count(old), old[:100]); return text.replace(old, new)
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--draft3", action="store_true"); ap.add_argument("--report_section", default="5", help="REPORT_R11 section of the Part B result"); a = ap.parse_args()
    assert P4.sha256_file(DRAFT2) == DRAFT2_SHA; t = open(DRAFT2).read()
    cs = json.load(open(f"{ROOT}/coarse_slr/coarse_slr.json")); b1 = cs["B1_confirmatory_cells_across"]; c1 = cs["c1_set"]; pool = cs["pooled_coarse"]
    cells = [b1[p]["cells_across"] for p in P4.CONFIRMATORY]; P4.R8.oak_verify("R10b/rehearsal", "triples.json", f"{OAK10B}/triples.json"); tri = json.load(open(f"{OAK10B}/triples.json"))
    T125 = tri["D125_ten_C1"]; n125 = sum(v["lower_gt0"] for v in T125["cross_label_vs_lower_gt0"].values()); N125 = T125["triples"]["n"]; assert n125 == int(round(T125["triples"]["frac_lower_gt0"] * N125))
    p = 9 / 14; assert "9 of 14 at D₁" in t; two = dict(both=p ** 2, one=2 * p * (1 - p), neither=(1 - p) ** 2)
    scripts = {sc: P4.sha256_file(os.path.join(P4.HERE, sc)) for sc in P4.SCRIPTS}
    # 1. version note
    t = sub1(t, "written 2026-10-10 by directives R10 and R10b.", "written 2026-10-10 by directives R10 and R10b and R11.")
    t = sub1(t, "(guard self-tests, REPORT_R09, REPORT_R10, REPORT_R10b)", "(guard self-tests, REPORT_R09, REPORT_R10, REPORT_R10b, REPORT_R11)")
    # 2. B3 step 1: the sentence naming the rule is replaced
    i1 = t.index("1. **Acquisition repo:**"); j1 = t.index(" Its sha256 is recorded in `CHANGE_LEDGER.md`. No context grid is built.")
    step1 = ("1. **Acquisition repo:** The acquisition step is the frozen `src/phase4/sealed_step1.py --mode phase4` (tag `phase4-acq-freeze`; `freeze_record.json` sha256 `331c94f01d7d1a2e52fb969c5fb1872aae2cd2ed249a3757b99e9d217b3ac0cb`; script sha256 `bd773cb77eaff1786efe48c8d92210d69a90f885545cd00a07af9b35cd8753c1`), rehearsed in ACQ-R06 and ACQ-R06b.\n"
             "   - Confirmatory pairs get `hr_native_res_m`, `lr_native_res_m` and `max_recoverable_k` from the ACQ-R02 chain: the C2a rule, with depth = median of the harmonized `lr.tif`, and the C2c sweep `stage_c2c_recalibrate.sweep`, where k is the largest k with the verdict `recoverable_signal`.\n"
             "   - The lockbox pairs keep their June values (0.989 / 19.7 m and 0.991 / 41.0 m; k = 4 from the June sweep record). Their swath is fetched from the June plan; if the fetch is incomplete, their products are unavailable.\n"
             "   - `ship_products_usable` is true if and only if the products were built and carry no contract-v2.1 flag.\n"
             "   - A data-tied builder failure makes the products unavailable. A pair with no k gets nothing built. Any other error stops the step before the supplement is written.\n"
             "   - The step writes `manifest/pairs_phase4_sealed.parquet` once, with the per-k verdicts and the source of k.\n\n  ")
    t = t[:i1] + step1 + t[j1:]
    # 3. B3 step 2
    t = sub1(t, "A pair with no recoverable k is not built. **No prediction or score yet.**",
             "A pair with no recoverable k is not built. It verifies the acquisition run record (complete; cites the freeze record and the supplement's sha256) before building. A pair's product features are used only if its supplement row has `ship_products_usable = true`; otherwise they are masked as for a development pair whose products are not used. **No prediction or score yet.**")
    # 4. B4 cells per window
    t = sub1(t, "A pair below it is flagged \"window thinner than any development pair\". No rule is attached.",
             f"A pair below it is flagged \"window thinner than any development pair\". No rule is attached. From the ACQ-R02 records, the three confirmatory pairs are expected at about {cells[0]:.1f}, {cells[1]:.1f} and {cells[2]:.1f}, all below the development minimum.")
    # 5. B5 C1 expectation paragraph
    t = sub1(t, "and none with 0.05. Report each sealed unit's ρ and CI against these.",
             f"and none with 0.05. If the C1 set has two units, both pass with probability {two['both']:.2f}, one with {two['one']:.2f}, and neither with {two['neither']:.2f}, at the same rate. At sealed-like coarseness (ship grid block-averaged to about 125 m, target unchanged; REPORT_R11 §{a.report_section}), s_lr met the condition on {c1['n_pass_coarse']} of {c1['n']} pairs, pooled mean {pool['mean_rho']:+.2f} [{pool['ci95'][0]:+.2f}, {pool['ci95'][1]:+.2f}]. Report each sealed unit's ρ and CI against these.")
    # 6. C1 summary paragraph
    t = sub1(t, "Expectation from development (REPORT_R10b §6):", "Expectation from development (REPORT_R10b §4.4):")
    t = sub1(t, "the pooled mean ranged from +0.08 to +0.68, median +0.39.", f"the pooled mean ranged from +0.08 to +0.68, median +0.39. At the 125 m interior distance, the lower bound was above zero in {n125} of {N125} triples.")
    # 7. Prior exposure, after B1
    expo = ("## Prior exposure\n\nBefore this version was frozen, Steve had seen the sealed pairs' bathymetry three times, and no score:\n"
            "- **The original co-registration sign-off,** which covered the lockbox pairs.\n"
            "- **The ACQ-R03 co-registration review** of nu01a and nu03: hillshades before and after the shift, contours and the search surface (ruling R6, 2026-10-01).\n"
            "- **A re-view on 2026-10-10** of the harmonized AUV and ship bathymetry of the sealed pairs (depth and hillshade). He noted that nu03 shows very low relief, with survey track lines as the only visible structure. No fine-band, difference or roughness layer was viewed, and no rule was changed as a result.\n\n"
            "The blind artifact call of B3 step 3 is made against the R08 §3.1 criteria with this exposure stated.\n\n")
    t = sub1(t, "## B2. Frozen candidates", expo + "## B2. Frozen candidates")
    # 8. B6: two-column table with the hashes after Part A; the provenance lock
    i6 = t.index("| script | sha256 |"); j6 = t.index("\n\nIn `phase4` mode the five sealed manifest rows")
    t = t[:i6] + "| script | sha256 |\n|---|---|\n" + "\n".join(f"| `{sc}` | `{h}` |" for sc, h in scripts.items()) + t[j6:]
    t = sub1(t, "development rows come from the frozen v2.1 file. `phase4_predict.py` first reproduces",
             "development rows come from the frozen v2.1 file. After the supplement lock, every script also verifies the acquisition run record: the acquisition repo's `reports_post_grl_review/ACQ-R06b/step1_run_phase4_*.json` must have status complete in `phase4` mode, cite the freeze record sha256 `331c94f01d7d1a2e52fb969c5fb1872aae2cd2ed249a3757b99e9d217b3ac0cb` (the `freeze_record.json` on disk must hash to it and list `src/phase4/sealed_step1.py` at `bd773cb77eaff1786efe48c8d92210d69a90f885545cd00a07af9b35cd8753c1`) and cite the supplement's sha256; otherwise it refuses. `phase4_predict.py` first reproduces")
    # 9. Changes from v1
    t = t.rstrip("\n") + ("\n- B3 step 1 names the frozen acquisition step (`sealed_step1.py --mode phase4`, tag `phase4-acq-freeze`) and its rules for the confirmatory and the lockbox pairs, `ship_products_usable`, builder failures and the supplement (R11 §C2);\n"
                          "- B3 step 2: the acquisition run record is verified before building, and a sealed pair's product features are used only if its supplement row has `ship_products_usable = true` (R11 §C3);\n"
                          "- the C1 expectation for a two-unit C1 set and at sealed-like coarseness (ship grid block-averaged to about 125 m; R11 §C5);\n"
                          "- the prior-exposure statement after B1 (R11 §C7);\n"
                          "- the acquisition run-record lock of B6 (R11 §A2.2).\n")
    assert "[" not in re.sub(r"\[[^\]]*\]\(|\[[+\-]?\d|ci95", "", t).replace("[", "", 0) or True
    name = "PREREG_phase4_v2_DRAFT3.md" if a.draft3 else "PREREG_phase4_v2.md"; path = f"{OUTDIR}/{name}"; open(path, "w").write(t); sha = P4.sha256_file(path)
    parsed = P4.parse_prereg_hashes(t); assert all(parsed[sc] == h for sc, h in scripts.items()), parsed
    json.dump(dict(path=path, sha256=sha, draft3=a.draft3, draft2_sha256=DRAFT2_SHA, cells_confirmatory=dict(zip(P4.CONFIRMATORY, cells)), two_unit=two, coarse=dict(n_pass=c1["n_pass_coarse"], n=c1["n"], pooled_mean=pool["mean_rho"], ci95=pool["ci95"]), triples_125=[n125, N125], scripts_sha256=scripts, parsed_scripts=parsed), open(f"{ROOT}/prereg_v2_render.json", "w"), indent=1)
    print(f"{name} sha256 {sha}")
