"""R09 §B1 — lockbox history: evidence that the two lockbox pairs were scored (and earlier used as splits) by the submitted
pipeline. Reads only report / record files whose paths carry no sealed id (the guard is installed); records each file's
sha256 and the relevant lines. Writes scores/lockbox_history.json."""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import r09_common as R9
REPO = R9.R8.REPO; LB = R9.C4.LOCKBOX
EV = [("dir55 external test results (the manuscript S5 source)", "/oak/stanford/groups/hilley/auv_ship_colocated_bathy/analysis_archive_2026-09-28/dir55/external_test/results.json"),
      ("REPORT_55 (2026-07-08) §A/§C/§E", f"{REPO}/prompts_reports/REPORT_55_2026-07-08.md"), ("REPORT_57 (2026-07-09) table provenance", f"{REPO}/prompts_reports/REPORT_57_2026-07-09.md"),
      ("manuscript tables (transcribed, directive 57)", "/scratch/users/scdobbs/manuscript_package/tables/MANUSCRIPT_TABLES.md"),
      ("report 22 Phase 1 results (2026-06-30) split table", f"{REPO}/prompts_reports/22_phase1_results_2026-06-30.md"), ("report 26 native results (2026-07-01)", f"{REPO}/prompts_reports/26_native_results_2026-07-01.md")]
if __name__ == "__main__":
    R9.start("lockbox_history_r09", check=False); out = dict(lockbox_pairs=list(LB), evidence=[])
    for label, path in EV:
        txt = open(path).read(); lines = [(i + 1, l.strip()[:300]) for i, l in enumerate(txt.splitlines()) if any(t in l for t in ("AT42-03", "TN159", "never-trained", "never trained", "het_production_v1_repro", "model_label", "frozen_model"))]
        out["evidence"].append(dict(label=label, path=path, sha256=R9.R8.sha256_file(path), lines=lines[:40]))
    d = json.load(open(EV[0][1])); out["dir55_results"] = dict(model_label=d["model_label"], frozen_model=d["frozen_model"], n_new_units=d["n_new_units"], note=d["note"], lockbox_rows={u: d["per_unit"][u] for u in ("lu03", "lu17")})
    out["finding"] = ("CONFIRMED: the submitted manuscript's '10 never-trained units' table (S5_external_test; REPORT_57 maps it to dir55/external_test/results.json) scores AT42-03__MGDS_32007 (lu03, 237 patches, within-pair R2 0.654) and TN159__MGDS_21981 (lu17, 113 patches, 0.202) with the frozen all-native model labelled 'het_production_v1_repro (all-native)' in the results file, i.e. the §C reproduction of het_production_v1 that REPORT_55 §C labels het_retrain_nonrepro (the production weights were never persisted; REPORT_55 §A). Earlier, the first het training (report 22, 2026-06-30) used TN159 as a test pair and AT42-03 as a validation pair, and report 26 reports calibration on both; REPORT_55 then declared them the 'never-trained reserve' and asserted them absent from every train / held set of the expanded retrain.")
    R9.R8.dump(out, f"{R9.OUT_SCORES}/lockbox_history.json"); print(out["finding"]); print({e["label"]: e["sha256"][:12] for e in out["evidence"]})
