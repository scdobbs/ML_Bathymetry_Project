"""R09 Part B — render PREREG_phase4_v1.md from PREREG_phase4_v1_template.md and verified records: the Part A validation
result, the lockbox-history finding, the candidate artifacts and their sha256, the development expectation range (R08b),
the frozen code hashes. Prints the sha256 of the rendered file."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import r09_common as R9
R9.start("render_prereg_r09", check=False)
S = json.load(open(f"{R9.OUT_SCORES}/scores_r09.json")); A = json.load(open(f"{R9.OUT_ART}/gbt_ship_all16_record.json")); H = json.load(open(f"{R9.OUT_SCORES}/lockbox_history.json")); S8 = json.load(open(R9.R08B_SCORES))
prov = sorted(p for p in os.listdir(R9.PROV) if p.startswith("score_r09")); pv1 = json.load(open(f"{R9.PROV}/{prov[-1]}"))["fixed_inputs_r09"]["pv1_all"]
v = S["validation"]; exp = np.array([S8["per_unit"]["D1"]["ref_s_lr"][u] for u in R9.TEN if u not in R9.NO_INTERIOR], float)
fb = (f"**validated** (|paired median Δ| = {abs(v['s_lr']['paired_median']):.3f} for s_lr and {abs(v['GBT_ship']['paired_median']):.3f} for GBT_ship on {v['s_lr']['n_units']} primary units, both ≤ {v['tol']}; REPORT_R09 §1): B4 rule 2 is in force."
      if v["validated"] else f"**not validated** (|paired median Δ| = {abs(v['s_lr']['paired_median']):.3f} for s_lr and {abs(v['GBT_ship']['paired_median']):.3f} for GBT_ship on {v['s_lr']['n_units']} primary units, tolerance {v['tol']}; REPORT_R09 §1, HOLD): B4 rule 2 is NOT in force; Phase 4 scores interior tiles only.")
r05 = A["r05_ew0207_model"]; note = (f"The inner criterion selects R05's EW0207 configuration, and the artifact reproduces R05's EW0207 predictions to {r05['max_abs_diff_vs_r05_ew0207_pred_m']:.1e} m." if r05["same_as_inner_choice"]
        else f"The inner criterion selects a different configuration from R05's EW0207 mode (cfg {r05['cfg_idx_mode']} = {r05['cfg']}); the artifact therefore differs from R05's in-memory EW0207 model (max |Δ| {r05['max_abs_diff_vs_r05_ew0207_pred_m']:.2e} m on EW0207's tiles; REPORT_R09 §1, HOLD).")
rep = {"{{FALLBACK_STATUS}}": fb, "{{LOCKBOX_HISTORY}}": H["finding"].replace("CONFIRMED: ", "") + f" Source: `{H['evidence'][0]['path']}` (sha256 `{H['evidence'][0]['sha256']}`); reports 22, 26, 55, 57 (sha256 in `scores/lockbox_history.json`).",
       "{{ECP_SHA}}": R9.R8.sha256_file(f"{R9.R8.REPO}/preprocessing/extract_clean_patches.py"), "{{R05_MODE_COUNT}}": str(r05["mode_count"]), "{{GBT_CFG_IDX}}": str(A["cfg_idx"]), "{{GBT_CFG}}": json.dumps(A["config"]), "{{GBT_SHA}}": A["sha256"], "{{GBT_R05_NOTE}}": note,
       "{{PV1_2002}}": pv1["20261002"]["ckpt.pt"], "{{PV1_2003}}": pv1["20261003"]["ckpt.pt"], "{{PV1_2004}}": pv1["20261004"]["ckpt.pt"], "{{TVI_SHA}}": R9.R8.sha256_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), "targets_v1_int.py")),
       "{{VALIDATED_SHORT}}": "validated" if v["validated"] else "NOT validated, so this rule does not apply", "{{EXP_MIN}}": f"{exp.min():.2f}", "{{EXP_MAX}}": f"{exp.max():.2f}", "{{EXP_MED}}": f"{np.median(exp):.2f}", "{{R07_CODE_SHA}}": R9.R7.code_hash()}
txt = open(f"{R9.REP}/PREREG_phase4_v1_template.md").read()
for k, val in rep.items(): assert k in txt, k; txt = txt.replace(k, val)
assert "{{" not in txt; open(f"{R9.REP}/PREREG_phase4_v1.md", "w").write(txt); sha = R9.R8.sha256_file(f"{R9.REP}/PREREG_phase4_v1.md")
json.dump(dict(sha256=sha, fields=rep), open(f"{R9.OUT_SCORES}/prereg_render.json", "w"), indent=1); print("PREREG_phase4_v1.md sha256", sha)
