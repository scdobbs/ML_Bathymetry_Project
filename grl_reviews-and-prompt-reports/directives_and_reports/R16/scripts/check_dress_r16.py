"""R16 Part C3 must-hold checks on the dress rehearsal (development data; counts, flags, paths; no value is interpreted):
  3.1 every step rc 0, no guard refusal (PermissionError) and no traceback in any log; warning categories against the known set
      (ResourceWarning, the fork DeprecationWarning, the torch.load FutureWarning, the fields_r08.py:106 All-NaN RuntimeWarning of
      the set-B page, all seen in R10 / R12c);
  3.2 / 3.3 the per-run harness records: the phase5 data locks passed on the mock files (supplement, acquisition record, calls,
      single run); every mock path refused before the allow-list; eligible mock paths allowed after it; the not-testable site,
      a FAKE id, the bare roots, an admitted site's sealed_candidates_v3 directory, the real Phase 5 names and a Phase 4 path refused;
  3.4 the not-testable row: status "not testable (...)" in the build summary, no build / fields / inputs / predictions directory,
      no page (key no_page), score row "not testable", no opened path holds its id;
  3.5 the score results: C1 primary (pooled) label, the per-site secondary, C2 with CIs for GBT_ship and P-V1 and the differences
      (per site and pooled); the artifact-call run reads "not testable" (C1 set < 3);
  3.6 the calls join inside the score reproduces the mock calls exactly (code -> pair = the key; every call field per pair equals
      the mock file's row for its code);
  3.7 the path patterns each script opens in phase5 mode against the opened-path logs (every pattern >= 1 hit);
  3.8 (EW0207 control here; the R10 reproduction is checked by repro_r10_r16.py).
Usage: python check_dress_r16.py <step logs...>. Writes R16/tables/dress_check.json."""
import glob, json, os, re, sys
import numpy as np, pandas as pd
import r16_common as R
ROOT = R.ROOT; DR = R.DR; RT = re.escape(ROOT); MH = re.escape(R.MOCK_HARM); P = {pid: re.escape(pid) for pid in R.MOCK_SITES}
def pat_build(pid):
    d = f"{MH}/{P[pid]}"; out = {f: (f"{d}/{f.replace('.', '[.]')}$", "r") for f in R.FILES}
    out["products ship_products_v2_1/*.tif"] = (f"{d}/ship_products_v2_1/(ship_count|ship_sd|ship_rsd|ship_xtrack_frac|ship_beam_angle)[.]tif$", "r")
    for f in ("tiles_v1", "tiles_v1_1", "blocks_v1_1", "features", "tiles_int"): out[f"build/{f}.parquet (w)"] = (f"{RT}/build/{P[pid]}/{f}[.]parquet$", "w")
    out["build/pair_stats.json (w)"] = (f"{RT}/build/{P[pid]}/pair_stats[.]json$", "w")
    for f in ("F_hr", "F_res", "bicubic_m", "joint_v1"): out[f"fields/{f}.tif (w)"] = (f"{RT}/fields/{P[pid]}/{f}[.]tif$", "w")
    for f in ("x.npy", "lrv.npy", "meta.parquet"): out[f"inputs/{f} (w)"] = (f"{RT}/inputs/{P[pid]}/{f.replace('.', '[.]')}$", "w")
    return out
PAT_BUILD_COMMON = {"build/eligibility_status.json (w)": (f"{RT}/build/eligibility_status[.]json$", "w"), "build/build_summary (w)": (f"{RT}/build/build_summary_\\d+[.]json$", "w")}
def pat_figures(pid):
    d = f"{MH}/{P[pid]}"
    return {"build/pair_stats.json": (f"{RT}/build/{P[pid]}/pair_stats[.]json$", "r"), "build/tiles_v1_1.parquet": (f"{RT}/build/{P[pid]}/tiles_v1_1[.]parquet$", "r"), "fields/F_hr.tif": (f"{RT}/fields/{P[pid]}/F_hr[.]tif$", "r"), "fields/joint_v1.tif": (f"{RT}/fields/{P[pid]}/joint_v1[.]tif$", "r"),
            "hr.tif (HR hillshade)": (f"{d}/hr[.]tif$", "r"), "hr_valid.tif": (f"{d}/hr_valid[.]tif$", "r"), "figures/work/pages/B_*.png (w)": (f"{RT}/figures/work/pages/B_.*{P[pid]}[.]png$", "w")}
PAT_FIG_COMMON = {"figures/work/pages/A_Q*.png (w)": (f"{RT}/figures/work/pages/A_Q\\d\\d[.]png$", "w"), "figures/key/figset_A_key.json (w)": (f"{RT}/figures/key/figset_A_key[.]json$", "w"), "figures/figset_A_blind.pdf (w)": (f"{RT}/figures/figset_A_blind[.]pdf$", "w"), "figures/work/pages_record.json (w)": (f"{RT}/figures/work/pages_record[.]json$", "w")}
def pat_predict(pid): return {"build/pair_stats.json": (f"{RT}/build/{P[pid]}/pair_stats[.]json$", "r"), "build/tiles_v1_1.parquet": (f"{RT}/build/{P[pid]}/tiles_v1_1[.]parquet$", "r"), "build/features.parquet": (f"{RT}/build/{P[pid]}/features[.]parquet$", "r"), "inputs/x.npy": (f"{RT}/inputs/{P[pid]}/x[.]npy$", "r"), "inputs/lrv.npy": (f"{RT}/inputs/{P[pid]}/lrv[.]npy$", "r"), "inputs/meta.parquet": (f"{RT}/inputs/{P[pid]}/meta[.]parquet$", "r"), "predictions/<pid>.parquet (w)": (f"{RT}/predictions/{P[pid]}[.]parquet$", "w")}
PAT_PRED_COMMON = {"R10 OAK EW0207 inputs / build / predictions (control)": (r"grl_review_archive/R10/rehearsal/(inputs|build|predictions)/EW0207__MGDS_32556", "r"), "GBT artifact joblib": (r"gbt_ship_all16_A_fine_tile[.]joblib$", "r"), "P-V1 ckpt.pt (3 seeds)": (r"P-V1/seed\d+/fold_ALL/ckpt[.]pt$", "r"), "predictions/predict_record (w)": (f"{RT}/predictions/predict_record_\\d+[.]json$", "w"), "provenance/environment_control (w)": (f"{RT}/provenance/environment_control_.*[.]json$", "w")}
PAT_SCORE = {"build/<pair>/pair_stats.json (pinned assembly)": (f"{RT}/build/[^/]+/pair_stats[.]json$", "r"), "build/<pair>/tiles_v1_1.parquet (pinned assembly)": (f"{RT}/build/[^/]+/tiles_v1_1[.]parquet$", "r"), "build/<pair>/tiles_int.parquet (pinned assembly)": (f"{RT}/build/[^/]+/tiles_int[.]parquet$", "r"),
             "predictions/<pair>.parquet (pinned assembly)": (f"{RT}/predictions/[^/]+__[^/]+[.]parquet$", "r"), "predictions/predict_record (pinned assembly)": (f"{RT}/predictions/predict_record_\\d+[.]json$", "r"), "scores/inputs_*.parquet (w)": (f"{RT}/scores/inputs_D1(_artifact)?[.]parquet$", "w"), "scores/stats_*.json (w)": (f"{RT}/scores/stats_D1(_artifact)?[.]json$", "w"), "calls/calls_mock*.csv (code-keyed)": (f"{RT}/calls/calls_mock(_artifact)?[.]csv$", "r"), "figures/key/figset_A_key.json (join)": (f"{RT}/figures/key/figset_A_key[.]json$", "r"),
             "acq record (--acq_record)": (r"PHASE5/step1_run_rehearsal_r09[.]json$", "r"), "scores/results_*.json (w)": (f"{RT}/scores/results_D1(_artifact)?[.]json$", "w"), "scores/per_pair_*.csv (w)": (f"{RT}/scores/per_pair_D1(_artifact)?[.]csv$", "w")}
def opened(script):
    p = f"{ROOT}/opened_paths_{script}.log"
    return [l.rstrip("\n").split("\t") for l in open(p) if l.strip()] if os.path.exists(p) else []
def match(pats, rows):
    out = {}
    for label, (rx, mode) in pats.items():
        hits = [r for r in rows if re.search(rx, r[3]) and (mode == "r" or ("w" in r[2] or r[1] == "DataFrame.to_parquet"))]
        out[label] = dict(n=len(hits), ok=len(hits) >= 1)
    return out
def scan_log(p):
    if not os.path.exists(p): return dict(found=False)
    L = open(p, errors="replace").read().splitlines(); cats = {}
    for l in L:
        for c in re.findall(r"\b([A-Za-z]*Warning)\b", l): cats[c] = cats.get(c, 0) + 1
    rw = [l for l in L if "RuntimeWarning" in l]; rw_other = [l for l in rw if not ("fields_r08.py:106" in l or "All-NaN slice" in l)]
    return dict(found=True, n_lines=len(L), warning_categories=cats, runtime_warnings_not_fields_r08=rw_other[:5], permission_errors=[l[:200] for l in L if "PermissionError" in l], tracebacks=sum("Traceback (most recent" in l for l in L), rc=(re.findall(r"rc=(\d+)", L[-1])[0] if L and "rc=" in L[-1] else None))
if __name__ == "__main__":
    logs = sys.argv[1:]; out = dict(root=ROOT, logs={os.path.basename(p): scan_log(p) for p in logs}, probes={}, paths={}, outputs={})
    for p in sorted(glob.glob(f"{ROOT}/mock_guard/probes_*.json")):
        j = json.load(open(p)); out["probes"][os.path.basename(p)] = {k: j[k] for k in ("script", "eligible", "not_testable", "allowlist", "probes_before_allowlist_all_refused", "probes_after_allowlist_eligible_allowed_not_testable_refused", "stay_refused_all", "guard")} | dict(locks={k: (v if k != "supplement" else {kk: v[kk] for kk in ("sha256", "eligible", "not_testable")}) for k, v in j["locks"].items()}, stay_labels=sorted(j["stay"]))
    rb, rf, rp, rs = (opened(f"phase5_{s}.py") for s in ("build", "figures", "predict", "score")); E = R.ELIGIBLE_MOCK
    out["paths"]["phase5_build.py"] = {pid: match(pat_build(pid), rb) for pid in E} | {"(common)": match(PAT_BUILD_COMMON, rb)}
    out["paths"]["phase5_figures.py"] = {pid: match(pat_figures(pid), rf) for pid in E} | {"(common)": match(PAT_FIG_COMMON, rf)}
    out["paths"]["phase5_predict.py"] = {pid: match(pat_predict(pid), rp) for pid in E} | {"(common)": match(PAT_PRED_COMMON, rp)}
    out["paths"]["phase5_score.py"] = {"(common)": match(PAT_SCORE, rs)}
    n_pat = sum(len(g) for sc in out["paths"].values() for g in sc.values()); n_ok = sum(v["ok"] for sc in out["paths"].values() for g in sc.values() for v in g.values())
    out["paths_summary"] = dict(n_patterns=n_pat, n_hit=n_ok, per_script={sc: [sum(v["ok"] for g in grp.values() for v in g.values()), sum(len(g) for g in grp.values())] for sc, grp in out["paths"].items()})
    out["provenance_written"] = {s: len(glob.glob(f"{ROOT}/provenance/phase5_{s}.py_*.json")) for s in ("build", "figures", "predict", "score")}
    allrows = rb + rf + rp + rs; out["n_opened_lines"] = dict(build=len(rb), figures=len(rf), predict=len(rp), score=len(rs))
    out["not_testable_paths_opened"] = sorted({r[3] for r in allrows if R.NOT_TESTABLE_MOCK in r[3]})
    out["real_sealed_paths_opened"] = sorted({r[3] for r in allrows if ("harmonized_confirmatory/nu0" in r[3] or any(t in r[3] for t in R.P5.SEALED8) or "/oak/stanford/groups/hilley/auv_ship_colocated_bathy/phase5" in r[3] or "/scratch/groups/hilley/auv_ship_colocated_bathy/sealed_candidates_v3" in r[3])})
    # 3.4 not-testable row
    bs = sorted(glob.glob(f"{ROOT}/build/build_summary_*.json")); summ = json.load(open(bs[-1])) if bs else {}; key = json.load(open(f"{ROOT}/figures/key/figset_A_key.json"))
    nt = R.NOT_TESTABLE_MOCK; nt_stat = [p.get("status") for p in summ.get("pairs", []) if p.get("pair") == nt]
    out["not_testable_row"] = dict(build_summary_status=nt_stat, eligibility_status=json.load(open(f"{ROOT}/build/eligibility_status.json")).get("status", {}).get(nt), dirs_exist={d: os.path.exists(f"{ROOT}/{d}/{nt}") for d in ("build", "fields", "inputs")}, prediction_exists=os.path.exists(f"{ROOT}/predictions/{nt}.parquet"),
                                   key_no_page=[x for x in key.get("no_page", []) if x["pair"] == nt], in_key_order=any(o["pair"] == nt for o in key["order"]))
    # outputs + 3.5 / 3.6
    try:
        from pypdf import PdfReader; rd = PdfReader(f"{ROOT}/figures/figset_A_blind.pdf"); out["outputs"]["figures"] = dict(n_pages=len(rd.pages), text_chars_per_page=[len((pg.extract_text() or "").strip()) for pg in rd.pages], codes=[o["code"] for o in key["order"]], no_page=key.get("no_page"))
    except Exception as e: out["outputs"]["figures"] = dict(error=str(e)[:200])
    for pid in E:
        o = {}
        try: st = json.load(open(f"{ROOT}/build/{pid}/pair_stats.json")); o.update(status=st["status"], product_features=st.get("product_features"), n_tiles=st.get("n_tiles_v1_1"), D1=st.get("D1_px"), cells=st.get("cells_across"), view_B8=(st.get("views") or {}).get("view_B8"), k_source=(st.get("k_provenance") or {}).get("k_source_class"))
        except Exception as e: o["pair_stats"] = str(e)[:120]
        try: pr = pd.read_parquet(f"{ROOT}/predictions/{pid}.parquet"); o["predictions"] = dict(n=len(pr), finite={c: int(np.isfinite(pr[c].values.astype(float)).sum()) for c in ("s_lr", "GBT_ship", "P-V1")})
        except Exception as e: o["predictions"] = str(e)[:120]
        out["outputs"][pid] = o
    for tag, cf in (("D1", "calls_mock.csv"), ("D1_artifact", "calls_mock_artifact.csv")):
        try:
            res = json.load(open(f"{ROOT}/scores/results_{tag}.json")); calls = pd.read_csv(f"{ROOT}/calls/{cf}").fillna(""); c2p = res["calls_join"]["code_to_pair"]
            key_map = {o["code"]: o["pair"] for o in key["order"]}; pp = {r["pair"]: r for r in res["per_pair"]}
            join_ok = (c2p == key_map) and all(pp[c2p[r.code]]["call_relief"] == r.relief and pp[c2p[r.code]]["call_criteria"] == str(r.criteria) and pp[c2p[r.code]]["call_dominant"] == r.dominant and pp[c2p[r.code]]["call_ab_whole_survey"] == str(r.ab_whole_survey).lower() for r in calls.itertuples())
            c2s = res["C2"]["per_site"]; c2p_ = res["C2"]["pooled_C1_set"]
            has_ci = lambda d: all(isinstance(d.get(q, {}).get("ci95"), list) and len(d[q]["ci95"]) == 2 for q in ("GBT_ship", "P-V1", "d_GBT_minus_s_lr", "d_PV1_minus_s_lr"))
            out["outputs"][f"score_{tag}"] = dict(per_pair_status={r["pair"]: dict(view=r.get("view"), status=r.get("status"), n_scored=r.get("n_scored"), ci_estimable=r.get("ci_estimable")) for r in res["per_pair"]},
                C1_primary=res["C1_primary"]["reading"], C1_primary_n=res["C1_primary"]["n_sites"], C1_secondary=res["C1_per_site_secondary"]["reading"], C1_secondary_set=res["C1_per_site_secondary"]["c1_set"],
                C2_per_site_with_CIs={p: (has_ci(d) if d else None) for p, d in c2s.items()}, C2_pooled_with_CIs=(has_ci(c2p_) if c2p_ else None), C2_estimable={p: {q: d[q]["estimable"] for q in ("GBT_ship", "P-V1", "d_GBT_minus_s_lr", "d_PV1_minus_s_lr")} for p, d in c2s.items() if d},
                calls_join_reproduces_mock_calls=bool(join_ok), not_testable=res.get("not_testable"))
        except Exception as e: out["outputs"][f"score_{tag}"] = dict(error=f"{type(e).__name__}: {e}"[:300])
    ctrl = sorted(glob.glob(f"{ROOT}/provenance/environment_control_*.json")); out["environment_control"] = [dict(job=json.load(open(p)).get("slurm_job"), **{k: json.load(open(p))["environment_control"][k] for k in ("node", "cpu", "s_lr_identical", "gbt_max_abs_diff_m", "pv1_seedavg_max_abs_diff_log10", "passed")}) for p in ctrl]
    known = {"ResourceWarning", "DeprecationWarning", "FutureWarning", "RuntimeWarning"}; out["warnings_unknown"] = {k: [c for c in v.get("warning_categories", {}) if c not in known] + (["RuntimeWarning (not fields_r08:106)"] if v.get("runtime_warnings_not_fields_r08") else []) for k, v in out["logs"].items()}
    s1 = out["outputs"].get("score_D1", {}); s2 = out["outputs"].get("score_D1_artifact", {}); ntr = out["not_testable_row"]
    mh = {"3.1 completed, no refusal / traceback / new warning": bool(logs and all(v["found"] and v["rc"] == "0" and not v["permission_errors"] and v["tracebacks"] == 0 for v in out["logs"].values()) and not any(out["warnings_unknown"].values())),
          "3.2 data locks on mock files in every run": bool(out["probes"] and all({"supplement", "acq_run_record"} <= set(v["locks"]) for v in out["probes"].values())),
          "3.3 allow-list (before refused / after admitted / others refused)": bool(out["probes"] and all(v["probes_before_allowlist_all_refused"] and v["probes_after_allowlist_eligible_allowed_not_testable_refused"] and v["stay_refused_all"] for v in out["probes"].values())),
          "3.4 eligible = False row not testable, nothing built": bool(ntr["build_summary_status"] and ntr["build_summary_status"][0].startswith("not testable") and not any(ntr["dirs_exist"].values()) and not ntr["prediction_exists"] and ntr["key_no_page"] and not ntr["in_key_order"] and not out["not_testable_paths_opened"] and str((s1.get("not_testable") or {}).get(nt, "")).startswith("not testable")),
          "3.5 pooled primary + per-site secondary + C2 CIs; < 3 sites -> not testable": bool(s1.get("C1_primary") in ("confirmed", "not confirmed") and s1.get("C1_secondary") and s1.get("C2_pooled_with_CIs") and all(v for v in (s1.get("C2_per_site_with_CIs") or {}).values() if v is not None) and str(s2.get("C1_primary", "")).startswith("not testable")),
          "3.6 calls join reproduces the mock calls": bool(s1.get("calls_join_reproduces_mock_calls") and s2.get("calls_join_reproduces_mock_calls")),
          "3.7 every path pattern exercised": out["paths_summary"]["n_hit"] == out["paths_summary"]["n_patterns"] and all(v >= 1 for v in out["provenance_written"].values()),
          "3.8 EW0207 control passed": bool(out["environment_control"] and all(c["passed"] for c in out["environment_control"])),
          "no real sealed path opened": len(out["real_sealed_paths_opened"]) == 0}
    out["must_hold"] = mh | dict(all=all(mh.values())); os.makedirs(f"{DR}/R16/tables", exist_ok=True); json.dump(out, open(f"{DR}/R16/tables/dress_check{'_r16b' if R.WHICH == 'mock_b' else ''}.json", "w"), indent=1, default=str); print(json.dumps(out["must_hold"], indent=1)); print(json.dumps(out["paths_summary"], indent=1))
