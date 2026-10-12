"""R12c Part C must-hold checks on the dress rehearsal (development data; counts, flags and paths only):
  1. the four scripts completed (job log rc per step), no guard refusal during a run (no PermissionError in any log), warnings inventory by
     category against the known set (ResourceWarning; the Python 3.12 fork DeprecationWarning and the torch.load FutureWarning seen in R12);
  2. the mock-guard probe records: before the allow-list every mock path refused; after it every mock path allowed, the FAKE probe under
     harmonized_confirmatory/nu99/ refused, a real sealed path refused;
  3. the Part A.4 path patterns against the opened-path logs of each script (every pattern must have >= 1 hit; the masked pair must open
     no product raster);
  6. the EW0207 environment control record.
Also: pages / codes of the mock figure set, predictions finite per pair, the score run's per-pair statuses (no value).
Usage: python check_dress_r12c.py <step logs...>. Writes R12c/tables/dress_check.json."""
import glob, json, os, re, sys
import numpy as np, pandas as pd
import r12c_common as R
ROOT = R.ROOT; DR = R.DR; P = {pid: re.escape(pid) for pid in R.MOCK_PAIRS}; MOCK = re.escape(R.MOCK); RT = re.escape(ROOT)
def pat_build(pid, usable):
    d = f"{MOCK}/harmonized(_confirmatory/nu9\\d)?/{P[pid]}"; out = {f"{f}": (f"{d}/{f.replace('.', '[.]')}$", "r") for f in R.FILES}
    out["products ship_products_v2_1/*.tif"] = (f"{d}/ship_products_v2_1/(ship_count|ship_sd|ship_rsd|ship_xtrack_frac|ship_beam_angle)[.]tif$", "r" if usable else "ABSENT")
    for f in ("tiles_v1", "tiles_v1_1", "blocks_v1_1", "features", "tiles_int"): out[f"build/{f}.parquet (w)"] = (f"{RT}/build/{P[pid]}/{f}[.]parquet$", "w")
    out["build/pair_stats.json (w)"] = (f"{RT}/build/{P[pid]}/pair_stats[.]json$", "w")
    for f in ("F_hr", "F_res", "bicubic_m", "joint_v1"): out[f"fields/{f}.tif (w)"] = (f"{RT}/fields/{P[pid]}/{f}[.]tif$", "w")
    for f in ("x.npy", "lrv.npy", "meta.parquet"): out[f"inputs/{f} (w)"] = (f"{RT}/inputs/{P[pid]}/{f.replace('.', '[.]')}$", "w")
    return out
def pat_figures(pid):
    d = f"{MOCK}/harmonized(_confirmatory/nu9\\d)?/{P[pid]}"
    return {"build/pair_stats.json": (f"{RT}/build/{P[pid]}/pair_stats[.]json$", "r"), "build/tiles_v1_1.parquet": (f"{RT}/build/{P[pid]}/tiles_v1_1[.]parquet$", "r"), "fields/F_hr.tif": (f"{RT}/fields/{P[pid]}/F_hr[.]tif$", "r"), "fields/joint_v1.tif": (f"{RT}/fields/{P[pid]}/joint_v1[.]tif$", "r"),
            "hr.tif (HR hillshade)": (f"{d}/hr[.]tif$", "r"), "hr_valid.tif": (f"{d}/hr_valid[.]tif$", "r"), "figures/work/pages/B_*.png (w)": (f"{RT}/figures/work/pages/B_.*{P[pid]}[.]png$", "w")}
PAT_FIG_COMMON = {"figures/work/pages/A_Q*.png (w)": (f"{RT}/figures/work/pages/A_Q\\d\\d[.]png$", "w"), "figures/key/figset_A_key.json (w)": (f"{RT}/figures/key/figset_A_key[.]json$", "w"), "figures/figset_A_blind.pdf (w)": (f"{RT}/figures/figset_A_blind[.]pdf$", "w"), "figures/work/pages_record.json (w)": (f"{RT}/figures/work/pages_record[.]json$", "w")}
def pat_predict(pid): return {"build/tiles_v1_1.parquet": (f"{RT}/build/{P[pid]}/tiles_v1_1[.]parquet$", "r"), "build/features.parquet": (f"{RT}/build/{P[pid]}/features[.]parquet$", "r"), "inputs/x.npy": (f"{RT}/inputs/{P[pid]}/x[.]npy$", "r"), "inputs/lrv.npy": (f"{RT}/inputs/{P[pid]}/lrv[.]npy$", "r"), "inputs/meta.parquet": (f"{RT}/inputs/{P[pid]}/meta[.]parquet$", "r"), "predictions/<pid>.parquet (w)": (f"{RT}/predictions/{P[pid]}[.]parquet$", "w")}
PAT_PRED_COMMON = {"R10 OAK EW0207 inputs / build / predictions (control)": (r"grl_review_archive/R10/rehearsal/(inputs|build|predictions)/EW0207__MGDS_32556", "r"), "GBT artifact joblib": (r"gbt_ship_all16_A_fine_tile[.]joblib$", "r"), "P-V1 ckpt.pt (3 seeds)": (r"P-V1/seed\d+/fold_ALL/ckpt[.]pt$", "r"), "predictions/predict_record (w)": (f"{RT}/predictions/predict_record_\\d+[.]json$", "w")}
PAT_SCORE = {"score_inputs/inputs_D1.parquet": (f"{RT}/score_inputs/inputs_D1[.]parquet$", "r"), "score_inputs/calls_mock.csv": (f"{RT}/score_inputs/calls_mock[.]csv$", "r"), "score_inputs/stats.json": (f"{RT}/score_inputs/stats[.]json$", "r"), "mock supplement (--rows)": (f"{MOCK}/mock_supplement[.]parquet$", "r"), "scores/results_D1.json (w)": (f"{RT}/scores/results_D1[.]json$", "w"), "scores/per_pair_D1.csv (w)": (f"{RT}/scores/per_pair_D1[.]csv$", "w")}
def opened(script):
    p = f"{ROOT}/opened_paths_{script}.log"
    if not os.path.exists(p): return []
    return [l.rstrip("\n").split("\t") for l in open(p) if l.strip()]
def match(pats, rows):
    out = {}
    for label, (rx, mode) in pats.items():
        hits = [r for r in rows if re.search(rx, r[3]) and (mode == "r" or mode == "ABSENT" or ("w" in r[2] or r[1] == "DataFrame.to_parquet"))]
        out[label] = dict(n=len(hits), expected=("0 (masked pair)" if mode == "ABSENT" else ">= 1"), ok=(len(hits) == 0 if mode == "ABSENT" else len(hits) >= 1))
    return out
def scan_log(p):
    if not os.path.exists(p): return dict(found=False)
    L = open(p, errors="replace").read().splitlines(); cats = {}
    for l in L:
        for c in re.findall(r"\b([A-Za-z]*Warning)\b", l): cats[c] = cats.get(c, 0) + 1
    return dict(found=True, n_lines=len(L), warning_categories=cats, permission_errors=[l[:200] for l in L if "PermissionError" in l], tracebacks=sum("Traceback (most recent" in l for l in L), rc=(re.findall(r"rc=(\d+)", L[-1])[0] if L and "rc=" in L[-1] else None))
if __name__ == "__main__":
    logs = sys.argv[1:]; out = dict(root=ROOT, logs={os.path.basename(p): scan_log(p) for p in logs}, probes={}, paths={}, outputs={})
    for p in sorted(glob.glob(f"{ROOT}/mock_guard/probes_*.json")):
        j = json.load(open(p)); out["probes"][os.path.basename(p)] = {k: j[k] for k in ("script", "mock_pairs", "allowlist", "tokens_remaining", "probes_before_allowlist_all_refused", "probes_after_allowlist_all_allowed", "guard")} | dict(fake_refused=j["fake_probe"]["refused"], real_sealed_refused=j["real_sealed_probe"]["refused"], n_tokens_added=len(j["mock_tokens_added"]))
    rb = opened("phase4_build.py"); rf = opened("phase4_figures.py"); rp = opened("phase4_predict.py"); rs = opened("phase4_score.py")
    out["paths"]["phase4_build.py"] = {pid: match(pat_build(pid, R.MOCK_ROWS[pid]["usable"]), rb) for pid in R.MOCK_PAIRS}; out["paths"]["phase4_build.py"]["(common)"] = match({"provenance (w)": (f"{RT}/provenance/phase4_build[.]py_.*[.]json$", "w"), "build_summary (w)": (f"{RT}/build/build_summary_\\d+[.]json$", "w")}, rb)
    out["paths"]["phase4_figures.py"] = {pid: match(pat_figures(pid), rf) for pid in R.MOCK_PAIRS}; out["paths"]["phase4_figures.py"]["(common)"] = match(PAT_FIG_COMMON, rf)
    out["paths"]["phase4_predict.py"] = {pid: match(pat_predict(pid), rp) for pid in R.MOCK_PAIRS}; out["paths"]["phase4_predict.py"]["(common)"] = match(PAT_PRED_COMMON, rp)
    out["paths"]["phase4_score.py"] = {"(common)": match(PAT_SCORE, rs)}
    out["paths_all_ok"] = bool(all(v["ok"] for sc in out["paths"].values() for grp in sc.values() for v in grp.values())); out["n_opened_lines"] = dict(build=len(rb), figures=len(rf), predict=len(rp), score=len(rs))
    out["sealed_real_paths_opened"] = sorted({r[3] for r in rb + rf + rp + rs if "harmonized_confirmatory/nu0" in r[3] or any(t in r[3] for t in ("AT42-03__MGDS_32007", "TN159__MGDS_21981", "KN182L03", "RC2511", "SUM1004"))})
    # outputs
    try:
        from pypdf import PdfReader; pdf = f"{ROOT}/figures/figset_A_blind.pdf"; rd = PdfReader(pdf); key = json.load(open(f"{ROOT}/figures/key/figset_A_key.json"))
        out["outputs"]["figures"] = dict(n_pages=len(rd.pages), text_objects_per_page=[len((pg.extract_text() or "").strip()) for pg in rd.pages], codes=[o["code"] for o in key["order"]], no_page=key.get("no_page"))
    except Exception as e: out["outputs"]["figures"] = dict(error=str(e)[:200])
    for pid in R.MOCK_PAIRS:
        o = {}
        try: st = json.load(open(f"{ROOT}/build/{pid}/pair_stats.json")); o.update(status=st["status"], product_features=st.get("product_features"), n_tiles=st.get("n_tiles_v1_1"), D1=st.get("D1_px"), view_B8=(st.get("views") or {}).get("view_B8"), interior=(st.get("views") or {}).get("interior", {}).get("n_tiles"), int_blocks=(st.get("views") or {}).get("interior", {}).get("n_blocks"))
        except Exception as e: o["pair_stats"] = str(e)[:120]
        try: pr = pd.read_parquet(f"{ROOT}/predictions/{pid}.parquet"); o["predictions"] = dict(n=len(pr), finite={c: int(np.isfinite(pr[c].values.astype(float)).sum()) for c in ("s_lr", "GBT_ship", "P-V1")})
        except Exception as e: o["predictions"] = str(e)[:120]
        out["outputs"][pid] = o
    try: res = json.load(open(f"{ROOT}/scores/results_D1.json")); out["outputs"]["score"] = dict(per_pair_status={r["pair"]: dict(view=r.get("view"), status=r.get("status"), n_scored=r.get("n_scored"), ci_estimable=r.get("ci_estimable")) for r in res["per_pair"]}, C1={k: v["reading"] for k, v in res["C1"].items()})
    except Exception as e: out["outputs"]["score"] = str(e)[:200]
    ctrl = sorted(glob.glob(f"{ROOT}/provenance/environment_control_*.json")); out["environment_control"] = [dict(job=json.load(open(p)).get("slurm_job"), **{k: json.load(open(p))["environment_control"][k] for k in ("node", "cpu", "s_lr_identical", "gbt_max_abs_diff_m", "pv1_seedavg_max_abs_diff_log10", "passed")}) for p in ctrl]
    known = {"ResourceWarning", "DeprecationWarning", "FutureWarning"}; out["warnings_unknown"] = {k: [c for c in v.get("warning_categories", {}) if c not in known] for k, v in out["logs"].items()}
    mh = dict(m1_completed_no_refusal=bool(logs and all(v["found"] and v["rc"] == "0" and not v["permission_errors"] and v["tracebacks"] == 0 for v in out["logs"].values()) and not any(out["warnings_unknown"].values())),
              m2_fake_probe_refused=bool(out["probes"] and all(v["fake_refused"] and v["real_sealed_refused"] and v["probes_before_allowlist_all_refused"] and v["probes_after_allowlist_all_allowed"] for v in out["probes"].values())), m3_paths_all_exercised=out["paths_all_ok"], m6_control=bool(out["environment_control"] and all(c["passed"] for c in out["environment_control"])), no_real_sealed_path_opened=(len(out["sealed_real_paths_opened"]) == 0))
    out["must_hold"] = mh | dict(all=all(mh.values())); os.makedirs(f"{DR}/R12c/tables", exist_ok=True); json.dump(out, open(f"{DR}/R12c/tables/dress_check.json", "w"), indent=1, default=str); print(json.dumps(out["must_hold"], indent=1)); print(json.dumps(out["outputs"], indent=1, default=str)[:3000])
