"""R12c Part E2 (R12b Part B) must-hold checks on the Phase 4 build outputs (grl_review_archive/phase4), reading ONLY the fields the report may carry
(§5.1: status, tile / interior / fallback / block counts, view at B_min = 8, D1, cells across, product-feature decision). Every other
field of pair_stats.json (the amplitude statistics, r04 / r06 summaries) is dropped before anything is printed or written. For the
finite-target check the tiles table is reduced to counts (n tiles with joint-valid px; n with finite A_fine). No value is printed.
Also records the provenance record of the build (lock outcomes; the accepted acquisition record) and the job log's redacted lines.
Writes R12b/tables/build_check.json (copied to scratch) and prints a one-line verdict per pair."""
import glob, json, os, re, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); DR = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, os.path.join(DR, "phase4", "scripts"))
import phase4_common as P4
ROOT = P4.ROOT["phase4"]; PAIRS = list(P4.SEALED5); ALLOWED = ("pair", "unit", "status", "hr_res", "lr_native", "k", "D1_px", "cells_across", "dev_min_cells", "thinner_than_development", "n_tiles_v1", "n_tiles_v1_1", "n_blocks_kept", "n_feature_rows", "ship_products_usable", "product_features", "contract_columns_frac_nan", "seconds")
EXPECT = {"KN182L03__MGDS_30193": (P4.NO_K, None), "RC2511__MGDS_32239": ("built", "masked"), "SUM1004__MGDS_33090": ("built", "used"), "AT42-03__MGDS_32007": ("built", "used"), "TN159__MGDS_21981": ("built", "used")}
def redact(s): return re.sub(r"amplitude [0-9.eE+\-]+ m", "amplitude [redacted]", s)
if __name__ == "__main__":
    # The build outputs carry the pair id in their path, so the R04 / R09 guards (installed on import) refuse them until the sanctioned
    # route has run: guard self-test, every phase4 lock, then the allow-list (phase4_common.start; it writes a provenance record).
    _m, _root, _rec = P4.start("check_build_r12c.py", "phase4", PAIRS); assert _root == ROOT
    log = sys.argv[1] if len(sys.argv) > 1 else None; out = dict(root=ROOT, pairs={}, must_hold={}, start_record=dict(guard=[_rec.get("guard_probes_blocked"), _rec.get("guard_probes")], acq_run_record=_rec["acq_run_record"]["sha256"] if _rec.get("acq_run_record") else None, allowlist_n=len(_rec.get("allowlist") or [])))
    for pid in PAIRS:
        p = f"{ROOT}/build/{pid}/pair_stats.json"
        if not os.path.exists(p): out["pairs"][pid] = dict(status="MISSING pair_stats.json"); continue
        st = json.load(open(p)); r = {k: st.get(k) for k in ALLOWED if k in st}; v = st.get("views") or {}
        for view in ("interior", "fallback"): r[f"{view}_n_tiles"] = (v.get(view) or {}).get("n_tiles"); r[f"{view}_n_blocks"] = (v.get(view) or {}).get("n_blocks")
        r["view_B8"] = v.get("view_B8"); r["files"] = sorted(os.path.basename(f) for f in glob.glob(f"{ROOT}/build/{pid}/*")); r["fields_dir"] = os.path.isdir(f"{ROOT}/fields/{pid}"); r["inputs_dir"] = os.path.isdir(f"{ROOT}/inputs/{pid}")
        if st.get("status") == "built":
            t = pd.read_parquet(f"{ROOT}/build/{pid}/tiles_v1_1.parquet", columns=["tile_id", "n_joint", "A_fine"]); r["n_tiles_with_joint_px"] = int((t.n_joint > 0).sum()); r["n_A_fine_finite_on_those"] = int(np.isfinite(t.A_fine.values[t.n_joint.values > 0]).sum()); r["n_A_fine_finite_all"] = int(np.isfinite(t.A_fine.values).sum()); del t
            ti = pd.read_parquet(f"{ROOT}/build/{pid}/tiles_int.parquet", columns=["tile_id", "interior_D1", "A_fine_int_D1"]); r["n_interior_D1_flag"] = int(ti.interior_D1.sum()); r["n_A_fine_int_defined"] = int(ti.A_fine_int_D1.notna().sum()); del ti
            r["finite_target_everywhere"] = bool(r["n_A_fine_finite_on_those"] == r["n_tiles_with_joint_px"])
        es, ep = EXPECT[pid]; r["expected_status"] = es; r["expected_product_features"] = ep
        r["ok"] = bool(st.get("status") == es and (ep is None or st.get("product_features") == ep) and (st.get("status") != P4.NO_K or r["files"] == ["pair_stats.json"]) and (st.get("status") != "built" or r["finite_target_everywhere"]))
        out["pairs"][pid] = r
    prov = sorted(glob.glob(f"{ROOT}/provenance/phase4_build.py_*.json"))
    if prov:
        pr = json.load(open(prov[-1])); out["provenance"] = dict(path=prov[-1], host=pr.get("host"), slurm_job=pr.get("slurm_job"), git_head=pr.get("git_head"), guard=[pr.get("guard_probes_blocked"), pr.get("guard_probes")], signoff_sha256=pr.get("signoff_sha256"), supplement=pr.get("supplement"), acq_run_record={k: pr["acq_run_record"].get(k) for k in ("path", "sha256", "status", "mode", "freeze_record_sha256", "supplement_sha256")} if pr.get("acq_run_record") else None, allowlist_n=len(pr.get("allowlist") or []), prereg_sha256=(pr.get("verify") or {}).get("prereg", {}).get("sha256"), script_sha256=pr.get("script_sha256"))
    if log and os.path.exists(log): out["log_redacted"] = [redact(l.rstrip()) for l in open(log, errors="replace") if l.strip().startswith(("built ", "  built ", "  KN", "[phase4_common.start]", "===", "CMD", "done"))]
    mh = out["must_hold"]; P = out["pairs"]
    mh["1_locks_passed_citation_record"] = bool(out.get("provenance") and out["provenance"]["acq_run_record"] and out["provenance"]["acq_run_record"]["sha256"] == "6e51efd61f36f92066ff91de2749cdd7ce85192c6f8d20611c1a3de45ffa431e")
    mh["2_KN182L03_no_k_only_pair_stats"] = bool(P["KN182L03__MGDS_30193"].get("ok")); mh["3_product_features"] = bool(all(P[p].get("product_features") == EXPECT[p][1] for p in PAIRS if EXPECT[p][1])); mh["4_finite_target_everywhere"] = bool(all(P[p].get("finite_target_everywhere") for p in PAIRS if EXPECT[p][0] == "built"))
    mh["all"] = bool(all(v for k, v in mh.items() if k != "all") and all(r.get("ok") for r in P.values()))
    os.makedirs(f"{DR}/R12c/tables", exist_ok=True); json.dump(out, open(f"{DR}/R12c/tables/build_check.json", "w"), indent=1, default=str)
    for pid, r in P.items(): print(pid, r.get("status"), "ok" if r.get("ok") else "FAIL", "tiles", r.get("n_tiles_v1_1"), "interior", r.get("interior_n_tiles"), "/", r.get("interior_n_blocks"), "fallback", r.get("fallback_n_tiles"), "/", r.get("fallback_n_blocks"), "view", r.get("view_B8"), "D1", r.get("D1_px"), "cells", r.get("cells_across"), "prod", r.get("product_features"))
    print("must_hold", mh)
