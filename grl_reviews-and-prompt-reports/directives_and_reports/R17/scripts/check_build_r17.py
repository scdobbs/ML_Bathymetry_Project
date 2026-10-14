"""R17 §2.2 / §3.1 must-hold checks on the Phase 5 build (grl_review_archive/phase5), COUNTS ONLY. Reads only the fields the report may carry
(status, tile / interior / fallback / block counts, view at B_min = 8, D1, cells across, product-feature decision); amplitude statistics and every
other field of pair_stats.json are dropped before anything is printed or written. No raster value is printed.

Boundary test (§2.2). ACQ-R13 zeroed joint_valid.tif on every tile of the stride-256 lattice (origin = the harmonized grid's (0, 0)) that crosses a
co-registration group boundary (drift case; localcoreg_p5.finalize). The group masks are not on disk, so a zeroed lattice tile is detected as one whose
joint_valid.tif is all 0 while the pre-zeroing joint (colocated_utils.compute_joint_valid derivation: HR valid & ship valid resampled nearest) is not
empty. The count is compared with the run record's n_tiles_dropped_boundary; the rigid sites are the control (expected 0). Then, per site:
  - CNN candidate windows (stride 128) lying wholly inside zeroed lattice tiles, and how many of them the frozen filter excluded;
  - kept tiles (tiles_v1_1; tile = the window's central 128 px) whose tile footprint touches a zeroed lattice tile, and the number of
    target-joint (joint_v1) pixels of kept tiles inside zeroed lattice tiles. Must-hold: both 0.
Runs through phase5_common.start (guard self-test, every phase5 lock, allow-list of the eligible sites; writes a provenance record).
Writes R17/tables/build_check.json and prints one count line per site."""
import glob, json, os, re, sys
import numpy as np, pandas as pd, rasterio
HERE = os.path.dirname(os.path.abspath(__file__)); DR = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, os.path.join(DR, "phase5", "scripts"))
import phase5_common as P5
import r04_common as C4
from preprocessing.colocated_utils import resolve_pair_paths, compute_joint_valid
ROOT = P5.ROOT["phase5"]; ALL8 = list(P5.SEALED8); WIN, TILE, MARGIN, STRIDE = 256, 128, 64, 128
ALLOWED = ("pair", "unit", "status", "lr_native", "k", "D1_px", "cells_across", "dev_min_cells", "thinner_than_development", "n_tiles_v1", "n_tiles_v1_1", "n_blocks_kept",
           "n_feature_rows", "ship_products_usable", "product_features", "contract_columns_frac_nan", "seconds")
def redact(s): return re.sub(r"amplitude [0-9.eE+\-naN]+ m", "amplitude [redacted]", s)

def boundary(pid, m, acq_pp):
    row = C4.pair_row(m, pid); paths = resolve_pair_paths(C4.DATA, row)
    jf, _ = compute_joint_valid(paths); jf = jf.astype(bool)
    jd, _ = compute_joint_valid(dict(paths, joint_valid=None)); jd = jd.astype(bool); H, W = jf.shape
    Z = np.zeros((H, W), bool); nz = 0
    for r0 in range(0, H - WIN + 1, WIN):
        for c0 in range(0, W - WIN + 1, WIN):
            s = (slice(r0, r0 + WIN), slice(c0, c0 + WIN))
            if not jf[s].any() and jd[s].any(): Z[s] = True; nz += 1
    out = dict(lattice_tiles_zeroed_detected=nz, run_record_n_tiles_dropped_boundary=acq_pp.get("n_tiles_dropped_boundary"), coreg_case=acq_pp.get("coreg_case"),
               px_jointfile_gt_derived=int((jf & ~jd).sum()) )
    del jd
    # CNN windows wholly inside zeroed lattice tiles; were they excluded?
    t1 = pd.read_parquet(f"{ROOT}/build/{pid}/tiles_v1.parquet", columns=["r0", "c0"]); t11 = pd.read_parquet(f"{ROOT}/build/{pid}/tiles_v1_1.parquet", columns=["r0", "c0"])
    acc1 = set(zip(t1.r0.astype(int), t1.c0.astype(int))); kept = set(zip(t11.r0.astype(int), t11.c0.astype(int)))
    inside = [(r0, c0) for r0 in range(0, H - WIN + 1, STRIDE) for c0 in range(0, W - WIN + 1, STRIDE) if Z[r0:r0 + WIN, c0:c0 + WIN].all()]
    out["windows_wholly_in_zeroed"] = len(inside); out["of_those_accepted_tiles_v1"] = sum(p in acc1 for p in inside); out["of_those_kept_tiles_v1_1"] = sum(p in kept for p in inside)
    with rasterio.open(f"{ROOT}/fields/{pid}/joint_v1.tif") as s: J = s.read(1) == 1
    touch = 0; px = 0; tiles_with_px = 0
    for r0, c0 in kept:
        s = (slice(r0 + MARGIN, r0 + MARGIN + TILE), slice(c0 + MARGIN, c0 + MARGIN + TILE)); z = Z[s]
        if z.any():
            touch += 1; n = int((J[s] & z).sum()); px += n; tiles_with_px += int(n > 0)
    out.update(kept_tiles=len(kept), kept_tiles_touching_zeroed=touch, kept_tiles_with_target_px_in_zeroed=tiles_with_px, target_px_in_zeroed_total=px,
               must_hold=bool(tiles_with_px == 0 and out["of_those_kept_tiles_v1_1"] == 0))
    return out

if __name__ == "__main__":
    _m, _root, _rec = P5.start("check_build_r17.py", "phase5", ALL8); assert _root == ROOT
    adm = list(_rec["admitted"]); acq = _rec["acq_run_record"]; per = acq.get("per_pair") or {}
    log = sys.argv[1] if len(sys.argv) > 1 else None
    out = dict(root=ROOT, admitted=adm, not_testable=_rec.get("not_testable"), start_record=dict(guard=[_rec.get("guard_probes_blocked"), _rec.get("guard_probes")], guard_p5=[_rec.get("guard_p5_probes_blocked"), _rec.get("guard_p5_probes")],
               acq_run_record=dict(path=acq["path"], sha256=acq["sha256"]), supplement_sha256=_rec["supplement"]["sha256"], phase5_common_plumbing=_rec["verify"]["phase5_common.py"].get("plumbing")), pairs={}, must_hold={})
    es = json.load(open(f"{ROOT}/build/eligibility_status.json")); out["eligibility_status"] = es["status"]
    for pid in adm:
        st = json.load(open(f"{ROOT}/build/{pid}/pair_stats.json")); r = {k: st.get(k) for k in ALLOWED if k in st}; v = st.get("views") or {}
        for view in ("interior", "fallback"): r[f"{view}_n_tiles"] = (v.get(view) or {}).get("n_tiles"); r[f"{view}_n_blocks"] = (v.get(view) or {}).get("n_blocks")
        r["view_B8"] = v.get("view_B8"); r["ship_products_usable_acq"] = per.get(pid, {}).get("ship_products_usable")
        r["boundary"] = boundary(pid, _m, per.get(pid, {}))
        out["pairs"][pid] = r
        b = r["boundary"]; print(f"{pid}: status {r['status']}; tiles v1 {r['n_tiles_v1']} -> v1.1 {r['n_tiles_v1_1']}; interior {r['interior_n_tiles']}/{r['interior_n_blocks']}; fallback {r['fallback_n_tiles']}/{r['fallback_n_blocks']}; view {r['view_B8']}; products {r['product_features']}; "
              f"zeroed lattice detected {b['lattice_tiles_zeroed_detected']} (record {b['run_record_n_tiles_dropped_boundary']}); windows inside {b['windows_wholly_in_zeroed']} accepted {b['of_those_accepted_tiles_v1']} kept {b['of_those_kept_tiles_v1_1']}; "
              f"kept tiles touching {b['kept_tiles_touching_zeroed']}, with target px {b['kept_tiles_with_target_px_in_zeroed']} ({b['target_px_in_zeroed_total']} px); must_hold {b['must_hold']}", flush=True)
    prov = sorted(glob.glob(f"{ROOT}/provenance/phase5_build.py_*.json"))
    if prov:
        pr = json.load(open(prov[-1])); out["build_provenance"] = dict(path=prov[-1], slurm_job=pr.get("slurm_job"), git_head=pr.get("git_head"), guard=[pr.get("guard_probes_blocked"), pr.get("guard_probes")], guard_p5=[pr.get("guard_p5_probes_blocked"), pr.get("guard_p5_probes")])
    if log and os.path.exists(log): out["log_redacted"] = [redact(l.rstrip()) for l in open(log, errors="replace") if l.strip().startswith(("built ", "[phase5_common.start]", "===", "CMD", "done")) or "not testable" in l]
    mh = out["must_hold"]; P = out["pairs"]
    mh["1_five_built"] = sorted(adm) == sorted(["Escanaba__MGDS_30368", "KN161L04__MGDS_22436", "KN173L02__MGDS_31675", "MV1003__MGDS_31860", "AT26-24__MGDS_31954"]) and all(P[p]["status"] == "built" for p in adm)
    mh["2_boundary_tiles_excluded"] = all(P[p]["boundary"]["must_hold"] for p in adm)
    mh["2b_zeroed_detected_equals_record"] = all(P[p]["boundary"]["lattice_tiles_zeroed_detected"] == P[p]["boundary"]["run_record_n_tiles_dropped_boundary"] for p in adm)
    mh["3_product_features_match_supplement"] = all((P[p]["product_features"] == "used") == bool(P[p]["ship_products_usable_acq"]) for p in adm)
    mh["4_not_testable_unbuilt"] = all(not os.path.exists(f"{ROOT}/build/{p}") and not os.path.exists(f"{ROOT}/fields/{p}") for p in ALL8 if p not in adm)
    mh["all"] = all(v for k, v in mh.items() if k != "all"); print("must_hold:", json.dumps(mh), flush=True)
    json.dump(out, open(f"{DR}/R17/tables/build_check.json", "w"), indent=1, default=str)
