"""R16c §2.1 (= R16 Part C1 against freeze v2) — build the mock layout named for the real Phase 5 roots (see r16c_common). COPY (shutil.copy2, never symlink):
  - the ACQ-R11b rehearsal r11b harmonized directories of NA080__MGDS_31289, KM0923__MGDS_31429, KM0923__MOCK_R09repeat and the
    ACQ-R11b harness (ruling S3) directory of FK151121__MGDS_22383 -> mock/auv_ship_colocated_bathy/phase5/harmonized/<pair>/;
  - the ACQ-R11b rehearsal run record step1_run_rehearsal_r11b.json and the freeze record PHASE5/freeze_record_phase5_v2.json ->
    mock/reports_post_grl_review/PHASE5/ (byte copies; sha256 asserted equal);
  - empty mock/auv_ship_colocated_bathy/sealed_candidates_v3/<hr id>/ directories (probe targets; nothing in them).
Mock supplement mock/manifest/pairs_phase5_sealed.parquet = the four R16 rows of supplement_rehearsal_r11b.parquet (70 columns,
schema unchanged) with: every path column -> the absolute mock path (raw_path_hr -> the mock sealed_candidates_v3/<hr id>);
FK151121's row given the harness values (lr_native 65.99, res_ratio 65.99, offsets 9.05 / -6.22, cells_across 3.880437,
eligible, checksums of the copied harness rasters; ruling S3); KM0923__MOCK_R09repeat's row set coreg_pass = False,
eligible = False with a mock reason (the not-testable row, R16 §C1). NR07-1__MGDS_31813 (5th r11b row) is not used. The mock ledger gets the mock supplement's sha256.
This builder imports NO guard module on purpose (the Phase 5 root token `auv_ship_colocated_bathy/phase5` is in every mock
path). It reads only ACQ-R11b rehearsal outputs (development data) and records every row difference."""
import hashlib, json, os, shutil, sys
import pandas as pd
BASE = "/scratch/users/scdobbs/grl_review/R16c"; MOCK = f"{BASE}/mock"; MOCK_DATA = f"{MOCK}/auv_ship_colocated_bathy"; MOCK_HARM = f"{MOCK_DATA}/phase5/harmonized"; MOCK_CAND = f"{MOCK_DATA}/sealed_candidates_v3"
ACQ = "/home/users/scdobbs/co-located-auv-ship-based-multibeam-acquisition"; REH = f"{ACQ}/reports_post_grl_review/ACQ-R11b/rehearsal"; FREEZE = f"{ACQ}/reports_post_grl_review/PHASE5/freeze_record_phase5_v2.json"
OUTS = "/scratch/groups/hilley/auv_ship_colocated_bathy/ACQ-R11b/rehearsal"
SRC = {"NA080__MGDS_31289": f"{OUTS}/r11b/harmonized/NA080__MGDS_31289", "KM0923__MGDS_31429": f"{OUTS}/r11b/harmonized/KM0923__MGDS_31429",
       "FK151121__MGDS_22383": f"{OUTS}/r11b_fk151121_isolation/harmonized/FK151121__MGDS_22383", "KM0923__MOCK_R09repeat": f"{OUTS}/r11b/harmonized/KM0923__MOCK_R09repeat"}
MOCK_SUP = f"{MOCK}/manifest/pairs_phase5_sealed.parquet"; MOCK_LEDGER = f"{MOCK}/CHANGE_LEDGER_mock.md"; MOCK_PHASE5 = f"{MOCK}/reports_post_grl_review/PHASE5"
PATHCOLS = {"harmonized_path": "", "harmonized_path_hr": "/hr.tif", "harmonized_path_lr": "/lr.tif", "hr_valid": "/hr_valid.tif", "lr_valid": "/lr_valid.tif", "joint_valid": "/joint_valid.tif", "footprint_wkt": "/footprint.geojson"}
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
if __name__ == "__main__":
    assert not any(k in sys.modules for k in ("phase5_common", "phase4_common", "r04_common")), "make_mock must run unguarded"
    rec = dict(copies={}, rows={}, sources=SRC); os.makedirs(MOCK_HARM, exist_ok=True); os.makedirs(MOCK_PHASE5, exist_ok=True); os.makedirs(os.path.dirname(MOCK_SUP), exist_ok=True)
    for pid, src in SRC.items():
        dst = f"{MOCK_HARM}/{pid}"
        if os.path.exists(dst): shutil.rmtree(dst)
        shutil.copytree(src, dst, symlinks=False, copy_function=shutil.copy2); c = {}
        for dp, _, fs in os.walk(src):
            for f in fs:
                s = os.path.join(dp, f); d = os.path.join(dst, os.path.relpath(s, src)); c[os.path.relpath(s, src)] = dict(sha256=sha(d), same_as_source=(sha(s) == sha(d)), link=os.path.islink(d))
        rec["copies"][pid] = dict(src=src, dst=dst, n_files=len(c), all_same_as_source=all(v["same_as_source"] for v in c.values()), any_link=any(v["link"] for v in c.values()), files=c)
        os.makedirs(f"{MOCK_CAND}/{pid.partition('__')[2]}", exist_ok=True)
    for s, d in ((f"{REH}/step1_run_rehearsal_r11b.json", f"{MOCK_PHASE5}/step1_run_rehearsal_r11b.json"), (FREEZE, f"{MOCK_PHASE5}/freeze_record_phase5_v2.json")):
        if os.path.exists(d): os.chmod(d, 0o644); os.remove(d)
        shutil.copy2(s, d); rec["copies"][os.path.basename(d)] = dict(src=s, dst=d, sha256=sha(d), same_as_source=(sha(s) == sha(d)))
    S = pd.read_parquet(f"{REH}/supplement_rehearsal_r11b.parquet"); rec["rehearsal_supplement"] = dict(path=f"{REH}/supplement_rehearsal_r11b.parquet", sha256=sha(f"{REH}/supplement_rehearsal_r11b.parquet"), rows=list(S.pair_id))
    S = S[S.pair_id.isin(SRC)].reset_index(drop=True); assert sorted(S.pair_id) == sorted(SRC), list(S.pair_id); rec["rows_used"] = list(S.pair_id)
    H = S.copy().reset_index(drop=True); hfk = json.load(open(f"{REH}/harness_fk151121_r11b.json"))["per_pair"]; site = json.load(open(f"{SRC['FK151121__MGDS_22383']}/site_record.json"))
    for i in H.index:
        pid = H.loc[i, "pair_id"]; d = f"{MOCK_HARM}/{pid}"; before = H.loc[i].to_dict()
        for c, suf in PATHCOLS.items(): H.loc[i, c] = d + suf
        H.loc[i, "raw_path_hr"] = f"{MOCK_CAND}/{pid.partition('__')[2]}"
        for c, f in (("checksum_hr", "hr.tif"), ("checksum_lr", "lr.tif"), ("checksum_hr_valid", "hr_valid.tif"), ("checksum_lr_valid", "lr_valid.tif"), ("checksum_joint_valid", "joint_valid.tif")): H.loc[i, c] = sha(f"{d}/{f}")
        if pid == "FK151121__MGDS_22383":
            co = site["harmonize"]["coreg"]
            for c, v in (("lr_native_res_m", hfk["lr_native_res_m"]), ("res_ratio", hfk["res_ratio"]), ("hr_native_res_m", hfk["hr_native_res_m"]), ("max_recoverable_k", float(hfk["max_recoverable_k"])), ("k_sweep_status", hfk["k_sweep_status"]),
                         ("ship_products_usable", bool(hfk["ship_products_usable"])), ("horiz_offset_m", co["offset_m"]), ("vert_offset_m", co["dz_m"]), ("coreg_pass", bool(hfk["coreg_pass"])), ("k_recoverable", bool(hfk["k_recoverable"])),
                         ("cells_across", float(hfk["cells_across"])), ("hr_res_in_range", bool(hfk["hr_res_in_range"])), ("eligible", bool(hfk["eligible"])), ("eligibility_reason", hfk["eligibility_reason"])):
                H.loc[i, c] = v
            H.loc[i, "notes"] = str(H.loc[i, "notes"]) + " | R16 mock: values of the ACQ-R11b FK151121 harness (ruling S3, whole-footprint development region)"
        if pid == "KM0923__MOCK_R09repeat":
            H.loc[i, "coreg_pass"] = False; H.loc[i, "eligible"] = False; H.loc[i, "eligibility_reason"] = "1 co-registration (R16 mock: coreg_pass set False to exercise the not-testable path)"
        after = H.loc[i].to_dict(); rec["rows"][pid] = {c: [str(before[c])[:110], str(after[c])[:110]] for c in H.columns if str(before[c]) != str(after[c])}
    assert list(H.columns) == list(S.columns)
    for c in ("coreg_pass", "k_recoverable", "hr_res_in_range", "eligible", "ship_products_usable"): H[c] = H[c].astype(bool)
    import pyarrow as pa, pyarrow.parquet as pq; ref = pq.read_schema(f"{REH}/supplement_rehearsal_r11b.parquet")
    pq.write_table(pa.Table.from_pandas(H, schema=ref, preserve_index=False), MOCK_SUP); ssha = sha(MOCK_SUP)   # written with the rehearsal supplement's arrow schema (all-null string columns keep type string)
    b = pq.read_schema(MOCK_SUP); rec["schema_diff_vs_rehearsal_supplement"] = [(x.name, str(x.type), str(y.type)) for x, y in zip(ref, b) if x.name != y.name or x.type != y.type]; assert not rec["schema_diff_vs_rehearsal_supplement"], rec["schema_diff_vs_rehearsal_supplement"]
    with open(MOCK_LEDGER, "w") as f: f.write("# Mock ledger (R16c dress rehearsal; scratch only; never committed)\n\n| # | what | sha256 |\n|---|---|---|\n" + f"| m1 | mock supplement {MOCK_SUP} | {ssha} |\n")
    rec["mock_supplement"] = dict(path=MOCK_SUP, sha256=ssha, pairs=list(H.pair_id), eligible={r.pair_id: bool(r.eligible) for r in H.itertuples()}); rec["mock_ledger"] = MOCK_LEDGER
    json.dump(rec, open(f"{MOCK}/mock_record.json", "w"), indent=1, default=str)
    print(json.dumps({k: (v if k != "copies" else {p: {kk: vv for kk, vv in c.items() if kk != "files"} for p, c in v.items()}) for k, v in rec.items()}, indent=1, default=str))
