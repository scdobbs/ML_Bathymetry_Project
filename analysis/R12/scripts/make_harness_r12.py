"""R12 §A2: write the k = 32 harness supplement (rehearsal only) under the R12 rehearsal root. The rows are the manifest v2.1
rows of FK151121__MGDS_22383 (H1) and NA080__MGDS_31289 (H2) with ONLY lr_native_res_m and max_recoverable_k changed
(150.77 / 32 and 124.62 / 32: the sealed geometry of SUM1004 and RC2511). The ship and AUV data paths are unchanged.
A record (original and harness values, sha256 of v2.1 and of the harness file) is written beside it."""
import json, os
import pandas as pd
import r12_common as R12
P4 = R12.P4; C4 = P4.C4

def same(a, b):
    if a is None and b is None: return True
    if isinstance(a, float) and isinstance(b, float) and a != a and b != b: return True
    try: return bool(a == b)
    except Exception: return False

if __name__ == "__main__":
    m = C4.load_manifest(); os.makedirs(os.path.dirname(R12.HARNESS), exist_ok=True)
    H = m[m.pair_id.isin(R12.PAIRS)].copy().reset_index(drop=True); rec = {}
    for pid, h in R12.HARNESS_ROWS.items():
        i = H.index[H.pair_id == pid][0]; r = H.loc[i]
        rec[pid] = dict(label=h["label"], original=dict(hr_native_res_m=float(r.hr_native_res_m), lr_native_res_m=float(r.lr_native_res_m), max_recoverable_k=float(r.max_recoverable_k), leakage_unit=str(r.leakage_unit), ship_products_usable=bool(C4.ship_usable(r))),
                        harness=dict(lr_native_res_m=h["lr_native_res_m"], max_recoverable_k=h["max_recoverable_k"]))
        H.loc[i, "lr_native_res_m"] = float(h["lr_native_res_m"]); H.loc[i, "max_recoverable_k"] = float(h["max_recoverable_k"])
    assert list(H.columns) == list(m.columns)
    changed = {pid: [c for c in m.columns if not same(m[m.pair_id == pid].iloc[0][c], H[H.pair_id == pid].iloc[0][c])] for pid in R12.PAIRS}
    assert all(sorted(v) == ["lr_native_res_m", "max_recoverable_k"] for v in changed.values()), changed
    H.to_parquet(R12.HARNESS, index=False)
    out = dict(harness=R12.HARNESS, sha256=P4.sha256_file(R12.HARNESS), manifest_v2_1=dict(path=C4.MANIFEST, sha256=m.attrs.get("sha256")), rows=rec, columns_changed=changed,
               note="rehearsal only (R11 §A2.3 --rows mechanism); the harness file lives under the R12 rehearsal root, never at the supplement path; phase4 mode admits exactly the five sealed ids and refuses --rows")
    json.dump(out, open(f"{os.path.dirname(R12.HARNESS)}/harness_record.json", "w"), indent=1, default=str); print(json.dumps(out, indent=1, default=str))
