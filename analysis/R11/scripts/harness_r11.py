"""R11 §A2.3 / §A3 — the acquisition hand-off harness (rehearsal only; the harness supplement never reaches a real run).
`make`: verify the ACQ-R06 rehearsal supplement on OAK (acq_r06_rehearsal/SHA256SUMS, itself sha256 18cf21b9…f3c9) and write a
        harness supplement holding its rows for TN399__MGDS_33068, FK151121__MGDS_22383 and EX1202L2__MGDS_20815 under the R11
        rehearsal root, plus a stats JSON for the score script (role, no-k status of EX1202L2 from the harness build);
        also the schema comparison of §A2.1 (column names, dtypes, empty-value conventions) against manifest v2.1.
The builds themselves run through phase4_build.py --rows / --out_root (cpu_step CMD), the scoring through phase4_score.py --rows
--acq_record (the ACQ-R06b ten-pair rerun record, verified against acq_r06b_rehearsal/SHA256SUMS)."""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "phase4", "scripts"))
import phase4_common as P4
OAK = P4.C4.DATA; ROOT = P4.ROOT["rehearsal"]; HROOT = f"{ROOT}/harness"; PAIRS = ["TN399__MGDS_33068", "FK151121__MGDS_22383", "EX1202L2__MGDS_20815"]
R06_SUMS_SHA = "18cf21b9abd64938ecb03b090c4c4f95c732e77af388cd7630f3d317fba9f3c9"; SUP_SHA = "297e0a0e6d88b6e7fa6965822b6beefe1b2201bf25df8bf3849f4576b053b516"
def sums(d):
    p = f"{OAK}/{d}/SHA256SUMS"; return p, {l.split(maxsplit=1)[1].strip().lstrip("./"): l.split()[0] for l in open(p) if l.strip()}
def verified(d, rel, sums_sha=None):
    p, S = sums(d)
    if sums_sha: assert P4.sha256_file(p) == sums_sha, (p, P4.sha256_file(p))
    f = f"{OAK}/{d}/{rel}"; h = P4.sha256_file(f); assert h == S[rel], (f, h, S[rel]); return f, h
def empties(v):
    if v is None: return "None"
    if isinstance(v, float) and np.isnan(v): return "NaN"
    if isinstance(v, str) and v == "": return "''"
    return None
if __name__ == "__main__":
    m, root, rec = P4.start("harness_r11.py", "rehearsal", PAIRS[:2]); os.makedirs(HROOT, exist_ok=True)
    sup_path, sup_sha = verified("acq_r06_rehearsal", "records/rehearsal/supplement_rehearsal_full.parquet", R06_SUMS_SHA); assert sup_sha == SUP_SHA
    r06b_path, r06b_sha = verified("acq_r06b_rehearsal", "records/rehearsal/step1_run_rehearsal_r06b.json")
    S = pd.read_parquet(sup_path)
    # §A2.1 schema comparison
    schema = dict(columns_identical=bool(list(S.columns) == list(m.columns)), n_columns=[len(S.columns), len(m.columns)], dtype_differences={c: [str(S[c].dtype), str(m[c].dtype)] for c in m.columns if str(S[c].dtype) != str(m[c].dtype)},
                  value_kinds={c: dict(supplement=sorted({type(v).__name__ for v in S[c]}), manifest=sorted({type(v).__name__ for v in m[c]})) for c in ("ship_products_usable", "max_recoverable_k", "products_v2_1_flags", "k_sweep_status", "manifest_version")},
                  empty_conventions={c: dict(supplement=sorted({e for e in map(empties, S[c]) if e}), manifest=sorted({e for e in map(empties, m[c]) if e})) for c in m.columns if any(map(empties, S[c])) or any(map(empties, m[c]))},
                  lock_checks=dict(column_names="load_supplement compares list(columns) only (names and order); dtypes are not compared", usable_reader="r04_common.ship_usable accepts True / 1 / 'true' (bool and object columns alike)", k_reader="row_status / k_provenance accept float NaN or None for an empty k"))
    diffs = {}
    for pid in PAIRS:
        a = S[S.pair_id == pid].iloc[0]; b = m[m.pair_id == pid].iloc[0]; d = {}
        for c in m.columns:
            x, y = a[c], b[c]; same = (x == y) if not (isinstance(x, float) and isinstance(y, float)) else (x == y or (np.isnan(x) and np.isnan(y)))
            if not bool(same) and not (x is None and y is None): d[c] = [None if x is None else (float(x) if isinstance(x, (float, np.floating)) else str(x)[:160]), None if y is None else (float(y) if isinstance(y, (float, np.floating)) else str(y)[:160])]
        diffs[pid] = d
    H = S[S.pair_id.isin(PAIRS)].reset_index(drop=True); hp = f"{HROOT}/harness_supplement.parquet"; H.to_parquet(hp, index=False)
    stats = {pid: dict(role="development", status=(P4.NO_K if P4.row_status(H[H.pair_id == pid].iloc[0]) == P4.NO_K else "ok")) for pid in PAIRS}
    json.dump(stats, open(f"{HROOT}/harness_stats.json", "w"), indent=1)
    json.dump(dict(source=dict(path=sup_path, sha256=sup_sha, sums_sha256=R06_SUMS_SHA), acq_r06b_record=dict(path=r06b_path, sha256=r06b_sha), harness=dict(path=hp, sha256=P4.sha256_file(hp), pairs=PAIRS, row_status={pid: P4.row_status(H[H.pair_id == pid].iloc[0]) for pid in PAIRS}),
                   schema=schema, row_differences_vs_v2_1=diffs, note="harness only: lives under the R11 rehearsal root, never at the supplement path; phase4 mode admits exactly the five sealed ids"), open(f"{HROOT}/harness_record.json", "w"), indent=1, default=str)
    print(json.dumps(schema, indent=1)); print({p: list(d) for p, d in diffs.items()}); print("harness", hp)
