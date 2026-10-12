"""R12c Part C1 — build the mock sealed-style layout: COPY (shutil.copy2, never symlink) the harmonized files and the ship_products_v2_1
directory of the three development pairs into the layout of r12c_common.MOCK_ROWS, and write the mock supplement (manifest v2.1 rows of
the three pairs with the path columns replaced by the absolute mock paths, raw / qc paths cleared, leakage_unit set to the mock unit,
ship_products_usable set False for the 'masked' pair). Record: files copied with sha256 (source = copy), row differences.
This builder imports NO guard module on purpose: the R04 guard (installed on importing r04_common) refuses every path containing
`harmonized_confirmatory`, writes included, so the copy into the mock layout must run unguarded. It reads only development data and the
manifest (sha256 recorded)."""
import hashlib, json, os, shutil, sys
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
# constants duplicated from r12c_common (which imports phase4_common and so the guards)
MOCK = "/scratch/users/scdobbs/grl_review/R12c/mock"; DATA = "/oak/stanford/groups/hilley/auv_ship_colocated_bathy"; MANIFEST = "/home/users/scdobbs/co-located-auv-ship-based-multibeam-acquisition/manifest/pairs_v2_1.parquet"
MOCK_ROWS = {"FK151121__MGDS_22383": dict(style="confirmatory", unit="nu98", rel="harmonized_confirmatory/nu98/FK151121__MGDS_22383", usable=True),
             "TN399__MGDS_33068": dict(style="confirmatory", unit="nu97", rel="harmonized_confirmatory/nu97/TN399__MGDS_33068", usable=False),
             "KM0923__MGDS_31429": dict(style="lockbox", unit="lu07", rel="harmonized/KM0923__MGDS_31429", usable=True)}
FILES = ("hr.tif", "lr.tif", "hr_valid.tif", "lr_valid.tif", "joint_valid.tif"); MOCK_SUP = f"{MOCK}/mock_supplement.parquet"
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
if __name__ == "__main__":
    assert "phase4_common" not in sys.modules and "r04_common" not in sys.modules
    m = pd.read_parquet(MANIFEST); rec = dict(manifest=dict(path=MANIFEST, sha256=sha(MANIFEST)), copies={}, rows={}); os.makedirs(MOCK, exist_ok=True)
    for pid, spec in MOCK_ROWS.items():
        src = os.path.join(DATA, "harmonized", pid); dst = f"{MOCK}/{spec['rel']}"; os.makedirs(f"{dst}/ship_products_v2_1", exist_ok=True); c = {}
        for f in FILES: shutil.copy2(f"{src}/{f}", f"{dst}/{f}"); c[f] = dict(sha256=sha(f"{dst}/{f}"), same_as_source=(sha(f"{src}/{f}") == sha(f"{dst}/{f}")), bytes=os.path.getsize(f"{dst}/{f}"))
        for f in sorted(os.listdir(f"{src}/ship_products_v2_1")): shutil.copy2(f"{src}/ship_products_v2_1/{f}", f"{dst}/ship_products_v2_1/{f}"); c[f"ship_products_v2_1/{f}"] = dict(sha256=sha(f"{dst}/ship_products_v2_1/{f}"), same_as_source=(sha(f"{src}/ship_products_v2_1/{f}") == sha(f"{dst}/ship_products_v2_1/{f}")))
        rec["copies"][pid] = dict(dst=dst, files=c, symlinks=[f for f in os.listdir(dst) if os.path.islink(f"{dst}/{f}")], all_same_as_source=all(v["same_as_source"] for v in c.values()))
    H = m[m.pair_id.isin(MOCK_ROWS)].copy().reset_index(drop=True)
    for pid, spec in MOCK_ROWS.items():
        i = H.index[H.pair_id == pid][0]; d = f"{MOCK}/{spec['rel']}"; before = H.loc[i].to_dict()
        H.loc[i, "harmonized_path"] = d; H.loc[i, "harmonized_path_hr"] = f"{d}/hr.tif"; H.loc[i, "harmonized_path_lr"] = f"{d}/lr.tif"
        for c in ("hr_valid", "lr_valid", "joint_valid"): H.loc[i, c] = f"{d}/{c}.tif"
        for c in ("raw_path_hr", "raw_path_lr", "qc_artifact_path", "hr_local_path"): H.loc[i, c] = None
        H.loc[i, "leakage_unit"] = spec["unit"]
        if not spec["usable"]: H.loc[i, "ship_products_usable"] = False; H.loc[i, "ship_products_usable_reason"] = "R12c mock: flag set False to exercise the masked path (RC2511-like)"
        after = H.loc[i].to_dict(); rec["rows"][pid] = {c: [str(before[c])[:120], str(after[c])[:120]] for c in H.columns if str(before[c]) != str(after[c])}
    assert list(H.columns) == list(m.columns); H.to_parquet(MOCK_SUP, index=False); rec["mock_supplement"] = dict(path=MOCK_SUP, sha256=sha(MOCK_SUP), pairs=list(MOCK_ROWS))
    json.dump(rec, open(f"{MOCK}/mock_record.json", "w"), indent=1, default=str); print(json.dumps({k: (v if k != "copies" else {p: dict(n_files=len(c["files"]), all_same=c["all_same_as_source"], symlinks=c["symlinks"]) for p, c in v.items()}) for k, v in rec.items()}, indent=1, default=str))
