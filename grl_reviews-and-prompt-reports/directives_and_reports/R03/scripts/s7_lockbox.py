"""R03 §7 — lockbox record: pair ids and sha256 of their manifest rows (the rasters are never opened); grep evidence
that every new R03 script installs the guard."""
import sys, os, glob, json, subprocess
sys.path.insert(0, "/scratch/users/scdobbs/grl_review/R03"); import r03_common as C
sys.path.insert(0, "/home/users/scdobbs/GAN_Bathymetry_Project"); from preprocessing.colocated_utils import load_manifest
m = load_manifest(C.DATA); rec = {p: C.manifest_row_sha256(m[m.pair_id == p].iloc[0]) for p in C.LOCKBOX}
scripts = sorted(glob.glob(f"{C.ROOT}/*.py")); ev = {}
for s in scripts:
    txt = open(s).read(); ev[os.path.basename(s)] = dict(imports_r03_common=("import r03_common" in txt), defines_guard=("install_lockbox_guard" in txt))
grep = subprocess.run(["grep", "-l", "r03_common", *scripts], capture_output=True, text=True).stdout.split()
out = dict(lockbox_pairs=list(C.LOCKBOX), manifest_row_sha256=rec, guard_mechanism="r03_common.install_lockbox_guard() wraps builtins.open, rasterio.open and numpy.load; any path containing a lockbox pair id raises PermissionError before any read; run at import",
           scripts=ev, grep_l_r03_common=[os.path.basename(x) for x in grep], all_scripts_guarded=all(v["imports_r03_common"] or v["defines_guard"] for v in ev.values()))
os.makedirs(f"{C.ROOT}/s7", exist_ok=True); json.dump(out, open(f"{C.ROOT}/s7/lockbox.json", "w"), indent=1); print(json.dumps(out, indent=1))
