"""R03 §1.1 — run preprocessing/extract_clean_patches.py UNCHANGED on the Cal DIG (18) and expansion (7) pairs.
Lockbox guard installed by r03_common (any open of a lockbox raster raises). Pairs without a sweep k are STOPPED and
listed (directive §1.1: do not infer)."""
import sys, json, os
sys.path.insert(0, "/scratch/users/scdobbs/grl_review/R03"); sys.path.insert(0, "/home/users/scdobbs/GAN_Bathymetry_Project")
import r03_common as C
from preprocessing import extract_clean_patches as ecp
from preprocessing.colocated_utils import load_k_map
pairs = C.assert_no_lockbox([p for d in (C.CALDIG, C.EXPANSION7) for ps in d.values() for p in ps])
kmap = load_k_map(C.DATA)
missing = {p: kmap.get(p) for p in pairs if kmap.get(p) is None}
assert not missing, f"pairs lack k in the sweep — refusing to infer: {missing}"
json.dump(dict(pairs=pairs, stopped_no_k=C.STOPPED_NO_K, lockbox=list(C.LOCKBOX)), open(f"{C.ROOT}/expanded_pair_list.json", "w"), indent=1)
print(f"{len(pairs)} pairs; stopped (no k): {list(C.STOPPED_NO_K)}; lockbox excluded: {C.LOCKBOX}", flush=True)
sys.argv = ["extract_clean_patches",
            "--split_path", "/scratch/users/scdobbs/grl_review/R01/patches_restored/patches_native_phase2/split_assignment.parquet",
            "--out_dir", f"{C.ROOT}/patches_clean_expanded", "--pairs", ",".join(pairs),
            "--hr_px", "256", "--train_stride_frac", "0.25", "--eval_stride_frac", "0.5", "--min_valid", "0.50",
            "--flat_threshold_m", "0.05", "--s_floor", "0.5", "--leak_test_tiles", "100", "--workers", os.environ.get("SLURM_CPUS_PER_TASK", "8")]
ecp.main()
