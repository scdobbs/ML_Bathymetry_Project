"""R16c (= R16 must-hold 3.8) — the R10 reproduction with the Phase 5 modules: the plain rehearsal builds of KM0923, AR26 and EW0207 by
phase5_build.py (R16C_ROOT=repro) against the FROZEN artifacts, through R11's compare_r11.a1 unchanged (tiles_v1.1 keys / r0 / c0 /
s_lr / A_fine; F_hr / joint_v1; features within 1e-6 with the NaN pattern; interior flag at D1; A_fine_int at D1). Writes R16c/tables/repro_r10.json."""
import json, os, sys
os.environ["R16C_ROOT"] = "repro"
import r16c_common as R
sys.path.insert(0, f"{R.DR}/R11/scripts"); import compare_r11 as C
if __name__ == "__main__":
    assert C.ROOT == R.ROOT and R.WHICH.startswith("repro"), C.ROOT; out = C.a1(R.REPRO_PAIRS); out["_all_identical"] = bool(all(v["all_identical"] for v in out.values())); out["_node"] = os.uname().nodename; out["_cpu"] = C.cpu()
    os.makedirs(R.TABLES, exist_ok=True); json.dump(out, open(f"{R.TABLES}/repro_r10.json", "w"), indent=1, default=float); print("R10 reproduction all identical:", out["_all_identical"])
