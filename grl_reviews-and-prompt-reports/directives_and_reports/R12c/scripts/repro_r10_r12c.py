"""R12c must-hold 4 — the R10 reproduction check with the edited phase4_common.py: the plain rehearsal builds of KM0923, AR26 and EW0207
(R12C_ROOT=repro) against the FROZEN artifacts, through R11's compare_r11.a1 unchanged (tiles_v1.1 keys / r0 / c0 / s_lr / A_fine;
F_hr / joint_v1; features within 1e-6 with the NaN pattern; interior flag at D1; A_fine_int at D1). Writes R12c/tables/repro_r10.json."""
import json, os, sys
os.environ["R12C_ROOT"] = "repro"
import r12c_common as R
sys.path.insert(0, f"{R.DR}/R11/scripts"); import compare_r11 as C
if __name__ == "__main__":
    assert C.ROOT == R.ROOTS["repro"], C.ROOT; out = C.a1(R.REPRO_PAIRS); out["_all_identical"] = bool(all(v["all_identical"] for v in out.values())); out["_node"] = os.uname().nodename; out["_cpu"] = C.cpu()
    os.makedirs(f"{R.DR}/R12c/tables", exist_ok=True); json.dump(out, open(f"{R.DR}/R12c/tables/repro_r10.json", "w"), indent=1, default=float); print("R10 reproduction all identical:", out["_all_identical"])
