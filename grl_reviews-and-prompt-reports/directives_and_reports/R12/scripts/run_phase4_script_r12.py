"""R12 Part A: run one Phase 4 script UNCHANGED in rehearsal mode with the R12 rehearsal root.
    python run_phase4_script_r12.py phase4_build.py --mode rehearsal --pairs "..." --rows <harness>
    python run_phase4_script_r12.py phase4_predict.py --mode rehearsal --pairs "..."
r12_common re-points phase4_common.ROOT['rehearsal'] (a module-level dict the scripts read at run time), then the named script is
executed as __main__ from its committed file (sha256 printed first). Nothing in phase4/scripts is edited; phase4 mode is refused here."""
import os, runpy, sys
import r12_common as R12
if __name__ == "__main__":
    script = sys.argv[1]; assert script in R12.P4.SCRIPTS, script; assert "phase4" not in sys.argv[2:], "rehearsal only"
    path = os.path.join(R12.P4_DIR, script); print(f"[run_phase4_script_r12] {script} sha256 {R12.P4.sha256_file(path)}; ROOT[rehearsal] = {R12.P4.ROOT['rehearsal']}", flush=True)
    sys.argv = [path] + sys.argv[2:]; sys.path.insert(0, R12.P4_DIR); runpy.run_path(path, run_name="__main__")
