"""R12c must-hold 4 — run a Phase 4 script UNCHANGED in plain rehearsal mode (no mock scaffolding) with the R12c reproduction root
(env R12C_ROOT=repro). Usage: R12C_ROOT=repro python run_plain_r12c.py phase4_build.py --mode rehearsal --pairs "..."."""
import os, runpy, sys
import r12c_common as R
if __name__ == "__main__":
    script = sys.argv[1]; assert script in R.P4.SCRIPTS and "phase4" not in sys.argv[2:]; path = os.path.join(R.P4_DIR, script)
    print(f"[run_plain_r12c] {script} sha256 {R.P4.sha256_file(path)}; ROOT[rehearsal] = {R.P4.ROOT['rehearsal']}", flush=True); sys.argv = [path] + sys.argv[2:]; sys.path.insert(0, R.P4_DIR); runpy.run_path(path, run_name="__main__")
