"""R16 must-hold 3.8 — run a Phase 5 script UNCHANGED in plain rehearsal mode (no mock scaffolding) with the R16 reproduction root
(env R16_ROOT=repro). Usage: R16_ROOT=repro python run_plain_r16.py phase5_build.py --mode rehearsal --pairs "..."."""
import os, runpy, sys
import r16_common as R
if __name__ == "__main__":
    script = sys.argv[1]; assert script in R.P5.SCRIPTS and "phase5" not in sys.argv[2:]; path = os.path.join(R.P5_DIR, script)
    print(f"[run_plain_r16] {script} sha256 {R.P5.sha256_file(path)}; ROOT[rehearsal] = {R.P5.ROOT['rehearsal']}", flush=True); sys.argv = [path] + sys.argv[2:]; sys.path.insert(0, R.P5_DIR); runpy.run_path(path, run_name="__main__")
