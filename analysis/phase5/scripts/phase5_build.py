"""Phase 5 build (pre-registration v3, step 2): a parameterised copy of phase4_build.py's driver. The per-pair build is the
pinned phase4_build.build_pair, IMPORTED unchanged (R16 §B8: R04 windows / fields / targets_v1, R05 tiles_v1.1, R04 features,
R09 targets_v1_int at D1, R06 P-V1 inputs, the B4 amplitude statistic per view before any prediction; product features used or
masked by r04_common.ship_usable; k provenance from the acquisition run record). Changes from Phase 4 (§B1, §B4):
  - a supplement row with eligible = False is "not testable (<eligibility_reason>)": nothing is opened or built and only its
    status is written (in build_summary_*.json and eligibility_status.json under <root>/build/, paths that hold no site id);
  - phase5_common.start (Phase 5 locks, guard, allow-list of the eligible pairs) and ROOT[phase5] = grl_review_archive/phase5."""
from __future__ import annotations
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase5_common as P5
import phase4_build as PB4                                       # the pinned Phase 4 build (build_pair, block_stats, require_row)
import r04_common as C4

def build_pair(m, pid, root, workers, log=print, acq_rec=None):
    """Eligibility first (§B1); an eligible row goes to the pinned phase4_build.build_pair unchanged."""
    st = P5.row_status(C4.pair_row(m, pid))
    if st.startswith(P5.NOT_TESTABLE):
        log(f"  {pid}: {st}; nothing opened or built", flush=True); return dict(pair=pid, status=st, note="spec §3.3: failed eligibility at step 1; no page, prediction or score")
    return PB4.build_pair(m, pid, root, workers, log=log, acq_rec=acq_rec)

if __name__ == "__main__":
    ap = P5.argparser("Phase 5 build"); ap.add_argument("--rows", default=None, help="rehearsal only: parquet whose rows replace the listed pairs' manifest rows (harness supplement)"); ap.add_argument("--out_root", default=None, help="rehearsal only: output root instead of ROOT[rehearsal]"); a = ap.parse_args(); pairs = P5.split_pairs(a.pairs)
    if a.mode == "phase5" and (a.rows or a.out_root): raise P5.Refusal("--rows and --out_root are rehearsal-only (phase5 mode takes the rows from the supplement and writes to ROOT[phase5])")
    m, root, rec = P5.start("phase5_build.py", a.mode, pairs); hrec = None
    if a.rows: m, hrec = P5.harness_rows(m, a.rows, pairs)
    if a.out_root: root = a.out_root; os.makedirs(f"{root}/provenance", exist_ok=True); json.dump(rec | dict(out_root=root, harness_rows=hrec), open(f"{root}/provenance/phase5_build.py_{rec['time'].replace(':', '')}_{os.getpid()}.json", "w"), indent=1, default=str)
    os.makedirs(f"{root}/build", exist_ok=True)
    allst = [build_pair(m, pid, root, a.workers, acq_rec=rec.get("acq_run_record")) for pid in pairs]
    elig = {s["pair"]: s["status"] for s in allst}; json.dump(dict(mode=a.mode, status=elig, time=rec["time"]), open(f"{root}/build/eligibility_status.json", "w"), indent=1)
    json.dump(dict(mode=a.mode, pairs=allst, harness_rows=hrec), open(f"{root}/build/build_summary_{int(time.time())}.json", "w"), indent=1, default=float); print("done", flush=True)
