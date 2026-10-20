"""Phase 5 predictions (pre-registration v3, step 4): a parameterised copy of phase4_predict.py's driver. The three frozen
candidates, the P-V1 loader / CPU float32 inference, the tile assembly and the EW0207 environment control are the pinned
phase4_predict functions, IMPORTED unchanged (R16 §B8): s_lr from the build; GBT_ship = the R09 all-16 artifact; P-V1 = the three
R07 all-unit checkpoints, seed-averaged in log10. The EW0207 control runs first (both modes) and the script refuses to continue
on failure. Changes from Phase 4: phase5_common.start (Phase 5 locks incl. the code-keyed calls file in the ledger, guard,
allow-list); a not-testable site (§B1) and a no-k pair are skipped. Writes <root>/predictions/<pair>.parquet and a record."""
from __future__ import annotations
import json, os, sys, time
import pandas as pd, joblib, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase5_common as P5
import phase4_predict as PP4                                     # the pinned Phase 4 prediction functions
import r04_common as C4

if __name__ == "__main__":
    ap = P5.argparser("Phase 5 predictions"); ap.add_argument("--calls", default=None, help="phase5 mode: the committed code-keyed blind-calls file (lock)"); ap.add_argument("--control_only", action="store_true", help="run the EW0207 environment control and stop"); a = ap.parse_args(); pairs = P5.split_pairs(a.pairs)
    m, root, rec = P5.start("phase5_predict.py", a.mode, pairs, calls_path=a.calls); t0 = time.time(); os.makedirs(f"{root}/predictions", exist_ok=True)
    gbt = joblib.load(P5.P4.GBT_ARTIFACT); models = PP4.load_models()
    ctrl = PP4.environment_control(gbt, models, a.workers); ctrl["seconds"] = time.time() - t0
    json.dump(dict(script="phase5_predict.py", mode=a.mode, environment_control=ctrl, time=rec["time"], slurm_job=rec["slurm_job"]), open(f"{root}/provenance/environment_control_{rec['time'].replace(':', '')}_{os.getpid()}.json", "w"), indent=1, default=str)
    print(f"environment control PASSED on {ctrl['node']} ({ctrl['cpu']}): s_lr identical {ctrl['s_lr_identical']}, GBT max |d| {ctrl['gbt_max_abs_diff_m']:.2e} m, P-V1 seed-avg max |d log10| {ctrl['pv1_seedavg_max_abs_diff_log10']:.2e} ({ctrl['seconds']:.0f} s)", flush=True)
    if a.control_only: sys.exit(0)
    recs = {}
    for pid in pairs:
        t1 = time.time(); rs = P5.row_status(C4.pair_row(m, pid))
        if rs.startswith(P5.NOT_TESTABLE): recs[pid] = dict(status=rs); print(f"  {pid}: {rs}, not predicted", flush=True); continue
        st = json.load(open(f"{root}/build/{pid}/pair_stats.json"))
        if st.get("status") == P5.P4.NO_K: recs[pid] = dict(status=P5.P4.NO_K); print(f"  {pid}: {P5.P4.NO_K}, not predicted", flush=True); continue
        tiles = pd.read_parquet(f"{root}/build/{pid}/tiles_v1_1.parquet"); feats = pd.read_parquet(f"{root}/build/{pid}/features.parquet"); ft = feats[feats.block_id < 0].set_index("tile_id")
        meta, pv = PP4.predict_pv1(models, f"{root}/inputs/{pid}", a.workers); P = PP4.assemble(tiles, ft, gbt, meta, pv)
        path = f"{root}/predictions/{pid}.parquet"; P.to_parquet(path); recs[pid] = dict(path=path, sha256=P5.sha256_file(path), n_tiles=int(len(P)), seconds=time.time() - t1)
        print(f"  predicted {pid}: {len(P)} tiles ({time.time()-t1:.0f} s)", flush=True)
    json.dump(dict(mode=a.mode, outputs=recs, environment_control=ctrl, torch=torch.__version__, threads=torch.get_num_threads()), open(f"{root}/predictions/predict_record_{int(time.time())}.json", "w"), indent=1, default=str); print(f"done ({time.time()-t0:.0f} s)", flush=True)
