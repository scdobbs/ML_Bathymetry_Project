"""Phase 4 predictions (pre-registration v2, B3 step 4) with the three frozen candidates on the listed pairs, on CPU:
  s_lr     the tiles_v1.1 s_lr of the build (build_inputs: SD of the valid LR cells of the 256-px window, floored at 0.5 m);
  GBT_ship the R09 all-16 artifact (joblib, sha256-verified) on the build's feature table (bar_v1_1.ALL columns), 10**pred (m);
  P-V1     the three R07 all-unit checkpoints (sha256-verified) run on the build's R06 input arrays with the R06 model code
           (BlockHeadNet, V1: no products, no crop) in float32 on CPU; per seed: 16 block log10 predictions, tile = RMS of
           10**blk; seed-averaged = mean of the three seeds' log10 block predictions (R07 convention).
Environment control (§A5, both modes, before any admitted pair is predicted): EW0207 is predicted from the R10 input arrays
and feature table on OAK (sha256-verified) and compared with the R10 EW0207 predictions on OAK (sha256-verified): s_lr identical,
GBT_ship max |diff| <= 1e-6 m, P-V1 seed-averaged max |d log10| <= 1e-4; the comparison goes to the provenance record and the
script refuses to continue on failure. A no-k pair (§A4) is skipped. Writes <root>/predictions/<pair>.parquet and a record."""
from __future__ import annotations
import json, os, sys, time
import numpy as np, pandas as pd, joblib, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase4_common as P4
import r04_common as C4, r06_common as R6, bar_v1_1 as B
from r06_model import BlockHeadNet

def load_models():
    models = {}
    for s, path in P4.PV1_CKPT.items():
        ck = torch.load(path, map_location="cpu"); cfg = ck["config"]; assert cfg["products"] is False and cfg["crop"] is False, cfg
        mdl = BlockHeadNet(cfg["base_ch"], tuple(cfg["ch_mult"]), 2, 4, products=False, crop=False); mdl.load_state_dict(ck["state_dict"]); mdl.eval(); models[s] = mdl
    return models

def predict_pv1(models, d, workers, batch=32):
    meta = pd.read_parquet(f"{d}/meta.parquet"); X = np.load(f"{d}/x.npy", mmap_mode="r"); LV = np.load(f"{d}/lrv.npy", mmap_mode="r")
    cond = torch.from_numpy(meta[["cond0", "cond1", "cond2", "cond3"]].values.astype(np.float32)); n = len(meta); torch.set_num_threads(max(1, workers)); out = {}
    for s, model in models.items():
        preds = []
        with torch.no_grad():
            for i in range(0, n, batch):
                inp = torch.from_numpy(np.stack([np.stack([np.asarray(X[j]), np.asarray(LV[j]).astype(np.float32)], 0) for j in range(i, min(n, i + batch))], 0))
                p = model(inp, cond[i:i + batch], None); preds.append(p.float().reshape(p.shape[0], -1).numpy())
        out[s] = np.concatenate(preds, 0); assert out[s].shape == (n, R6.NB * R6.NB)
    return meta, out

def assemble(tiles, ft, gbt, meta, pv):
    P = tiles[["tile_id", "pair", "unit", "r0", "c0", "A_fine", "s_lr"]].copy(); P["GBT_ship"] = 10 ** gbt.predict(ft.loc[P.tile_id, B.ALL].values); assert (meta.tile_id.values == P.tile_id.values).all()
    for s, arr in pv.items():
        P[f"P-V1_s{s}"] = np.sqrt(np.mean(10 ** (2 * arr), axis=1))
        for j in range(R6.NB * R6.NB): P[f"P-V1_s{s}_blk{j}"] = arr[:, j]
    M = np.mean([pv[s] for s in P4.PV1_CKPT], axis=0); P["P-V1"] = np.sqrt(np.mean(10 ** (2 * M), axis=1)); return P

def environment_control(gbt, models, workers):
    """§A5: EW0207 from the R10 OAK inputs against the R10 OAK predictions. Raises Refusal on failure; returns the comparison."""
    ew = P4.R8.EW_PAIR; sub = "R10/rehearsal"; d = f"{P4.R10_OAK}/rehearsal/inputs/{ew}"; ver = {}
    for rel in (f"inputs/{ew}/x.npy", f"inputs/{ew}/lrv.npy", f"inputs/{ew}/meta.parquet", f"build/{ew}/features.parquet", f"build/{ew}/tiles_v1_1.parquet", f"predictions/{ew}.parquet"):
        ver[rel] = P4.R8.oak_verify(sub, rel, f"{P4.R10_OAK}/rehearsal/{rel}")["sha256"]
    tiles = pd.read_parquet(f"{P4.R10_OAK}/rehearsal/build/{ew}/tiles_v1_1.parquet"); feats = pd.read_parquet(f"{P4.R10_OAK}/rehearsal/build/{ew}/features.parquet"); ft = feats[feats.block_id < 0].set_index("tile_id")
    meta, pv = predict_pv1(models, d, workers); P = assemble(tiles, ft, gbt, meta, pv); R = pd.read_parquet(f"{P4.R10_OAK}/rehearsal/predictions/{ew}.parquet").set_index("tile_id").reindex(P.tile_id)
    cmp = dict(s_lr_identical=bool(np.array_equal(P.s_lr.values, R.s_lr.values)), gbt_max_abs_diff_m=float(np.max(np.abs(P.GBT_ship.values - R.GBT_ship.values))),
               pv1_seedavg_max_abs_diff_log10=float(np.max(np.abs(np.log10(P["P-V1"].values) - np.log10(R["P-V1"].values)))), pv1_per_seed_max_abs_diff_log10={str(s): float(np.max(np.abs(np.log10(P[f"P-V1_s{s}"].values) - np.log10(R[f"P-V1_s{s}"].values)))) for s in P4.PV1_CKPT},
               tolerances=P4.CTRL_TOL, verified=ver, node=os.uname().nodename, torch=torch.__version__, threads=torch.get_num_threads(), cpu=open("/proc/cpuinfo").read().split("model name")[1].split("\n")[0].split(":")[1].strip() if os.path.exists("/proc/cpuinfo") else None)
    cmp["passed"] = bool(cmp["s_lr_identical"] and cmp["gbt_max_abs_diff_m"] <= P4.CTRL_TOL["gbt_m"] and cmp["pv1_seedavg_max_abs_diff_log10"] <= P4.CTRL_TOL["pv1_log10"])
    if not cmp["passed"]: raise P4.Refusal(f"LOCK environment control: EW0207 not reproduced: {json.dumps({k: cmp[k] for k in ('s_lr_identical', 'gbt_max_abs_diff_m', 'pv1_seedavg_max_abs_diff_log10')})}")
    return cmp

if __name__ == "__main__":
    ap = P4.argparser("Phase 4 predictions"); ap.add_argument("--calls", default=None, help="phase4 mode: the committed blind-calls file (lock)"); ap.add_argument("--control_only", action="store_true", help="run the §A5 environment control and stop"); a = ap.parse_args(); pairs = P4.split_pairs(a.pairs)
    m, root, rec = P4.start("phase4_predict.py", a.mode, pairs, calls_path=a.calls); t0 = time.time(); os.makedirs(f"{root}/predictions", exist_ok=True)
    gbt = joblib.load(P4.GBT_ARTIFACT); models = load_models()
    ctrl = environment_control(gbt, models, a.workers); ctrl["seconds"] = time.time() - t0
    json.dump(dict(script="phase4_predict.py", mode=a.mode, environment_control=ctrl, time=rec["time"], slurm_job=rec["slurm_job"]), open(f"{root}/provenance/environment_control_{rec['time'].replace(':', '')}_{os.getpid()}.json", "w"), indent=1, default=str)
    print(f"environment control PASSED on {ctrl['node']} ({ctrl['cpu']}): s_lr identical {ctrl['s_lr_identical']}, GBT max |d| {ctrl['gbt_max_abs_diff_m']:.2e} m, P-V1 seed-avg max |d log10| {ctrl['pv1_seedavg_max_abs_diff_log10']:.2e} ({ctrl['seconds']:.0f} s)", flush=True)
    if a.control_only: sys.exit(0)
    recs = {}
    for pid in pairs:
        t1 = time.time(); st = json.load(open(f"{root}/build/{pid}/pair_stats.json"))
        if st.get("status") == P4.NO_K: recs[pid] = dict(status=P4.NO_K); print(f"  {pid}: {P4.NO_K}, not predicted", flush=True); continue
        tiles = pd.read_parquet(f"{root}/build/{pid}/tiles_v1_1.parquet"); feats = pd.read_parquet(f"{root}/build/{pid}/features.parquet"); ft = feats[feats.block_id < 0].set_index("tile_id")
        meta, pv = predict_pv1(models, f"{root}/inputs/{pid}", a.workers); P = assemble(tiles, ft, gbt, meta, pv)
        path = f"{root}/predictions/{pid}.parquet"; P.to_parquet(path); recs[pid] = dict(path=path, sha256=P4.sha256_file(path), n_tiles=int(len(P)), seconds=time.time() - t1)
        print(f"  predicted {pid}: {len(P)} tiles ({time.time()-t1:.0f} s)", flush=True)
    json.dump(dict(mode=a.mode, outputs=recs, environment_control=ctrl, torch=torch.__version__, threads=torch.get_num_threads()), open(f"{root}/predictions/predict_record_{int(time.time())}.json", "w"), indent=1, default=str); print(f"done ({time.time()-t0:.0f} s)", flush=True)
