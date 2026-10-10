"""Phase 4 predictions (pre-registration v2, B3 step 4) with the three frozen candidates on the listed pairs, on CPU:
  s_lr     the tiles_v1.1 s_lr of the build (build_inputs: SD of the valid LR cells of the 256-px window, floored at 0.5 m);
  GBT_ship the R09 all-16 artifact (joblib, sha256-verified) on the build's feature table (bar_v1_1.ALL columns), 10**pred (m);
  P-V1     the three R07 all-unit checkpoints (sha256-verified) run on the build's R06 input arrays with the R06 model code
           (BlockHeadNet, V1: no products, no crop) in float32 on CPU; per seed: 16 block log10 predictions, tile = RMS of
           10**blk; seed-averaged = mean of the three seeds' log10 block predictions (R07 convention).
Writes <root>/predictions/<pair>.parquet and a record with the sha256 of each output."""
from __future__ import annotations
import json, os, sys, time
import numpy as np, pandas as pd, joblib, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase4_common as P4
import r04_common as C4, r06_common as R6, bar_v1_1 as B
from r06_model import BlockHeadNet

def predict_pv1(root, pid, workers, batch=32):
    d = f"{root}/inputs/{pid}"; meta = pd.read_parquet(f"{d}/meta.parquet"); X = np.load(f"{d}/x.npy", mmap_mode="r"); LV = np.load(f"{d}/lrv.npy", mmap_mode="r")
    cond = torch.from_numpy(meta[["cond0", "cond1", "cond2", "cond3"]].values.astype(np.float32)); n = len(meta); torch.set_num_threads(max(1, workers)); out = {}
    for s, path in P4.PV1_CKPT.items():
        ck = torch.load(path, map_location="cpu"); cfg = ck["config"]; assert cfg["products"] is False and cfg["crop"] is False, cfg
        model = BlockHeadNet(cfg["base_ch"], tuple(cfg["ch_mult"]), 2, 4, products=False, crop=False); model.load_state_dict(ck["state_dict"]); model.eval(); preds = []
        with torch.no_grad():
            for i in range(0, n, batch):
                inp = torch.from_numpy(np.stack([np.stack([np.asarray(X[j]), np.asarray(LV[j]).astype(np.float32)], 0) for j in range(i, min(n, i + batch))], 0))
                p = model(inp, cond[i:i + batch], None); preds.append(p.float().reshape(p.shape[0], -1).numpy())
        out[s] = np.concatenate(preds, 0); assert out[s].shape == (n, R6.NB * R6.NB)
    return meta[["tile_id", "unit", "pair"]].copy(), out

if __name__ == "__main__":
    ap = P4.argparser("Phase 4 predictions"); ap.add_argument("--calls", default=None, help="phase4 mode: the committed blind-calls file (lock)"); a = ap.parse_args(); pairs = P4.split_pairs(a.pairs)
    m, root, rec = P4.start("phase4_predict.py", a.mode, pairs, calls_path=a.calls); t0 = time.time(); os.makedirs(f"{root}/predictions", exist_ok=True)
    gbt = joblib.load(P4.GBT_ARTIFACT); recs = {}
    for pid in pairs:
        t1 = time.time(); tiles = pd.read_parquet(f"{root}/build/{pid}/tiles_v1_1.parquet"); feats = pd.read_parquet(f"{root}/build/{pid}/features.parquet"); ft = feats[feats.block_id < 0].set_index("tile_id")
        P = tiles[["tile_id", "pair", "unit", "r0", "c0", "A_fine", "s_lr"]].copy(); P["GBT_ship"] = 10 ** gbt.predict(ft.loc[P.tile_id, B.ALL].values)
        meta, pv = predict_pv1(root, pid, a.workers); assert (meta.tile_id.values == P.tile_id.values).all()
        for s, arr in pv.items():
            P[f"P-V1_s{s}"] = np.sqrt(np.mean(10 ** (2 * arr), axis=1))
            for j in range(R6.NB * R6.NB): P[f"P-V1_s{s}_blk{j}"] = arr[:, j]
        M = np.mean([pv[s] for s in P4.PV1_CKPT], axis=0); P["P-V1"] = np.sqrt(np.mean(10 ** (2 * M), axis=1))
        path = f"{root}/predictions/{pid}.parquet"; P.to_parquet(path); recs[pid] = dict(path=path, sha256=P4.sha256_file(path), n_tiles=int(len(P)), seconds=time.time() - t1)
        print(f"  predicted {pid}: {len(P)} tiles ({time.time()-t1:.0f} s)", flush=True)
    json.dump(dict(mode=a.mode, outputs=recs, torch=torch.__version__, threads=torch.get_num_threads()), open(f"{root}/predictions/predict_record_{int(time.time())}.json", "w"), indent=1); print(f"done ({time.time()-t0:.0f} s)", flush=True)
