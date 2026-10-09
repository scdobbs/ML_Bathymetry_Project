"""R07 §3.3 — scratch-equivalence test (required before any fine-tuning). Runs the R07 training code with init = none on
V1 / fold lu00 / seed 20261002 on the GPU type R06 used for that fold (H100 80GB HBM3, from its train_log.json) and
compares the block predictions with R06's (and the checkpoint weights). Expected: identical. Writes
checks/scratch_equiv/result.json; a non-identical result is reported as a HOLD (max |diff| recorded) and no fine-tuning starts."""
from __future__ import annotations
import json, os, sys, time
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r07_common as R7, r06_common as R6
import train_r07 as T

if __name__ == "__main__":
    R7.start("scratch_equiv_r07", check_corpus_files=False)
    gpu = torch.cuda.get_device_name(0); out = f"{R7.CHECKS}/scratch_equiv/V1/seed{R7.EQUIV_SEED}/fold_{R7.EQUIV_FOLD}"; os.makedirs(out, exist_ok=True)
    r06_dir = f"{R7.R06_MODELS}/V1/seed{R7.EQUIV_SEED}/fold_{R7.EQUIV_FOLD}"; rel = f"V1/seed{R7.EQUIV_SEED}/fold_{R7.EQUIV_FOLD}"
    hashes = {f: R7.verify_r06_file(f"{rel}/{f}") for f in ("pred.parquet", "ckpt.pt", "train_log.json")}
    r06_log = json.load(open(f"{r06_dir}/train_log.json")); assert r06_log["gpu"] == R7.EQUIV_GPU, r06_log["gpu"]
    if gpu != R7.EQUIV_GPU: raise SystemExit(f"STOP: this job is on {gpu}, R06 ran this fold on {R7.EQUIV_GPU}; resubmit with the H100 constraint")
    t0 = time.time(); model, rec = T.train("V1", R7.EQUIV_FOLD, R7.EQUIV_SEED, R6.STEPS, 8, out, init=None)
    pred = T.predict(model, "V1", rec and list(T.C4.UNITS[R7.EQUIV_FOLD]), workers=4); pred.to_parquet(f"{out}/pred.parquet")
    rec.update(arm="V1", wall_total_s=time.time() - t0, code_hash=R7.code_hash(), r06_training_code_hash=R6.code_hash(), git_head=R7.git_head()); json.dump(rec, open(f"{out}/train_log.json", "w"), indent=1)
    ref = pd.read_parquet(f"{r06_dir}/pred.parquet")
    assert list(ref.tile_id) == list(pred.tile_id), "tile order differs"
    cols = [f"blk{j}" for j in range(16)] + ["tile_pred_m"]
    A = ref[cols].values.astype(np.float64); B = pred[cols].values.astype(np.float64)
    max_abs = float(np.max(np.abs(A - B))); identical = bool(np.array_equal(ref[cols].values, pred[cols].values))
    sd_ref = torch.load(f"{r06_dir}/ckpt.pt", map_location="cpu")["state_dict"]; sd_new = model.state_dict()
    w_max = max(float((sd_ref[k].float().cpu() - sd_new[k].float().cpu()).abs().max()) for k in sd_ref)
    w_identical = all(torch.equal(sd_ref[k].cpu(), sd_new[k].cpu()) for k in sd_ref)
    res = dict(fold=R7.EQUIV_FOLD, seed=R7.EQUIV_SEED, gpu_this_run=gpu, gpu_r06=r06_log["gpu"], host_this_run=os.uname().nodename, host_r06=r06_log["host"],
               n_windows=len(pred), predictions_identical=identical, max_abs_diff_log10_blocks_and_tile_m=max_abs,
               max_abs_diff_blocks_log10=float(np.max(np.abs(A[:, :16] - B[:, :16]))), max_abs_diff_tile_m=float(np.max(np.abs(A[:, 16] - B[:, 16]))),
               weights_identical=w_identical, weights_max_abs_diff=w_max, final_loss_100_this=rec["final_loss_100"], final_loss_100_r06=r06_log["final_loss_100"],
               n_train_windows_this=rec["n_train_windows"], n_train_windows_r06=r06_log["n_train_windows"], s_per_step_this=rec["s_per_step"], s_per_step_r06=r06_log["s_per_step"],
               r06_files_verified_against_oak=hashes, torch_this=torch.__version__, torch_r06=r06_log["torch"], verdict=("IDENTICAL" if identical else "NOT IDENTICAL -> HOLD"))
    json.dump(res, open(f"{R7.CHECKS}/scratch_equiv/result.json", "w"), indent=1); print(json.dumps(res, indent=1))
    print(f"SCRATCH-EQUIVALENCE: {res['verdict']} (max |diff| predictions {max_abs:.3e}, weights {w_max:.3e})", flush=True)
