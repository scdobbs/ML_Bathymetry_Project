"""R06 §5.2-5.3: extrapolate the pilot (V1 / lu00 full schedule + 200-step probes of V0, V2, V3) to the 100 trainings
of the round; stop rule against the 400 GPU-hour cap. Writes compute_estimate.json."""
import glob, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import r06_common as R
log = json.load(open(f"{R.MODELS}/V1/seed{R.SEED_MAIN}/fold_lu00/train_log.json"))
probe = {}
for f in sorted(glob.glob(f"{R.LOGS}/r06_pilot_*.out"), key=os.path.getmtime)[-1:]:
    for line in open(f):
        if line.startswith("TIMING"): d = json.loads(line[7:]); probe[d["variant"]] = d["s_per_step"]
probe["V1"] = log["s_per_step"]
n_train = {"V0": 16, "V1": 16 * 3, "V2": 16, "V3": 16}; n_all = {"V0": 1, "V1": 1, "V2": 1, "V3": 1}
infer_h = log["wall_infer_s"] / 3600 * (log["n_held_windows"] and 23438 / 16 / log["n_held_windows"])   # scale to the mean unit size
est = {v: dict(s_per_step=probe.get(v), trainings=n_train[v] + n_all[v], gpu_h_per_training=(probe.get(v, float("nan")) * R.STEPS / 3600 + infer_h)) for v in n_train}
for v in est: est[v]["gpu_h"] = est[v]["gpu_h_per_training"] * est[v]["trainings"]
total = sum(e["gpu_h"] for e in est.values()); n_total = sum(e["trainings"] for e in est.values())
rec = dict(pilot=dict(variant="V1", fold="lu00", seed=R.SEED_MAIN, wall_train_s=log["wall_train_s"], wall_infer_s=log["wall_infer_s"], wall_total_s=log["wall_total_s"], s_per_step=log["s_per_step"], gpu=log["gpu"], n_train_windows=log["n_train_windows"]),
           probes_s_per_step=probe, per_variant=est, n_trainings=n_total, gpu_h_estimate=total, budget_gpu_h=R.BUDGET_GPU_H, within_budget=bool(total <= R.BUDGET_GPU_H),
           note="estimate on the pilot GPU; A100 nodes, if used, are expected to be ~1.5-2x slower per step")
json.dump(rec, open(f"{R.ROOT}/compute_estimate.json", "w"), indent=1)
print(json.dumps(rec, indent=1)); print(f"ESTIMATE: {n_total} trainings, {total:.1f} GPU-h vs cap {R.BUDGET_GPU_H:.0f} -> {'WITHIN BUDGET' if rec['within_budget'] else 'EXCEEDS BUDGET: STOP (HOLD)'}")
