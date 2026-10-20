"""R07 §6: compute estimate after the pilot (§2.3) and the scratch-equivalence test (§3.3), against the 300 GPU-h cap.
Pretraining: pilot s/step x 60,000 (P-V1; P-V2 scaled by the R06 V2/V1 step-time ratio measured in the §3.3 run or R06).
Fine-tuning: 102 pretrained-arm + 34 scratch-V2 trainings at the §3.3 run's s/step (V1) and R06's V2/V1 ratio, plus
inference, with a 1.5x allowance for A100 / L40S nodes (R06 §4 measured ~1.5x). Writes compute_estimate.json; a
within_budget = false stops the full run (train_array.sbatch refuses to start)."""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import r07_common as R7, r06_common as R6
pil = json.load(open(f"{R7.PRETRAIN}/P-V1_pilot/pretrain_log.json")); eq = json.load(open(f"{R7.CHECKS}/scratch_equiv/result.json"))
r06 = json.load(open(f"{R6.SCORES}/compute_summary.json")); ratio_v2 = r06["by_variant"]["V2"]["s_per_step"] / r06["by_variant"]["V1"]["s_per_step"]
used = dict(pilot_gpu_h=pil["wall_train_s"] / 3600, equiv_gpu_h=eq["s_per_step_this"] * R6.STEPS / 3600 + 10 / 3600)
pre = {"P-V1": pil["s_per_step"] * R7.PRE_STEPS / 3600 * 1.08, "P-V2": pil["s_per_step"] * ratio_v2 * R7.PRE_STEPS / 3600 * 1.08}   # + ~8 % for the periodic held-out evaluations
s_v1 = eq["s_per_step_this"]; infer_h = 0.05 / 60 * 23438 / 16 / 60   # R06: ~5 s per 2,591 windows -> per mean unit
ft = {"P-V1": dict(n=51, gpu_h=51 * (s_v1 * R6.STEPS / 3600 + infer_h) * 1.5), "P-V2": dict(n=51, gpu_h=51 * (s_v1 * ratio_v2 * R6.STEPS / 3600 + infer_h) * 1.5), "V2": dict(n=34, gpu_h=34 * (s_v1 * ratio_v2 * R6.STEPS / 3600 + infer_h) * 1.5)}
total = sum(pre.values()) + sum(v["gpu_h"] for v in ft.values()) + sum(used.values())
rec = dict(pilot=dict(body="P-V1", steps=pil["steps"], s_per_step=pil["s_per_step"], gpu=pil["gpu"], wall_train_s=pil["wall_train_s"], rule=pil.get("pilot_rule")), scratch_equiv=dict(s_per_step=s_v1, gpu=eq["gpu_this_run"], verdict=eq["verdict"]),
           r06_v2_over_v1_step_time=ratio_v2, already_used_gpu_h=used, pretraining_gpu_h=pre, finetuning_gpu_h=ft, gpu_h_estimate=total, budget_gpu_h=R7.BUDGET_GPU_H, within_budget=bool(total <= R7.BUDGET_GPU_H),
           note="pretraining on the pilot's GPU type (H100 requested); fine-tuning with a 1.5x allowance for A100 / L40S nodes (R06 §4)")
json.dump(rec, open(f"{R7.ROOT}/compute_estimate.json", "w"), indent=1); print(json.dumps(rec, indent=1))
print(f"ESTIMATE: {total:.1f} GPU-h vs cap {R7.BUDGET_GPU_H:.0f} -> {'WITHIN BUDGET' if rec['within_budget'] else 'EXCEEDS BUDGET: STOP (HOLD)'}")
