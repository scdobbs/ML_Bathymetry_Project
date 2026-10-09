"""R07 §3 task lists for the fine-tuning arrays (one array task = one arm / seed and a list of folds run sequentially,
several trainings per GPU job). tasks_scratch_v2.txt: R06 V2 from scratch, seeds 20261003 / 20261004, 16 folds + ALL
each (34). tasks_pretrained.txt: P-V1 and P-V2, three seeds, 16 folds + ALL each (51 + 51). train_r07.py skips folds
whose pred.parquet exists, so a requeued task resumes."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import r07_common as R7, r04_common as C4
units = list(C4.UNITS); folds = units + ["ALL"]; per_task = {"V2": 3, "P-V1": 4, "P-V2": 3}
def chunks(arm, seed):
    k = per_task[arm]; c = [folds[i:i + k] for i in range(0, len(folds), k)]
    if len(c) > 1 and len(c[-1]) == 1: last = c.pop(); c[-1] = c[-1] + last
    return [f"{arm} {seed} {','.join(x)}" for x in c]
A = [l for s in (20261003, 20261004) for l in chunks("V2", s)]
B = [l for arm in ("P-V1", "P-V2") for s in R7.SEEDS for l in chunks(arm, s)]
here = os.path.dirname(os.path.abspath(__file__))
open(f"{here}/tasks_scratch_v2.txt", "w").write("\n".join(A) + "\n"); open(f"{here}/tasks_pretrained.txt", "w").write("\n".join(B) + "\n")
n = lambda L: sum(len(l.split()[2].split(",")) for l in L)
print(f"scratch V2: {len(A)} tasks, {n(A)} trainings\npretrained: {len(B)} tasks, {n(B)} trainings"); print("\n".join(A)); print("\n".join(B))
