"""R06 full run task list (directive §5.2: 4 variants x 16 folds + 2 extra seeds x 16 folds for V1 + 4 all-unit models
for EW0207 = 100 trainings). One array task = one variant/seed and a list of folds run sequentially (several trainings per
GPU job so each job lasts hours, not minutes). Existing fold outputs are skipped by train_r06.py (the pilot's V1/lu00)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import r06_common as R, r04_common as C4
units = list(C4.UNITS); per_task = {"V0": 4, "V1": 4, "V2": 3, "V3": 6}
lines = []
for v, seeds in (("V0", [R.SEED_MAIN]), ("V1", list(R.SEEDS_V1)), ("V2", [R.SEED_MAIN]), ("V3", [R.SEED_MAIN])):
    for s in seeds:
        folds = units + (["ALL"] if s == R.SEED_MAIN else [])
        k = per_task[v]; chunks = [folds[i:i + k] for i in range(0, len(folds), k)]
        if len(chunks) > 1 and len(chunks[-1]) == 1: last = chunks.pop(); chunks[-1] = chunks[-1] + last      # avoid a one-training task
        for c in chunks: lines.append(f"{v} {s} {','.join(c)}")
open(f"{os.path.dirname(os.path.abspath(__file__))}/tasks.txt", "w").write("\n".join(lines) + "\n")
print(f"{len(lines)} tasks, {sum(len(l.split()[2].split(',')) for l in lines)} trainings"); print("\n".join(lines))
