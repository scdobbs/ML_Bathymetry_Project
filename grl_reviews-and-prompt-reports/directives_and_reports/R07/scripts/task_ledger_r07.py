"""R07 §0.1 item 4 — ledger of every Slurm job / array task of this directive from sacct (state, elapsed, exit code, GPU
type from the training logs), with every non-COMPLETED or requeued entry listed with its reason. Reads the job ids recorded
in jobids_stage*.txt plus any extra ids given on the command line. Writes task_ledger.json and task_ledger.csv."""
import glob, json, os, subprocess, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import r07_common as R7
ids = set(sys.argv[1:])
for f in glob.glob(f"{R7.ROOT}/jobids_*.txt"):
    for tok in open(f).read().replace(";", " ").split():
        if tok.isdigit(): ids.add(tok)
for f in ("46903869", "46905122", "46904378"): ids.add(f)      # stage, equiv+cache, scratch-equivalence
out = subprocess.run(["sacct", "-j", ",".join(sorted(ids)), "-X", "-P", "-o", "JobID,JobName,State,Elapsed,ExitCode,NodeList,Partition,Submit,Start,End,ReqTRES"], capture_output=True, text=True).stdout
rows = [dict(zip(out.splitlines()[0].split("|"), l.split("|"))) for l in out.splitlines()[1:] if l]
df = pd.DataFrame(rows); df.to_csv(f"{R7.ROOT}/task_ledger.csv", index=False)
led = []
for st, g in df.groupby("State"):
    if st != "COMPLETED": led.append(dict(stage="slurm", item=f"jobs in state {st}", count=len(g), reason="; ".join(f"{r.JobID} {r.JobName} exit {r.ExitCode}" for r in g.itertuples())[:600]))
# requeues: sacct shows a requeued task's earlier attempt only with --duplicates
dup = subprocess.run(["sacct", "-j", ",".join(sorted(ids)), "-X", "-P", "--duplicates", "-o", "JobID,State"], capture_output=True, text=True).stdout
rq = [l for l in dup.splitlines()[1:] if "REQUEUED" in l or "NODE_FAIL" in l or "PREEMPTED" in l]
led.append(dict(stage="slurm", item="requeued / node-fail / preempted attempts", count=len(rq), reason="; ".join(rq)[:600] if rq else "none"))
json.dump(led, open(f"{R7.ROOT}/task_ledger.json", "w"), indent=1)
print(df[["JobID", "JobName", "State", "Elapsed", "ExitCode", "NodeList"]].to_string()); print(json.dumps(led, indent=1))
