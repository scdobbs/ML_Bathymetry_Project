"""R10 Part B — past access through the guard gap. Searches text files (job logs, provenance, scripts, records) of the R04-R09
run directories (scratch and OAK), the repository's directive scripts and the earlier CNN-repo run directories that still
have logs, for the 37 sealed tokens; only files whose OWN paths carry no token are searched (find -not -path), through
external find / grep (the Python guard is installed but not needed: no sealed-named file is ever listed to grep).
Each matching line is recorded with a rough class: 'path' if the token appears inside a path-like string, else 'mention';
'possible read' if a path-class match sits in a job log (.out/.err/.log). Writes logsearch.json and a CSV of matches."""
import json, os, subprocess, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "phase4", "scripts"))
import phase4_common as P4
R9 = P4.R9; OUT = "/scratch/users/scdobbs/grl_review/R10/logsearch"; os.makedirs(OUT, exist_ok=True)
ROOTS = ["/scratch/users/scdobbs/grl_review", f"{P4.R8.OAK_ARCHIVE}", f"{P4.DR}", "/home/users/scdobbs/GAN_Bathymetry_Project/logs", "/scratch/users/scdobbs/jobscripts"] + [f"/scratch/users/scdobbs/{d}" for d in ("dir44", "dir45", "dir47", "dir48", "dir49", "dir50", "dir51", "dir52", "dir53", "dir54", "dir55", "dir56", "dir57", "bathy_diffusion", "bathy_diffusion_v2", "bathy_diffusion_v3", "bathy_diffusion_v4", "smoke_coloc_phase1", "smoke_native_extract", "smoke_native_speed", "step0_bakeoff", "patches_colocated_smoke", "patches_smoke_evalonly", "patches_canon_smoke", "reconciliation_29a")]
EXT = ("out", "err", "log", "txt", "json", "csv", "md", "py", "sbatch", "sh", "yaml", "yml", "toml")
if __name__ == "__main__":
    t0 = time.time(); m0 = P4.C4.load_manifest(); R9.install_guard_r09(m0); toks = R9.sealed_tokens(m0); assert len(toks) == 37
    pat = "\n".join(toks); open(f"{OUT}/tokens.txt", "w").write(pat + "\n")
    rec = dict(tokens=toks, roots=[], n_files_searched=0, matches=[], missing_roots=[])
    for root in ROOTS:
        if not os.path.isdir(root): rec["missing_roots"].append(root); continue
        t1 = time.time(); cmd = ["find", root, "-xdev", "-type", "f"] + sum([["-not", "-path", f"*{t}*"] for t in toks], []) + ["(" ] + sum([["-o", "-name", f"*.{e}"] for e in EXT], [])[1:] + [")"]
        files = [f for f in subprocess.run(cmd, capture_output=True, text=True).stdout.split("\n") if f]
        n_hit = 0; hits = []
        for i in range(0, len(files), 2000):
            chunk = files[i:i + 2000]; r = subprocess.run(["grep", "-I", "-n", "-H", "-F", "-f", f"{OUT}/tokens.txt"] + chunk, capture_output=True, text=True)
            for line in r.stdout.split("\n"):
                if not line: continue
                path, _, rest = line.partition(":"); ln, _, text = rest.partition(":"); tok = [t for t in toks if t in text]
                kind = "path" if any(("/" + t) in text or (t + "/") in text or t in text and ("/" in text) for t in tok) else "mention"
                if kind == "path" and path.endswith((".out", ".err", ".log")): kind = "possible read (path in a job log)"
                hits.append(dict(root=root, file=path, line=int(ln) if ln.isdigit() else ln, tokens=tok, kind=kind, text=text[:240])); n_hit += 1
        rec["roots"].append(dict(root=root, n_files=len(files), n_matching_lines=n_hit, seconds=time.time() - t1)); rec["n_files_searched"] += len(files); rec["matches"] += hits
        print(f"  {root}: {len(files)} files, {n_hit} matching lines ({time.time()-t1:.0f} s)", flush=True)
    json.dump(rec, open(f"{OUT}/logsearch.json", "w"), indent=1)
    import pandas as pd; pd.DataFrame(rec["matches"]).to_csv(f"{OUT}/matches.csv", index=False)
    print(f"files searched {rec['n_files_searched']}, matching lines {len(rec['matches'])}, kinds: {pd.Series([h['kind'] for h in rec['matches']]).value_counts().to_dict() if rec['matches'] else {}} ({time.time()-t0:.0f} s)", flush=True)
