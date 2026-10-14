"""R16c §2.2 (= R16 Part C2 step 3) — the mock blind-calls files, keyed by Q-code exactly like Steve's file (columns code, relief, criteria,
dominant, ab_whole_survey, note; no pair column), committed to the scratch ledger (sha256 appended) before any prediction.
Only the list of page codes is read from the mock key (never the pairs). Two files:
  calls_mock.csv           every page relief-dominated (one flagged with criterion a): the C1 set holds every scorable mock site;
  calls_mock_artifact.csv  the same with Q02 dominant = artifact: the C1 set loses one site (3 -> 2), so the primary reads
                           "not testable" (must-hold 3.5, second statement).
Written under <ROOT>/calls/. Development data only."""
import hashlib, json, os
import r16c_common as R
ROOT = R.ROOT; OUT = f"{ROOT}/calls"
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
ROWS = {"Q01": ("yes", "none", "relief", "no"), "Q02": ("weak", "a", "relief", "no"), "Q03": ("yes", "none", "relief", "no")}
if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); codes = [o["code"] for o in json.load(open(f"{ROOT}/figures/key/figset_A_key.json"))["order"]]; assert codes == sorted(ROWS), codes
    out = {}
    for name, art in (("calls_mock.csv", None), ("calls_mock_artifact.csv", "Q02")):
        lines = ["code,relief,criteria,dominant,ab_whole_survey,note"]
        for c in codes:
            rel, cr, dom, ab = ROWS[c]; dom = "artifact" if c == art else dom
            lines.append(f"{c},{rel},{cr},{dom},{ab},R16 mock call (rehearsal scaffolding)")
        p = f"{OUT}/{name}"; open(p, "w").write("\n".join(lines) + "\n"); h = sha(p); out[name] = dict(path=p, sha256=h, codes=codes)
        with open(R.MOCK_LEDGER, "a") as f: f.write(f"| m-calls | mock calls {p} | {h} |\n")
    json.dump(out, open(f"{OUT}/record.json", "w"), indent=1); print(json.dumps(out, indent=1))
