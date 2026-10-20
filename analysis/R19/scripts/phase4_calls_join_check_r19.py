"""R19 §2: Phase 4 calls join check (Phase 4 is unblinded). Compares Steve's committed code-keyed calls (R13/phase4_calls.csv, ledger row 40)
with the joined copy the scorer read (grl_review_archive/phase4/score_inputs/phase4_calls_joined.csv, ledger row 41): every value of every code,
and the joined `pair` column against the figures key. Writes R19/tables/phase4_calls_join_check.json. Standard library only."""
import csv, hashlib, json, os
DR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); P4 = "/oak/stanford/groups/hilley/auv_ship_colocated_bathy/grl_review_archive/phase4"
A, B, K = f"{DR}/R13/phase4_calls.csv", f"{P4}/score_inputs/phase4_calls_joined.csv", f"{P4}/figures/key/figset_A_key.json"
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
a = list(csv.DictReader(open(A, newline=""))); b = {r["code"]: r for r in csv.DictReader(open(B, newline=""))}
key = {o["code"]: o["pair"] for o in json.load(open(K))["order"]}; cols = ["code", "relief", "criteria", "dominant", "ab_whole_survey", "note"]
diff = [(r["code"], c) for r in a for c in cols if r[c] != b[r["code"]][c]]
out = dict(steve_file=dict(path=A, sha256=sha(A), ledger_row=40), joined_file=dict(path=B, sha256=sha(B), ledger_row=41), key=dict(path=K, sha256=sha(K)),
           codes=[r["code"] for r in a], codes_equal=sorted(b) == sorted(r["code"] for r in a), fields_compared=len(a) * len(cols), differing_fields=diff,
           pair_equals_key=all(b[c]["pair"] == key[c] for c in b), code_to_pair=key, identical=not diff)
json.dump(out, open(f"{DR}/R19/tables/phase4_calls_join_check.json", "w"), indent=1); print(json.dumps({k: out[k] for k in ("codes_equal", "fields_compared", "differing_fields", "pair_equals_key", "identical")}))
