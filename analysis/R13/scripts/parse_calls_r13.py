"""R13 Part A.2 — parse Steve's calls file with an exact copy of phase4_score.py's parsing lines (line 120: read_csv(...).fillna("") -> records;
lines 87-93: crit / dom / ab and the status rule). Reports per Q-code only; does not join to the key. Writes R13/tables/calls_parse_check.json."""
import json, os, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); R13 = os.path.dirname(HERE)
calls = pd.read_csv(f"{R13}/phase4_calls.csv").fillna(""); recs = calls.to_dict("records"); out = {}
for call in recs:
    crit = str(call.get("criteria", "") or ""); dom = str(call.get("dominant", "")); ab = str(call.get("ab_whole_survey", "")).lower()
    rule = ("artifact-dominated" if dom == "artifact" else ("low-amplitude artifact-affected IF amplitude < 0.04 m, else in C1 (flagged)" if ab == "yes" else ("in C1" + (" (flagged)" if crit not in ("", "none") else ""))))
    out[call["code"]] = dict(relief=call.get("relief"), criteria_string=crit, criteria_parsed_as="the whole string; the script never splits it, it only tests crit not in ('', 'none') -> flagged", dominant=dom, dominant_accepted=(dom in ("relief", "artifact", "mixed")), ab_whole_survey=ab, ab_accepted=(ab in ("yes", "no", "na")), status_rule_before_amplitude=rule, note_len=len(str(call.get("note", ""))))
json.dump(out, open(f"{R13}/tables/calls_parse_check.json", "w"), indent=1); print(json.dumps(out, indent=1))
