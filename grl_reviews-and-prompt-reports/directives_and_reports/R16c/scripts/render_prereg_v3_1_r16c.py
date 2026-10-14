"""R16c §3 — render R16c/PREREG_phase5_v3_1.md from the signed R16/PREREG_phase5_v3.md (3df236e0...) by exact string replacement.
Every replacement must match exactly once (count asserted). Deterministic: re-rendering gives the same bytes. Writes
R16c/tables/prereg_v3_1_render.json, prereg_v3_1_diff.diff (unified diff v3 -> v3.1) and prereg_v3_1_diff.md (section by section).
Replacement groups (directive R16c §3):
  3.1  B1 item 1 (v3 l.35), the directive's wording verbatim;
  3.2  the blindness note (v3 l.47): the vertical-datum check added to the automated steps, only its pass or fail recorded;
  3.3  every citation of freeze v1 6769583198da... -> freeze v2 f5a30b87... (full sha256), with tag phase5-acq-freeze-v2 and entry
       script 83af2875...; where the same citation names the freeze file, the code commit or the rehearsal, those follow
       (freeze_record_phase5_v2.json; commit f88e21c, the 'commit' of freeze v2; ACQ-R11b) [H1 item 1, accepted by Steve 2026-10-13];
       EXCEPT 'Changes in v3', kept verbatim from v3 (H1 item 3, Steve's ruling 2026-10-13);
  3.4  the B6 row of phase5_common.py (767dec60... -> the new sha256);
  3.5  the 'Amendment v3.1' section (5 lines, appended);
  3.6  Prior exposure: the G values.
  B6 sign-off path (v3 l.145): R16/SIGNOFF_phase5_prereg_v3.md -> R16c/SIGNOFF_phase5_prereg_v3_1.md, the constant changed by
       §1.3 (H1 item 2, accepted by Steve 2026-10-13).
  Title / header (H1 item 4, Steve's ruling 2026-10-13): "version 3" -> "version 3.1"; "Version 3, rendered ... by directive R16" ->
       "Version 3.1, rendered ... by directive R16c"."""
import difflib, hashlib, json, os, re, sys
DR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V3 = f"{DR}/R16/PREREG_phase5_v3.md"; V3_SHA = "3df236e008466919a2b5acf94fb258a6f42abe6235a1ee1b77c0785ca8939d58"
V31 = f"{DR}/R16c/PREREG_phase5_v3_1.md"; TAB = f"{DR}/R16c/tables"; P5C = f"{DR}/phase5/scripts/phase5_common.py"
F1 = "6769583198da63d874472ea5a2ca62cb26c2f9f55003eb6e97838bfebab1d9d1"; F2 = "f5a30b87c05a90bb9155f37bbdf61de6d20fa55423d97662e59ff42b02809438"
S1 = "9f143e0c3f000163d9da48001cbd7578805c8c1042141f9994a972df73b1b582"; S2 = "83af2875e53a28ef9b01c95a7b533ef7befd7a00625dfa6b0fe87178a2accc47"
P5C_OLD = "767dec60213ac9bd9a8ec05c82a4ec95656fdfc3aeb2cbd7c6f5f47074802ea6"
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
def build():
    p5c = sha(P5C)
    R = [  # (group, old, new)
        ("title (H1 item 4 ruling)", "Phase 5 pre-registration, version 3 (a second pre-registered blind test)", "Phase 5 pre-registration, version 3.1 (a second pre-registered blind test)"),
        ("header (H1 item 4 ruling)", "> **Version 3, rendered 2026-10-13 by directive R16 from", "> **Version 3.1, rendered 2026-10-13 by directive R16c from"),
        ("3.1 B1 item 1",
         "1. co-registration passes the development QA thresholds (ACQ-R03: offset ≤ 30 m, overlap ≥ 1 km², ≥ 1 256-px tile with ≥ 50 % joint-valid cells);",
         "1. co-registration passes the development QA thresholds (ACQ-R03: offset ≤ 30 m, overlap ≥ 1 km², ≥ 1 256-px tile with ≥ 50 % joint-valid cells) and the vertical-datum check (ACQ-R11b): on the ship grid's joint-valid cells, the 95th percentile of |r − median r|, where r is the area-averaged high-resolution field minus the ship field after the rigid shift, divided by |median ship depth| there, is ≤ T_G = 0.200712 (calibrated on development pairs only). Ship line-to-line inconsistency is not checked."),
        ("3.2 blindness note",
         "Co-registration passes or fails by the automated development thresholds only;",
         "Co-registration passes or fails by the automated development thresholds and the vertical-datum check only (only its pass or fail is recorded);"),
        ("3.3 freeze citation, B3 step 1",
         f"(tag `phase5-acq-freeze`; code commit `4d93c8a`; freeze record `reports_post_grl_review/PHASE5/freeze_record_phase5.json` sha256 `{F1}`; script sha256 `{S1}`), rehearsed in ACQ-R09.",
         f"(tag `phase5-acq-freeze-v2`; code commit `f88e21c`; freeze record `reports_post_grl_review/PHASE5/freeze_record_phase5_v2.json` sha256 `{F2}`; script sha256 `{S2}`), rehearsed in ACQ-R09 and, against this freeze, in ACQ-R11b."),
        ("3.3 freeze citation, B6 statement", f"It also requires the acquisition freeze record sha256 `{F1}` to be stated here.", f"It also requires the acquisition freeze record sha256 `{F2}` to be stated here."),
        ("3.3 freeze citation, B6 run-record lock", f"- cite the freeze record sha256 `{F1}` in `locks.lock2_freeze_record`;", f"- cite the freeze record sha256 `{F2}` in `locks.lock2_freeze_record`;"),
        ("3.3 freeze citation, B6 freeze file and entry script", f"The `freeze_record_phase5.json` on disk must hash to that value and list `src/phase5/sealed_step1_p5.py` at `{S1}`.",
         f"The `freeze_record_phase5_v2.json` on disk must hash to that value and list `src/phase5/sealed_step1_p5.py` at `{S2}`."),
        ("3.4 B6 phase5_common.py row", f"| `phase5_common.py` | `{P5C_OLD}` |", f"| `phase5_common.py` | `{p5c}` |"),
        ("B6 sign-off path (constant of §1.3)", "Every Phase 5 script runs in `phase5` mode only if `R16/SIGNOFF_phase5_prereg_v3.md` exists", "Every Phase 5 script runs in `phase5` mode only if `R16c/SIGNOFF_phase5_prereg_v3_1.md` exists"),
        ("3.6 Prior exposure", "- The executor that ran ACQ-R08b saw only those numbers.\n",
         "- The executor that ran ACQ-R08b saw only those numbers.\n- Steve has seen the development and known-bad G values of the vertical-datum check (ACQ-R11, ACQ-R11b), and no value from any Phase 5 site.\n"),
    ]
    AMEND = ("\n## Amendment v3.1 (2026-10-13; directive R16c; rulings of Steve, 2026-10-13)\n\n"
             "- **Why:** ACQ-R10 stopped because two development QA steps, the vertical-datum check and the ship line-to-line check, were human judgements.\n"
             f"- **What changed:** the vertical-datum check is automated in the acquisition chain (B1 item 1, the blindness note; T_G = 0.200712 from development pairs only); the acquisition freeze is v2 (`freeze_record_phase5_v2.json` `{F2}`, tag `phase5-acq-freeze-v2`, entry script `{S2}`); in `phase5_common.py` only the freeze, entry-script, pre-registration and sign-off constants changed (B6).\n"
             "- **What did not change:** the criteria, thresholds, seeds, sites, models and product rules, the other four Phase 5 scripts and every Phase 4 file.\n"
             "- **Line check (fallback approved in advance):** ship line-to-line inconsistency is not checked, because the LR grid lattices it needs are not on disk for 36 of the 38 accepted development pairs (ACQ-R11b).\n"
             f"- **Blindness:** no sealed file was opened and no Phase 5 swath was fetched before v3.1. v3.1 supersedes v3 (`{V3_SHA}`, signed, tag `phase5-prereg-v3`), which stays as it is.\n")
    assert sha(V3) == V3_SHA, "signed v3 changed"
    t = open(V3, encoding="utf-8").read(); applied = []
    for g, a, b in R:
        n = t.count(a); assert n == 1, (g, n); t = t.replace(a, b); applied.append(dict(group=g, old=a, new=b))
    assert t.endswith("\n"); t = t + AMEND; applied.append(dict(group="3.5 Amendment v3.1 (appended)", old="", new=AMEND))
    cv3 = t.split("## Changes in v3")[1].split("## Amendment v3.1")[0]
    assert t.count(F1) == 1 and F1 in cv3, "a v1 freeze citation is left outside 'Changes in v3' (kept verbatim, H1 item 3 ruling)"; assert F2 in t and S2 in t and S1 not in t
    return t, applied, p5c
def sections(text):
    out, cur = {}, "(preamble)"
    for l in text.splitlines():
        if l.startswith("## ") or l.startswith("# "): cur = "# (title) + header" if l.startswith("# ") else l.strip()   # the title line itself changes (H1 item 4): key the title section by position
        out.setdefault(cur, []).append(l)
    for k in out:
        while out[k] and not out[k][-1].strip(): out[k].pop()   # a blank separator line before the appended section is not a change of the section above
    return out
if __name__ == "__main__":
    t, applied, p5c = build(); os.makedirs(os.path.dirname(V31), exist_ok=True)
    open(V31, "w", encoding="utf-8").write(t); h = sha(V31)
    t2, _, _ = build(); assert hashlib.sha256(t2.encode()).hexdigest() == h, "render not deterministic"
    a = open(V3, encoding="utf-8").read().splitlines(keepends=True); b = t.splitlines(keepends=True)
    ud = "".join(difflib.unified_diff(a, b, "R16/PREREG_phase5_v3.md", "R16c/PREREG_phase5_v3_1.md", n=0)); os.makedirs(TAB, exist_ok=True); open(f"{TAB}/prereg_v3_1_diff.diff", "w").write(ud)
    sa, sb = sections("".join(a)), sections(t); md = ["# v3 -> v3.1, section by section (R16c §3)\n", f"v3 `{V3_SHA}` -> v3.1 `{h}`\n", "| section | status | lines changed (v3 -> v3.1) |", "|---|---|---|"]
    for s in list(sa) + [s for s in sb if s not in sa]:
        if s not in sb: md.append(f"| {s} | removed | |"); continue
        if s not in sa: md.append(f"| {s} | added | {len(sb[s])} lines |"); continue
        d = [x for x in difflib.unified_diff(sa[s], sb[s], n=0, lineterm="") if x[:1] in "+-" and not x.startswith(("+++", "---"))]
        md.append(f"| {s} | {'verbatim' if not d else 'changed'} | {'' if not d else f'-{sum(x[0] == chr(45) for x in d)} / +{sum(x[0] == chr(43) for x in d)}'} |")
    md += ["", "## Replacements (each matched exactly once)", ""] + [f"- **{x['group']}**" for x in applied]
    open(f"{TAB}/prereg_v3_1_diff.md", "w").write("\n".join(md) + "\n")
    rec = dict(v3=dict(path=V3, sha256=V3_SHA), v3_1=dict(path=V31, sha256=h), phase5_common_sha256=p5c, deterministic=True, replacements=[x["group"] for x in applied],
               n_diff_lines=dict(minus=sum(1 for l in ud.splitlines() if l.startswith("-") and not l.startswith("---")), plus=sum(1 for l in ud.splitlines() if l.startswith("+") and not l.startswith("+++"))))
    json.dump(rec, open(f"{TAB}/prereg_v3_1_render.json", "w"), indent=1); print(json.dumps(rec, indent=1)); print("\n".join(md))
