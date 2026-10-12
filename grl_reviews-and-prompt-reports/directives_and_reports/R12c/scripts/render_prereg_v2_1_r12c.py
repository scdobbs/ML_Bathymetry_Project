"""R12c Part D1 — render PREREG_phase4_v2_1.md from R11/PREREG_phase4_v2.md (sha256 9795db3b… verified) with ONLY:
  (1) the B6 row of phase4_common.py replaced by the edited file's sha256;
  (2) a section "Revision v2.1" (at most 10 lines) appended at the end.
Every other byte is v2's. Writes R12c/PREREG_phase4_v2_1.md, prints the diff and a record (R12c/tables/prereg_v2_1_render.json)."""
import difflib, hashlib, json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__)); DR = os.path.dirname(os.path.dirname(HERE)); V2 = f"{DR}/R11/PREREG_phase4_v2.md"; OUT = f"{DR}/R12c/PREREG_phase4_v2_1.md"
V2_SHA = "9795db3b22d6c54f7a85618dbaccafa054e88487b33236a7b2083ecb8c605979"; COMMON = f"{DR}/phase4/scripts/phase4_common.py"
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
REVISION = """
## Revision v2.1 (2026-10-12; directive R12c; deviation note 03)

- **Defect (REPORT_R12b H1):** `r09_common.sealed_tokens` holds the bare directory token `harmonized_confirmatory`; `phase4_common.apply_allowlist` dropped only tokens naming an admitted pair, cruise or MGDS id, so that token survived and the guard refused every confirmatory raster after all locks had passed (job 47541164 stopped at RC2511's first `rasterio.open`; no byte was read).
- **Rule (R12c §B1), implemented in `phase4_common.apply_allowlist` only, applied only after every lock has passed:** a path containing an admitted full pair id is allowed; a path containing `harmonized_confirmatory` and no admitted full pair id is refused; every other sealed token behaves as before; before the locks nothing changes (the guard self-test still blocks 702 of 702).
- **Also changed in `phase4_common.py`:** the two constants naming the pre-registration file and the sign-off file now point at this revision (`R12c/PREREG_phase4_v2_1.md`, `R12c/SIGNOFF_phase4_prereg_v2_1.md`). No other script, artifact, target, feature, candidate, rule, seed or threshold changed; `r09_common.py` is untouched.
- **Blindness:** before this revision no sealed file was read and no sealed value was computed (R12: both runs stopped in the locks; R12b: the guard refused the first open). The only sealed-pair output that exists is KN182L03's `not scorable (no k)` record, to be moved aside under `grl_review_archive/phase4_stopped_R12b/` before the build reruns (R12c Part E1).
- **Rehearsal (R12c Part C):** the four Phase 4 scripts ran end to end in rehearsal mode on development pairs copied into a sealed-style `harmonized_confirmatory/nu98/<pair>/` layout (products used and masked) and a `harmonized/<pair>/` lockbox-style layout, through the same `start`, locks, `apply_allowlist` and guard as phase4 mode; a non-admitted `harmonized_confirmatory` probe was refused during the run; the R10 reproduction and refusal-test suites passed. v2 (`R11/PREREG_phase4_v2.md`, tag `phase4-prereg-v2`) is untouched.
- References: deviation notes 01 (`84fd9a18`), 02 (`faedeb08`) and 03; REPORT_R12, REPORT_R12b, REPORT_R12c.
"""
if __name__ == "__main__":
    assert sha(V2) == V2_SHA, sha(V2); text = open(V2).read(); new_sha = sha(COMMON)
    pat = r"\| `phase4_common\.py` \| `[0-9a-f]{64}` \|"; assert len(re.findall(pat, text)) == 1
    old_row = re.search(pat, text).group(0); out = re.sub(pat, f"| `phase4_common.py` | `{new_sha}` |", text)
    if not out.endswith("\n"): out += "\n"
    out += REVISION
    open(OUT, "w").write(out); d = list(difflib.unified_diff(text.splitlines(), out.splitlines(), "R11/PREREG_phase4_v2.md", "R12c/PREREG_phase4_v2_1.md", lineterm="", n=0))
    rec = dict(v2=dict(path=V2, sha256=V2_SHA), v2_1=dict(path=OUT, sha256=sha(OUT)), phase4_common_sha256=new_sha, old_b6_row=old_row, diff="\n".join(d), n_revision_lines=len([l for l in REVISION.strip().splitlines() if l.strip()]))
    os.makedirs(f"{DR}/R12c/tables", exist_ok=True); json.dump(rec, open(f"{DR}/R12c/tables/prereg_v2_1_render.json", "w"), indent=1); print("\n".join(d)); print("v2.1 sha256", rec["v2_1"]["sha256"], "phase4_common", new_sha)
