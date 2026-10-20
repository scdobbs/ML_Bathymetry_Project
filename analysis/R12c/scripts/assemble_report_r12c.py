"""Splice R12c/tables/*.md into REPORT_R12c_2026-10-12.md in place of its placeholders; --preserve <job> and --commit <hash> fill the two hash-note placeholders."""
import os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.dirname(HERE); REP = f"{R}/REPORT_R12c_2026-10-12.md"
def tab(n): return open(f"{R}/tables/{n}").read().strip()
if __name__ == "__main__":
    t = open(REP).read()
    for ph, n in (("A_TABLE", "a_code_reading.md"), ("B2_TABLE", "b2.md"), ("MOCK_TABLE", "mock.md"), ("DRESS_TABLE", "dress.md"), ("DRESS_PATHS_TABLE", "dress_paths.md"), ("REPRO_TABLE", "repro.md"), ("REFUSAL_TABLE", "refusal.md")):
        if ph in t: t = t.replace(ph, tab(n))
    a = sys.argv[1:]
    if "--preserve" in a: t = t.replace("JOB_PRESERVE", a[a.index("--preserve") + 1])
    if "--commit" in a: t = t.replace("R12C_COMMIT", f"`{a[a.index('--commit') + 1]}`")
    open(REP, "w").write(t); print("placeholders left:", re.findall(r"\b(A_TABLE|B2_TABLE|MOCK_TABLE|DRESS_TABLE|DRESS_PATHS_TABLE|REPRO_TABLE|REFUSAL_TABLE|JOB_PRESERVE|R12C_COMMIT)\b", t))
