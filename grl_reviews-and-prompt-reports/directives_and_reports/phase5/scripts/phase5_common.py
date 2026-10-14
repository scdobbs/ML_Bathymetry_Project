"""Phase 5 common module (directive R16 §B; spec phase_5/PHASE5_design_spec_2026-10-13.md): a parameterised copy of
phase4_common.py. Modes: rehearsal | phase5. Everything frozen is IMPORTED from the pinned Phase 4 module (R16 §B8), not
copied: the R09 guard and its 702-probe self-test, verify_frozen (artifacts, frozen code, the five Phase 4 scripts), the
pair-row helpers (k_provenance, row_status, unit_of), the constants (B_min, tiles, bootstrap seeds, tolerances).
The two modes differ ONLY in:
  (1) the site list      SITES[mode]   rehearsal: the development pairs + EW0207 + the no-k rows (as Phase 4);
                                       phase5: the eight pre-registered sites (spec §3.1; acquisition inputs/phase5_sites.csv)
  (2) the allow-list     phase5 only, after every lock: the ELIGIBLE rows of the supplement (§B1); a row with
                         eligible = False is "not testable (<eligibility_reason>)", is never admitted and gets only its status
  (3) the output root    ROOT[mode]    rehearsal: $SCRATCH/grl_review/R16/rehearsal; phase5: OAK grl_review_archive/phase5
  (4) the locks          enforce_locks() phase5 only: sign-off with the v3 sha256; v3 / artifact / frozen-code / Phase 4 and
                         Phase 5 script hashes; the acquisition freeze sha256 stated in v3; the supplement
                         manifest/pairs_phase5_sealed.parquet (sha256 in CHANGE_LEDGER.md; exactly the eight ids; the 64
                         Phase 4 columns unchanged + the six eligibility columns); the acquisition run record of
                         RECORD_CONTRACT_phase5.md (step1_run_phase5_*.json in reports_post_grl_review/PHASE5/); the
                         calls file (code-keyed, sha256 in the ledger; predict, score); no existing results file (score)
Guard (§B3): the R09 guard (Phase 4 tokens, 702 probes) is installed unchanged; this module adds the Phase 5 tokens (the
sealed roots `sealed_candidates_v3` and the acquisition output root `auv_ship_colocated_bathy/phase5`, and every site's
pair id, ship cruise and HR id) through r09_common._is_sealed, and self-tests them on the real Phase 5 names before any
lock. After every lock a path is admitted only if it contains an admitted (eligible) full pair id; every other path
holding a Phase 5 token stays refused; the Phase 4 tokens are never admitted here."""
from __future__ import annotations
import argparse, datetime, glob, hashlib, json, os, re, sys
DR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))      # .../directives_and_reports
P4_DIR = os.path.join(DR, "phase4", "scripts")
if P4_DIR not in sys.path: sys.path.insert(0, P4_DIR)
import phase4_common as P4                                       # inserts the R03..R09 script directories
R9, R8, R7, R6, C4, C3 = P4.R9, P4.R8, P4.R7, P4.R6, P4.C4, P4.C3
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = P4.REPO
MODES = ("rehearsal", "phase5")
Refusal = P4.Refusal
# ---- (1) sites: spec §3.1 / acquisition inputs/phase5_sites.csv (pair id = <ship cruise>__<hr id with ':' -> '_'>, RECORD_CONTRACT §3)
SITE_TABLE = (("TN188", "MGDS_18210"), ("Escanaba", "MGDS_30368"), ("Monterey", "MGDS_24756"), ("KN161L04", "MGDS_22436"),
              ("MV1209", "MGDS_21415"), ("KN173L02", "MGDS_31675"), ("MV1003", "MGDS_31860"), ("AT26-24", "MGDS_31954"))
SEALED8 = tuple(f"{c}__{h}" for c, h in SITE_TABLE)
OTHER_CANDIDATE_IDS = ("MGDS_31811",)                            # Galápagos 2nd product, not used (spec §3.1); sealed, never admitted
SITES = {"rehearsal": P4.ADMITTED["rehearsal"], "phase5": SEALED8}
ROOT = {"rehearsal": "/scratch/users/scdobbs/grl_review/R16/rehearsal", "phase5": f"{R8.OAK_ARCHIVE}/phase5"}
SUPPLEMENT = os.path.join(os.path.dirname(C4.MANIFEST), "pairs_phase5_sealed.parquet")
PHASE4_SUPPLEMENT = P4.SUPPLEMENT; PHASE4_SUPPLEMENT_SHA = "6968e04554ddeef3c335587d602cb5fab2185db32b46cbd152475a919eb59206"   # schema reference (arrow types of the 64 columns)
ELIG_COLS = (("coreg_pass", "bool"), ("k_recoverable", "bool"), ("cells_across", "double"), ("hr_res_in_range", "bool"), ("eligible", "bool"), ("eligibility_reason", "string"))
CELLS_MIN, CELLS_MAX, HR_MIN, HR_MAX = 3.88, 51.2, 0.985, 4.99   # spec §3.3 item 3 (bound kept as written, ruling ACQ-R09 H-1)
LEDGER = P4.LEDGER
PREREG_V3 = f"{DR}/R16c/PREREG_phase5_v3_1.md"; SIGNOFF = f"{DR}/R16c/SIGNOFF_phase5_prereg_v3_1.md"
# ---- (4) acquisition hand-off (RECORD_CONTRACT_phase5.md §1; tag phase5-acq-freeze)
ACQ_REPO = P4.ACQ_REPO; ACQ_RECORDS_DIR = f"{ACQ_REPO}/reports_post_grl_review/PHASE5"
ACQ_FREEZE_RECORD = f"{ACQ_RECORDS_DIR}/freeze_record_phase5_v2.json"; ACQ_FREEZE_SHA = "f5a30b87c05a90bb9155f37bbdf61de6d20fa55423d97662e59ff42b02809438"
ACQ_STEP1_REL = "src/phase5/sealed_step1_p5.py"; ACQ_STEP1_SHA = "83af2875e53a28ef9b01c95a7b533ef7befd7a00625dfa6b0fe87178a2accc47"
ACQ_REHEARSAL_DIR = f"{ACQ_REPO}/reports_post_grl_review/ACQ-R09/rehearsal"           # rehearsal records (dress rehearsal only)
SCRIPTS = ("phase5_common.py", "phase5_build.py", "phase5_figures.py", "phase5_predict.py", "phase5_score.py")
# ---- guard (§B3): the Phase 5 sealed roots (real names) and tokens
SEALED_CANDIDATES_V3 = "/scratch/groups/hilley/auv_ship_colocated_bathy/sealed_candidates_v3"; ACQ_OUT_ROOT = f"{C4.DATA}/phase5"
ROOT_TOKENS = ("sealed_candidates_v3", "auv_ship_colocated_bathy/phase5")
def p5_tokens():
    toks = set(ROOT_TOKENS) | set(OTHER_CANDIDATE_IDS)
    for pid in SEALED8: cruise, _, hr = pid.partition("__"); toks |= {pid, cruise, hr}
    return sorted(toks)
P5_TOKENS = p5_tokens(); NOT_TESTABLE = "not testable"

def sha256_file(p): return P4.sha256_file(p)
read_signoff = P4.read_signoff; ledger_has_sha = P4.ledger_has_sha; k_provenance = P4.k_provenance; unit_of = P4.unit_of; split_pairs = P4.split_pairs

# ---------------- guard extension ----------------
_admitted = (); _installed = False; _R09_IS_SEALED = None
def _p5_sealed(s): return any(t in s for t in P5_TOKENS) and not any(pid in s for pid in _admitted)
def install_guard_p5():
    """After R9.install_guard_r09: r09_common._is_sealed (looked up at call time by every guarded reader, lister and stat) becomes
    'Phase 4 sealed (unchanged R09 judgement) OR Phase 5 sealed'. Idempotent."""
    global _installed, _R09_IS_SEALED
    if _installed: return
    _R09_IS_SEALED = R9._is_sealed
    def is_sealed_p5(s): return _R09_IS_SEALED(s) or _p5_sealed(s)
    R9._is_sealed = is_sealed_p5; _installed = True

def p5_probe_paths():
    """Real Phase 5 names (never opened): per site the acquisition output directory and its files, the sealed candidate
    directory, the raw swath directory and the bare id; the bare roots; the unused Galápagos product."""
    out = {}
    for pid in SEALED8:
        cruise, _, hr = pid.partition("__"); base = f"{ACQ_OUT_ROOT}/harmonized/{pid}"
        out[pid] = [base] + [f"{base}/{f}" for f in ("hr.tif", "lr.tif", "hr_valid.tif", "lr_valid.tif", "joint_valid.tif", "ship_products_v2_1", "site_record.json")] + \
                   [f"{SEALED_CANDIDATES_V3}/{hr}", f"{C4.DATA}/raw_lr_swath/{cruise}", pid]
    out["(roots)"] = [SEALED_CANDIDATES_V3, ACQ_OUT_ROOT, f"{ACQ_OUT_ROOT}/harmonized", f"{SEALED_CANDIDATES_V3}/{OTHER_CANDIDATE_IDS[0]}"]
    return out

def guard_selftest_p5():
    """Every Phase 5 probe path through the 13 readers of the R09 self-test must raise the guard's PermissionError."""
    import io, pathlib, glob as _g, rasterio, numpy as np, pandas as pd, pyarrow.parquet as pq
    readers = {"open": open, "io.open": io.open, "rasterio.open": rasterio.open, "numpy.load": np.load, "pandas.read_parquet": pd.read_parquet, "pyarrow.read_table": pq.read_table,
               "os.listdir": os.listdir, "os.scandir": os.scandir, "os.stat": os.stat, "os.lstat": os.lstat, "glob.glob": _g.glob, "Path.open": lambda p: pathlib.Path(p).open(), "os.walk": lambda p: next(iter(os.walk(p)))}
    n = probes = 0
    for paths in p5_probe_paths().values():
        for p in paths:
            for name, fn in readers.items():
                probes += 1
                try: fn(p)
                except PermissionError as e:
                    if "GUARD" not in str(e): raise AssertionError(f"PermissionError not from the guard: {name}({p}): {e}")
                    n += 1; continue
                except Exception as e: raise AssertionError(f"guard did not block {name}({p}): {type(e).__name__}: {e}")
                raise AssertionError(f"guard did not block {name}({p})")
    assert n == probes; return n, probes

def apply_allowlist(pairs, sites=SEALED8):
    """phase5 mode only, after every lock passed (§B3): admit a path only if it contains one of the listed (eligible) FULL pair
    ids; every other path holding a Phase 5 token stays refused; the Phase 4 judgement is untouched. (`sites` is the
    pre-registered list; the dress rehearsal passes its mock list.)"""
    global _admitted
    bad = [p for p in pairs if p not in sites]
    if bad: raise Refusal(f"LOCK allow-list (phase5): {bad} are not site ids")
    _admitted = tuple(sorted(pairs)); return _admitted

# ---------------- locks (phase5 mode) ----------------
def parse_prereg_hashes(text):
    """B6 rows of the five Phase 5 scripts (`| `script` | `sha256` |`)."""
    out = {}
    for sc in SCRIPTS:
        m = re.search(rf"`{re.escape(sc)}`\s*\|?\s*`([0-9a-f]{{64}})`", text); out[sc] = m.group(1) if m else None
    return out

def verify_frozen(mode, prereg_path=None, prereg_sha=None, log=print):
    """rehearsal: phase4_common.verify_frozen('rehearsal') unchanged (v1 + artifacts). phase5: phase4_common.verify_frozen with the
    v3 file and the sign-off's sha256 (v3 sha256, the candidate artifacts, the frozen code and the five Phase 4 scripts this
    module imports, against v3's statements), then the five Phase 5 scripts against v3's B6 and the acquisition freeze sha256."""
    if mode != "phase5": return P4.verify_frozen(mode, log=log)
    pp = prereg_path or PREREG_V3
    rec = P4.verify_frozen("phase4", prereg_path=pp, prereg_sha=prereg_sha, log=log)
    text = open(pp).read(); stated = parse_prereg_hashes(text)
    for sc in SCRIPTS:
        e = stated.get(sc)
        if e is None: raise Refusal(f"LOCK pre-registration (phase5): no sha256 stated for {sc}")
        got = sha256_file(os.path.join(HERE, sc)); rec[sc] = dict(path=os.path.join(HERE, sc), sha256=got, expected=e)
        if got != e: raise Refusal(f"LOCK artifact / code (phase5): {sc} sha256 {got} != {e}")
    if ACQ_FREEZE_SHA not in text: raise Refusal(f"LOCK pre-registration (phase5): the acquisition freeze record sha256 {ACQ_FREEZE_SHA[:12]}… is not stated")
    rec["acq_freeze_sha256_stated"] = ACQ_FREEZE_SHA; return rec

def check_columns(sup, m):
    """§B1: the 64 Phase 4 supplement columns (= manifest v2.1 columns), same names, order and arrow types, then exactly the six
    eligibility columns. Returns the record; raises Refusal."""
    cols = list(sup.columns); n4 = len(m.columns)
    if cols[:n4] != list(m.columns): raise Refusal(f"LOCK supplement (phase5): the first {n4} columns differ from manifest v2.1 / the Phase 4 supplement ({sorted(set(cols[:n4]) ^ set(m.columns))})")
    if tuple(cols[n4:]) != tuple(c for c, _ in ELIG_COLS): raise Refusal(f"LOCK supplement (phase5): extra columns {cols[n4:]} != the six eligibility columns {[c for c, _ in ELIG_COLS]}")
    return dict(n_columns=len(cols), phase4_columns=n4, eligibility_columns=cols[n4:])

def check_types(path):
    """Arrow types on disk: the 64 equal the Phase 4 supplement's; the six as in RECORD_CONTRACT §4."""
    import pyarrow.parquet as pq
    if sha256_file(PHASE4_SUPPLEMENT) != PHASE4_SUPPLEMENT_SHA: raise Refusal("LOCK supplement (phase5): the Phase 4 supplement (schema reference) sha256 changed")
    ref = pq.read_schema(PHASE4_SUPPLEMENT); s = pq.read_schema(path); n4 = len(ref)
    bad = [(a.name, str(a.type), str(b.type)) for a, b in zip(list(ref)[:n4], list(s)[:n4]) if a.name != b.name or a.type != b.type]
    bad += [(c, t, str(s.field(c).type)) for c, t in ELIG_COLS if c in s.names and str(s.field(c).type) != t]
    if bad: raise Refusal(f"LOCK supplement (phase5): arrow types differ from the Phase 4 supplement / contract: {bad[:6]}")
    return dict(types_checked=len(ref) + len(ELIG_COLS), schema_reference=dict(path=PHASE4_SUPPLEMENT, sha256=PHASE4_SUPPLEMENT_SHA))

def eligibility(sup):
    """{pair: (eligible, reason)} with the consistency check eligible == items 1-3 of spec §3.3 (Refusal otherwise)."""
    out = {}
    for r in sup.to_dict("records"):
        e = r["eligible"]; c = r["cells_across"]
        if not isinstance(e, (bool,)) and str(e) not in ("True", "False"): raise Refusal(f"LOCK supplement (phase5): {r['pair_id']} eligible = {e!r} is not a bool")
        e = bool(e) if isinstance(e, bool) else (str(e) == "True")
        rule = bool(r["coreg_pass"]) and bool(r["k_recoverable"]) and bool(r["hr_res_in_range"]) and (c is not None and c == c and CELLS_MIN <= float(c) <= CELLS_MAX)
        if e != rule: raise Refusal(f"LOCK supplement (phase5): {r['pair_id']} eligible = {e} but spec §3.3 items 1-3 give {rule}")
        out[str(r["pair_id"])] = (e, "" if e else str(r.get("eligibility_reason") or ""))
    return out

def load_supplement(m, supplement=SUPPLEMENT, ledger=LEDGER, sites=SEALED8):
    """phase5 mode (§B1): the supplement must exist, its sha256 must be in CHANGE_LEDGER.md, it must hold exactly the listed site
    ids, with the 64 Phase 4 columns unchanged and the six eligibility columns. Returns (merged manifest = v2.1 rows of other
    pairs + the supplement rows, record with the eligible / not-testable split)."""
    import pandas as pd
    if not os.path.exists(supplement): raise Refusal(f"LOCK supplement (phase5): {supplement} does not exist")
    sha = sha256_file(supplement)
    if not ledger_has_sha(sha, ledger): raise Refusal(f"LOCK supplement (phase5): sha256 {sha} is not recorded in {ledger}")
    sup = pd.read_parquet(supplement)
    if sorted(sup.pair_id.astype(str)) != sorted(sites): raise Refusal(f"LOCK supplement (phase5): pair ids {sorted(sup.pair_id.astype(str))} != the {len(sites)} pre-registered site ids")
    crec = check_columns(sup, m) | check_types(supplement); el = eligibility(sup)
    merged = pd.concat([m[~m.pair_id.isin(sup.pair_id)], sup], ignore_index=True); merged.attrs["sha256"] = m.attrs.get("sha256")
    return merged, dict(path=supplement, sha256=sha, n_rows=int(len(sup)), columns=crec, eligible=sorted(p for p, (e, _) in el.items() if e),
                        not_testable={p: f"{NOT_TESTABLE} ({r})" for p, (e, r) in el.items() if not e})

def load_acq_run_record(supplement_sha, records_dir=ACQ_RECORDS_DIR, freeze_record=ACQ_FREEZE_RECORD, freeze_sha=ACQ_FREEZE_SHA, record_mode="phase5"):
    """phase5 mode (RECORD_CONTRACT_phase5.md §1), after the supplement lock: a step1_run_<record_mode>_*.json in records_dir whose
    status is complete, mode is <record_mode>, lock-2 entry cites the freeze record sha256 and supplement_sha256 is the
    supplement's; the freeze record must hash to that value and list src/phase5/sealed_step1_p5.py at the pinned sha256.
    record_mode = 'rehearsal' is used only by the dress rehearsal (ACQ-R09 rehearsal records). Returns the record (per_pair kept)."""
    if not os.path.exists(freeze_record): raise Refusal(f"LOCK acquisition run record (phase5): freeze record {freeze_record} does not exist")
    fz = sha256_file(freeze_record)
    if fz != freeze_sha: raise Refusal(f"LOCK acquisition run record (phase5): freeze record {freeze_record} sha256 {fz} != {freeze_sha}")
    fr = json.load(open(freeze_record)); st1 = (fr.get("files") or {}).get(ACQ_STEP1_REL)
    if st1 != ACQ_STEP1_SHA: raise Refusal(f"LOCK acquisition run record (phase5): freeze record lists {ACQ_STEP1_REL} at {st1} != {ACQ_STEP1_SHA}")
    cands = sorted(glob.glob(os.path.join(records_dir, f"step1_run_{record_mode}_*.json")))
    if not cands: raise Refusal(f"LOCK acquisition run record (phase5): no step1_run_{record_mode}_*.json in {records_dir}")
    seen = []
    for p in cands:
        r = json.load(open(p)); cite = ((r.get("locks") or {}).get("lock2_freeze_record") or {}).get("freeze_record_sha256"); ss = r.get("supplement_sha256")
        ok = r.get("status") == "complete" and r.get("mode") == record_mode and cite == freeze_sha and ss == supplement_sha
        seen.append(dict(path=p, status=r.get("status"), mode=r.get("mode"), freeze_record_sha256=cite, supplement_sha256=ss, accepted=ok))
        if ok: return dict(path=p, sha256=sha256_file(p), status=r["status"], mode=r["mode"], freeze_record_sha256=cite, supplement_sha256=ss, code_commit=r.get("code_commit"), per_pair=r.get("per_pair") or {}, candidates=seen)
    raise Refusal(f"LOCK acquisition run record (phase5): no record in {records_dir} is complete in {record_mode} mode, cites freeze record {freeze_sha[:12]}… and supplement {supplement_sha[:12]}…: {seen}")

def harness_rows(m, path, pairs):
    """Rehearsal only: replace the listed pairs' manifest rows by a harness supplement's rows (64 + 6 columns accepted, as §B1)."""
    import pandas as pd
    H = pd.read_parquet(path)
    if list(H.columns) != list(m.columns): check_columns(H, m)
    missing = [p for p in pairs if p not in set(H.pair_id.astype(str))]
    if missing: raise Refusal(f"harness rows {path}: no row for {missing}")
    H = H[H.pair_id.isin(pairs)]; merged = pd.concat([m[~m.pair_id.isin(H.pair_id)], H], ignore_index=True); merged.attrs["sha256"] = m.attrs.get("sha256")
    return merged, dict(path=path, sha256=sha256_file(path), pairs=list(H.pair_id.astype(str)))

def row_status(row):
    """'not testable (<eligibility_reason>)' for a supplement row with eligible = False (§B1; checked first: such a row is never
    opened); otherwise phase4_common.row_status (refused | not scorable (no k) | ok)."""
    e = row.get("eligible")
    if e is not None and e == e and str(e) in ("False", "0", "false"): return f"{NOT_TESTABLE} ({row.get('eligibility_reason') or ''})"
    return P4.row_status(row)

def enforce_locks(mode, script, pairs, calls_path=None, results_path=None, signoff=SIGNOFF, prereg_path=None, ledger=LEDGER, supplement=SUPPLEMENT, manifest=None, acq_records_dir=ACQ_RECORDS_DIR, acq_freeze_record=ACQ_FREEZE_RECORD, log=print):
    """All locks for one script. rehearsal: the frozen-input verification (v1) and the site list only. phase5: sign-off -> v3 and
    hashes -> supplement -> acquisition run record -> calls (predict, score) -> single run (score). rec['manifest'] is the merged
    manifest, rec['admitted'] the eligible ids, rec['not_testable'] {pair: status}."""
    rec = dict(mode=mode, script=script)
    bad = [p for p in pairs if p not in SITES[mode]]
    if bad: raise Refusal(f"LOCK pair list: {bad} not admitted in mode {mode}")
    if mode == "phase5":
        sha = read_signoff(signoff); rec["signoff_sha256"] = sha
        rec["verify"] = verify_frozen(mode, prereg_path=prereg_path or PREREG_V3, prereg_sha=sha, log=log)
        m = manifest if manifest is not None else C4.load_manifest(); rec["manifest"], rec["supplement"] = load_supplement(m, supplement, ledger)
        rec["admitted"] = rec["supplement"]["eligible"]; rec["not_testable"] = rec["supplement"]["not_testable"]
        rec["acq_run_record"] = load_acq_run_record(rec["supplement"]["sha256"], records_dir=acq_records_dir, freeze_record=acq_freeze_record)
        if script in ("phase5_predict.py", "phase5_score.py"): rec["calls_sha256"] = calls_lock(calls_path, ledger)
        if script == "phase5_score.py": single_run_lock(results_path)
    else:
        rec["verify"] = verify_frozen(mode, log=log)
    return rec

def calls_lock(calls_path, ledger=LEDGER):
    if not calls_path or not os.path.exists(calls_path): raise Refusal(f"LOCK calls file (phase5): {calls_path} does not exist")
    ch = sha256_file(calls_path)
    if not ledger_has_sha(ch, ledger): raise Refusal(f"LOCK calls file (phase5): sha256 {ch} is not recorded in {ledger}")
    return ch

def single_run_lock(results_path):
    if results_path and os.path.exists(results_path): raise Refusal(f"LOCK single run (phase5): results file already exists: {results_path}")

def argparser(desc):
    ap = argparse.ArgumentParser(description=desc); ap.add_argument("--mode", required=True, choices=MODES); ap.add_argument("--pairs", required=True, help="pair ids, comma- or space-separated (sbatch --export splits on commas: use spaces there)")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4))); return ap

def start(script, mode, pairs, calls_path=None, results_path=None, log=print):
    """Guard first (R09 unchanged, 702 / 702; then the Phase 5 extension and its self-test), locks, site list, then (phase5) the
    allow-list of the eligible listed pairs. Returns (manifest, root, record); record['not_testable'] = {pair: status}."""
    m0 = C4.load_manifest(); R9.install_guard_r09(m0); install_guard_p5(); n, probes, readers, rows = R9.guard_selftest_r09(m0)
    assert n == probes == 702, (n, probes)
    n5, probes5 = guard_selftest_p5()
    rec = enforce_locks(mode, script, pairs, calls_path=calls_path, results_path=results_path, manifest=m0, log=log)
    root = ROOT[mode]; os.makedirs(f"{root}/provenance", exist_ok=True)
    if mode == "phase5": rec["allowlist"] = apply_allowlist([p for p in pairs if p in rec["admitted"]]); m0 = rec.pop("manifest")
    rec.setdefault("not_testable", {})
    rec.update(time=datetime.datetime.now().isoformat(timespec="seconds"), host=os.uname().nodename, slurm_job=os.environ.get("SLURM_JOB_ID"), git_head=R8.git_head(), pairs=list(pairs),
               guard_probes_blocked=n, guard_probes=probes, guard_p5_probes_blocked=n5, guard_p5_probes=probes5, script_sha256={sc: sha256_file(os.path.join(HERE, sc)) for sc in SCRIPTS if os.path.exists(os.path.join(HERE, sc))})
    json.dump(rec, open(f"{root}/provenance/{script}_{rec['time'].replace(':', '')}_{os.getpid()}.json", "w"), indent=1, default=str)
    log(f"[phase5_common.start] {script} mode={mode}: guard {n}/{probes} + phase5 {n5}/{probes5} probes blocked; locks passed; pairs {list(pairs)}; root {root}", flush=True)
    return m0, root, rec
