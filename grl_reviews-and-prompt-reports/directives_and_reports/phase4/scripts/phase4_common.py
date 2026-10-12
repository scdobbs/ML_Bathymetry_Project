"""Phase 4 common module (directive R10 §A1): modes, the guard, the locks and the B6 verifications.

Modes (every Phase 4 script takes --mode rehearsal | phase4). The two modes differ ONLY in:
  (1) the admitted pair list      ADMITTED[mode]      rehearsal: the 38 development pairs + EW0207;  phase4: the 5 pre-registered sealed pairs
  (2) the guard allow-list        ALLOW[mode]         rehearsal: none (the R09 guard blocks every sealed token);  phase4: the 5 sealed pairs' tokens
  (3) the output root             ROOT[mode]          rehearsal: $SCRATCH/grl_review/R10/rehearsal;  phase4: OAK grl_review_archive/phase4
  (4) the locks of §A1            enforce_locks()     phase4 only: sign-off file with the v2 sha256; pre-registration / artifact / frozen-code /
                                                      script hashes against the pre-registration; the sealed-row supplement (R10b §A3); the
                                                      acquisition run record (R11 §A2.2: a complete step1_run_phase4_*.json citing the acquisition
                                                      freeze record and the supplement's sha256); calls file + its sha256 in CHANGE_LEDGER.md
                                                      (predict, score); no existing results file (score)
Order in start(): the R09 guard (37 sealed tokens, 13 entry points) is installed and self-tested FIRST; the locks are then
enforced; the pair list is checked against ADMITTED[mode]; only after every lock passes is the allow-list applied, so a
failed lock can never open a sealed file. The refusal tests of R10 §A5 call enforce_locks() directly with dummy files and
never reach the allow-list step."""
from __future__ import annotations
import argparse, datetime, glob, hashlib, json, os, re, sys
DR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))      # .../directives_and_reports
for sub in ("R09/scripts", "R08b/scripts", "R08/scripts", "R07/scripts", "R06/scripts", "R05/scripts", "R04/scripts", "R03/scripts"):
    p = os.path.join(DR, sub)
    if p not in sys.path: sys.path.insert(0, p)
import r09_common as R9, r08_common as R8, r07_common as R7, r06_common as R6, r04_common as C4, r03_common as C3
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = R8.REPO
MODES = ("rehearsal", "phase4")
SEALED5 = tuple(C4.SEALED)                                         # KN182L03 (nu01a), RC2511 (nu01b), SUM1004 (nu03), AT42-03, TN159
CONFIRMATORY = tuple(C4.CONFIRMATORY); LOCKBOX = tuple(C4.LOCKBOX)
ADMITTED = {"rehearsal": tuple(C4.DEV_PAIRS) + (R8.EW_PAIR,) + tuple(C4.NOT_USED_NO_K), "phase4": SEALED5}   # R11 §A2.3: the no-k development rows of v2.1 are admitted in rehearsal (their only outcome is "not scorable (no k)")
ROOT = {"rehearsal": "/scratch/users/scdobbs/grl_review/R11/rehearsal", "phase4": f"{R8.OAK_ARCHIVE}/phase4"}
R10_OAK = f"{R8.OAK_ARCHIVE}/R10"                                   # R10 rehearsal outputs (EW0207 environment control, §A5)
SUPPLEMENT = os.path.join(os.path.dirname(C4.MANIFEST), "pairs_phase4_sealed.parquet")   # phase4 mode: the five sealed rows, beside manifest v2.1 (§A3)
LEDGER = f"{REPO}/grl_reviews-and-prompt-reports/CHANGE_LEDGER.md"
PREREG_V1 = f"{DR}/R09/PREREG_phase4_v1.md"; PREREG_V1_SHA = "5c636f3d5b8906c562405768e292e056f8733c69054b67b458866dea8df54c1f"
PREREG_V2 = f"{DR}/R12c/PREREG_phase4_v2_1.md"; SIGNOFF = f"{DR}/R12c/SIGNOFF_phase4_prereg_v2_1.md"      # revision v2.1 (R12c Part D; v2 = R11/PREREG_phase4_v2.md stays at its tag) and the sign-off file the Phase 4 locks read
# ---- acquisition repo hand-off (R11 §A2.2): the frozen acquisition step (tag phase4-acq-freeze) and where its Phase 4 run record is written
ACQ_REPO = os.path.dirname(os.path.dirname(C4.MANIFEST)); ACQ_RECORDS_DIR = f"{ACQ_REPO}/reports_post_grl_review/ACQ-R06b"   # sealed_step1.write_run_record: <ACQ-R06b>/step1_run_phase4_<tag>.json
ACQ_FREEZE_RECORD = f"{ACQ_RECORDS_DIR}/freeze_record.json"; ACQ_FREEZE_SHA = "331c94f01d7d1a2e52fb969c5fb1872aae2cd2ed249a3757b99e9d217b3ac0cb"
ACQ_STEP1_REL = "src/phase4/sealed_step1.py"; ACQ_STEP1_SHA = "bd773cb77eaff1786efe48c8d92210d69a90f885545cd00a07af9b35cd8753c1"
SCRIPTS = ("phase4_common.py", "phase4_build.py", "phase4_figures.py", "phase4_predict.py", "phase4_score.py")
# ---- frozen candidates and code (B2 / B6); the constants below are the R09 / R08 freeze-record values
GBT_ARTIFACT = f"{R9.OUT_ART}/gbt_ship_all16_A_fine_tile.joblib"; GBT_SHA = "ef93467262cf59c63bee8d5ebf898b6d9911204c72219f9bb0e2455258ff3f1f"
PV1_CKPT = {s: f"{R7.MODELS}/P-V1/seed{s}/fold_ALL/ckpt.pt" for s in R7.SEEDS}
PV1_SHA = {20261002: "d3f4d92824f2d0ac2a062d07a759e459d82debe79f1a671270e7810957665b32", 20261003: "a8c8cf7ebfc8de079f47c1c90b94f4a6f12ff8cb51818ec819c33c8a594c50a2", 20261004: "6f2d050108a35f5329b4df5bddbdd9b6b217568cc6f5006f47b70e2883ef6c3b"}
TVI_PATH = f"{DR}/R09/scripts/targets_v1_int.py"; TVI_SHA = "e7f37cdf3cd9863901ef6395a8c949ce83473c044312fabb4b986cc6fbb8903c"
FIELDS_R08 = f"{DR}/R08/scripts/fields_r08.py"; FIELDS_R08_SHA = "1eb0196bdf67e5372d21414306cc566038ae8d898682accabd61d5919cba5e80"
WIDE_R08 = f"{DR}/R08/scripts/wide_window_r08.py"; WIDE_R08_SHA = "14dfecb7ebc12220d8dca0c288fbedf0128e61fb916f292ba5a4f969f1324995"
R07_CODE_SHA = "3d6dcae7a73f4dc62a7200a0fdfb7d3636052e292229f4b2d35caa3610d03a55"     # code hash recorded in the P-V1 fold_ALL train logs
# ---- B4 constants (pre-registration v2)
B_MIN_DEFAULT = 8; MIN_TILES = 10; MIN_INT_PX = 2048; INTERIOR_FRAC = 0.95; N_BOOT = 2000; BOOT_SEED = 20261009; UNDEF_MAX_FRAC = 0.05; LOWAMP_M = 0.04
FIG_SEED = {"rehearsal": 20261011, "phase4": 20261011}; FIG_CODE = {"rehearsal": "Q", "phase4": "Q"}
CANDIDATES = ("s_lr", "GBT_ship", "P-V1")
POOL_SEED = 20261013                                             # §A1 pooled row
DEV_MIN_CELLS = 3.880437183601317                                # 256 x hr_res / lr_native, minimum over the 38 development pairs (FK151121); recomputed and asserted by dev_min_cells()
CTRL_TOL = dict(gbt_m=1e-6, pv1_log10=1e-4)                       # §A5 EW0207 environment control
NO_K = "not scorable (no k)"

def sha256_file(p): return C4.sha256_file(p)
def sha256_text(s): return hashlib.sha256(s.encode()).hexdigest()

class Refusal(RuntimeError):
    """Raised by a Phase 4 lock; the message names the lock."""

# ---------------- locks (phase4 mode) ----------------
def read_signoff(path=SIGNOFF):
    if not os.path.exists(path): raise Refusal(f"LOCK sign-off: {path} does not exist")
    m = re.findall(r"\b[0-9a-f]{64}\b", open(path).read())
    if not m: raise Refusal(f"LOCK sign-off: {path} holds no sha256")
    return m[0]

def parse_prereg_hashes(text):
    """sha256 values the pre-registration states for the candidate artifacts, the frozen code and the five scripts, keyed by label."""
    out = {}
    for key, pat in (("GBT_ship", r"gbt_ship_all16_A_fine_tile\.joblib`?,? sha256 `([0-9a-f]{64})`"), ("targets_v1_int", r"targets_v1_int\.py`? \(sha256 `([0-9a-f]{64})`"), ("fields_r08", r"fields_r08\.py`? \(sha256 `([0-9a-f]{64})`"),
                     ("wide_window_r08", r"wide_window_r08\.py`?,? sha256 `([0-9a-f]{64})`"), ("R07_code", r"R07 model code hash `([0-9a-f]{64})`")):
        m = re.search(pat, text); out[key] = m.group(1) if m else None
    for s in R7.SEEDS:
        m = re.search(rf"seed {s} `([0-9a-f]{{64}})`", text); out[f"P-V1_{s}"] = m.group(1) if m else None
    for sc in SCRIPTS:                                              # B6 table row `| `script` | `sha256` |` (v2) or the inline form `script` `sha256`
        m = re.search(rf"`{re.escape(sc)}`\s*\|?\s*`([0-9a-f]{{64}})`", text); out[sc] = m.group(1) if m else None
    return out

def verify_frozen(mode, prereg_path=None, prereg_sha=None, log=print):
    """B6: the pre-registration sha256 (v2 in phase4 mode, v1 in rehearsal), the candidate artifacts, the frozen code and (phase4) the five scripts.
    Raises Refusal on any mismatch. Returns the record."""
    rec = {}
    pp = prereg_path or (PREREG_V2 if mode == "phase4" else PREREG_V1); exp = prereg_sha or (None if mode == "phase4" else PREREG_V1_SHA)
    if not os.path.exists(pp): raise Refusal(f"LOCK pre-registration: {pp} does not exist")
    h = sha256_file(pp); rec["prereg"] = dict(path=pp, sha256=h, expected=exp)
    if exp is not None and h != exp: raise Refusal(f"LOCK pre-registration: {pp} sha256 {h} != {exp}")
    text = open(pp).read(); stated = parse_prereg_hashes(text) if mode == "phase4" else {}
    checks = {"GBT_ship": (GBT_ARTIFACT, stated.get("GBT_ship") or GBT_SHA), "targets_v1_int": (TVI_PATH, stated.get("targets_v1_int") or TVI_SHA), "fields_r08": (FIELDS_R08, stated.get("fields_r08") or FIELDS_R08_SHA), "wide_window_r08": (WIDE_R08, stated.get("wide_window_r08") or WIDE_R08_SHA)}
    for s in R7.SEEDS: checks[f"P-V1_{s}"] = (PV1_CKPT[s], stated.get(f"P-V1_{s}") or PV1_SHA[s])
    if mode == "phase4":
        for sc in SCRIPTS: checks[sc] = (os.path.join(HERE, sc), stated.get(sc))
    for k, (path, e) in checks.items():
        if e is None: raise Refusal(f"LOCK pre-registration: no sha256 stated for {k}")
        got = sha256_file(path); rec[k] = dict(path=path, sha256=got, expected=e)
        if got != e: raise Refusal(f"LOCK artifact / code: {k} sha256 {got} != {e}")
    code = R7.code_hash(); rec["R07_code"] = dict(sha256=code, expected=stated.get("R07_code") or R07_CODE_SHA)
    if code != rec["R07_code"]["expected"]: raise Refusal(f"LOCK frozen code: R07 model code hash {code} != {rec['R07_code']['expected']}")
    rec["frozen_r05"] = R6.check_frozen(verbose=False)                      # SystemExit on mismatch (R05 §6 and the OAK sums)
    return rec

def ledger_has_sha(sha, ledger=LEDGER): return os.path.exists(ledger) and sha in open(ledger).read()

def load_supplement(m, supplement=SUPPLEMENT, ledger=LEDGER):
    """phase4 mode (§A3): the five sealed rows come from the supplement beside manifest v2.1; it must exist, its sha256 must be in
    CHANGE_LEDGER.md, it must hold exactly the five pre-registered ids with the v2.1 columns; development rows stay those of v2.1."""
    import pandas as pd
    if not os.path.exists(supplement): raise Refusal(f"LOCK supplement: {supplement} does not exist")
    sha = sha256_file(supplement)
    if not ledger_has_sha(sha, ledger): raise Refusal(f"LOCK supplement: sha256 {sha} is not recorded in {ledger}")
    sup = pd.read_parquet(supplement)
    if sorted(sup.pair_id.astype(str)) != sorted(SEALED5): raise Refusal(f"LOCK supplement: pair ids {sorted(sup.pair_id.astype(str))} != the five pre-registered sealed ids")
    if list(sup.columns) != list(m.columns): raise Refusal(f"LOCK supplement: columns differ from manifest v2.1 ({sorted(set(sup.columns) ^ set(m.columns))})")
    merged = pd.concat([m[~m.pair_id.isin(SEALED5)], sup], ignore_index=True); merged.attrs["sha256"] = m.attrs.get("sha256")
    return merged, dict(path=supplement, sha256=sha, n_rows=int(len(sup)))

def load_acq_run_record(supplement_sha, records_dir=ACQ_RECORDS_DIR, freeze_record=ACQ_FREEZE_RECORD, freeze_sha=ACQ_FREEZE_SHA):
    """phase4 mode (R11 §A2.2), after the supplement lock: the acquisition repo must hold a Phase 4 run record of the frozen step
    (step1_run_phase4_*.json in its ACQ-R06b report directory) whose status is complete, whose lock-2 entry cites the freeze
    record sha256 331c94f0…c0cb and whose supplement_sha256 is the supplement's; the freeze record itself must hash to that
    value and list sealed_step1.py at bd773cb7…53c1. Otherwise Refusal. Returns the record (with per_pair for §A3)."""
    if not os.path.exists(freeze_record): raise Refusal(f"LOCK acquisition run record: freeze record {freeze_record} does not exist")
    fz = sha256_file(freeze_record)
    if fz != freeze_sha: raise Refusal(f"LOCK acquisition run record: freeze record {freeze_record} sha256 {fz} != {freeze_sha}")
    fr = json.load(open(freeze_record)); st1 = (fr.get("files") or {}).get(ACQ_STEP1_REL)
    if st1 != ACQ_STEP1_SHA: raise Refusal(f"LOCK acquisition run record: freeze record lists {ACQ_STEP1_REL} at {st1} != {ACQ_STEP1_SHA}")
    cands = sorted(glob.glob(os.path.join(records_dir, "step1_run_phase4_*.json")))
    if not cands: raise Refusal(f"LOCK acquisition run record: no step1_run_phase4_*.json in {records_dir}")
    seen = []
    for p in cands:
        r = json.load(open(p)); cite = ((r.get("locks") or {}).get("lock2_freeze_record") or {}).get("freeze_record_sha256"); ss = r.get("supplement_sha256")
        ok = r.get("status") == "complete" and r.get("mode") == "phase4" and cite == freeze_sha and ss == supplement_sha
        seen.append(dict(path=p, status=r.get("status"), mode=r.get("mode"), freeze_record_sha256=cite, supplement_sha256=ss, accepted=ok))
        if ok: return dict(path=p, sha256=sha256_file(p), status=r["status"], mode=r["mode"], freeze_record_sha256=cite, supplement_sha256=ss, code_commit=r.get("code_commit"), per_pair=r.get("per_pair") or {}, candidates=seen)
    raise Refusal(f"LOCK acquisition run record: no record in {records_dir} is complete in phase4 mode, cites freeze record {freeze_sha[:12]}… and supplement {supplement_sha[:12]}…: {seen}")

def k_source_class(src):
    """'sweep' (stage_c2c_recalibrate.sweep run by the step), 'June record' (prior value kept, ACQ-R06b A1) or None."""
    if not src: return None
    s = str(src); return "sweep" if "sweep (this run)" in s else ("June record" if "June" in s else s)

def k_provenance(pid, row, acq_rec):
    """R11 §A3: for one pair, k and k_sweep_status from its (supplement) row; the per-k verdicts, the source of k and the spacing
    source from the acquisition run record's per_pair entry (None where the record holds no entry)."""
    pp = ((acq_rec or {}).get("per_pair") or {}).get(pid) or {}; k = row.get("max_recoverable_k"); kk = int(k) if (isinstance(k, (int, float)) and _isfinite(k)) else None
    rk = pp.get("max_recoverable_k"); per_k = pp.get("per_k")
    return dict(k=kk, k_sweep_status=row.get("k_sweep_status"), k_source=pp.get("k_source"), k_source_class=k_source_class(pp.get("k_source")), per_k_verdicts=per_k,
                per_k_text=("; ".join(f"{a}:{b}" for a, b in per_k.items()) if isinstance(per_k, dict) else None), k_in_acq_record=(int(rk) if rk is not None else None),
                k_agrees_with_acq_record=(None if rk is None and kk is None else (kk == (int(rk) if rk is not None else None))), values_source=pp.get("values_source"), acq_record=(acq_rec or {}).get("path"))

def harness_rows(m, path, pairs):
    """Rehearsal only (R11 §A2.3): replace the listed pairs' manifest rows by the rows of a harness supplement (v2.1 columns required).
    Returns (merged manifest, record). Never used in phase4 mode (the supplement lock admits exactly the five sealed ids)."""
    import pandas as pd
    H = pd.read_parquet(path)
    if list(H.columns) != list(m.columns): raise Refusal(f"harness rows {path}: columns differ from manifest v2.1 ({sorted(set(H.columns) ^ set(m.columns))})")
    missing = [p for p in pairs if p not in set(H.pair_id.astype(str))]
    if missing: raise Refusal(f"harness rows {path}: no row for {missing}")
    H = H[H.pair_id.isin(pairs)]; merged = pd.concat([m[~m.pair_id.isin(H.pair_id)], H], ignore_index=True); merged.attrs["sha256"] = m.attrs.get("sha256")
    return merged, dict(path=path, sha256=sha256_file(path), pairs=list(H.pair_id.astype(str)))

def row_status(row):
    """'refused' if a spacing or a harmonized path is missing; NO_K if max_recoverable_k is empty; else 'ok' (§A4)."""
    def has(c): v = row.get(c); return (isinstance(v, str) and bool(v)) or (isinstance(v, (int, float)) and _isfinite(v))
    if not all(has(c) for c in ("harmonized_path_hr", "harmonized_path_lr", "lr_native_res_m")): return "refused"
    return "ok" if has("max_recoverable_k") else NO_K

def _isfinite(v):
    try: return v == v and v not in (float("inf"), float("-inf"))
    except Exception: return False

def dev_min_cells():
    """256 x hr_res / lr_native over the 38 development pairs of tiles_v1.1; must equal DEV_MIN_CELLS."""
    t = R8.load_tiles(); t = t[t.pair.isin(C4.DEV_PAIRS)]; g = t.groupby("pair").agg(hr=("hr_res", "first"), lr=("lr_native", "first")); v = float((256 * g.hr / g.lr).min())
    assert abs(v - DEV_MIN_CELLS) < 1e-6, (v, DEV_MIN_CELLS); return v

def enforce_locks(mode, script, pairs, calls_path=None, results_path=None, signoff=SIGNOFF, prereg_path=None, ledger=LEDGER, supplement=SUPPLEMENT, manifest=None, acq_records_dir=ACQ_RECORDS_DIR, acq_freeze_record=ACQ_FREEZE_RECORD, log=print):
    """All locks of §A1 / §A3 (R10, R10b) and §A2.2 (R11) for one script. In rehearsal mode only the frozen-input verification (against v1) and the pair admission apply.
    Returns the record; in phase4 mode rec['manifest'] is the merged manifest (v2.1 development rows + the supplement's sealed rows) and
    rec['acq_run_record'] the accepted acquisition run record (per_pair carried for the k provenance of §A3)."""
    rec = dict(mode=mode, script=script)
    bad = [p for p in pairs if p not in ADMITTED[mode]]
    if bad: raise Refusal(f"LOCK pair list: {bad} not admitted in mode {mode}")
    if mode == "phase4":
        sha = read_signoff(signoff); rec["signoff_sha256"] = sha
        rec["verify"] = verify_frozen(mode, prereg_path=prereg_path or PREREG_V2, prereg_sha=sha, log=log)
        m = manifest if manifest is not None else C4.load_manifest(); rec["manifest"], rec["supplement"] = load_supplement(m, supplement, ledger)
        rec["acq_run_record"] = load_acq_run_record(rec["supplement"]["sha256"], records_dir=acq_records_dir, freeze_record=acq_freeze_record)
        if script in ("phase4_predict.py", "phase4_score.py"):
            if not calls_path or not os.path.exists(calls_path): raise Refusal(f"LOCK calls file: {calls_path} does not exist")
            ch = sha256_file(calls_path); rec["calls_sha256"] = ch
            if not ledger_has_sha(ch, ledger): raise Refusal(f"LOCK calls file: sha256 {ch} is not recorded in {ledger}")
        if script == "phase4_score.py":
            if results_path and os.path.exists(results_path): raise Refusal(f"LOCK single run: results file already exists: {results_path}")
    else:
        rec["verify"] = verify_frozen(mode, log=log)
    return rec

# ---------------- guard ----------------
_allowed = (); _R09_IS_SEALED = R9._is_sealed                      # the R09 judgement as installed (kept so that repeated calls never wrap twice)
def apply_allowlist(pairs):
    """phase4 mode only, after every lock passed: admit the listed sealed pairs' tokens through the R09 / R04 / R03 checks.
    Revision v2.1 (R12c §B1; REPORT_R12b H1): r09_common.sealed_tokens also holds the bare directory token `harmonized_confirmatory`,
    which names no pair and so survived the token filter below and refused every confirmatory path. Rule, applied only here:
      1. a path that contains an admitted FULL pair id is allowed;
      2. a path that contains `harmonized_confirmatory` and no admitted full pair id stays refused (the bare token is kept);
      3. every other sealed token behaves as before (tokens naming an admitted pair, cruise or MGDS id are dropped);
      4. nothing changes before the locks (the guard self-test still blocks 702 / 702 before this function runs)."""
    global _allowed
    toks = set()
    for pid in pairs:
        toks.add(pid); cruise, _, mgds = pid.partition("__"); toks.add(cruise); toks.add(mgds)
    _allowed = tuple(sorted(toks)); full_ids = tuple(sorted(pairs))
    def allowed(s): return any(t in s for t in _allowed)
    R9.SEALED_TOKENS[:] = [t for t in R9.SEALED_TOKENS if not allowed(t)]
    def is_sealed_allow(s):                                        # rules 1 and 2: r09_common._check_r09 / listdir / scandir look _is_sealed up at call time
        if any(pid in s for pid in full_ids): return False
        return _R09_IS_SEALED(s)
    R9._is_sealed = is_sealed_allow
    r04 = C4._check_r04
    def check_r04_allow(path):
        s = str(os.fspath(path)) if not isinstance(path, int) else ""
        if allowed(s): return path
        return r04(path)
    C4._check_r04 = check_r04_allow
    def c3_check(path):
        s = str(os.fspath(path)) if not isinstance(path, int) else ""
        if allowed(s): return path
        for l in C3.LOCKBOX:
            if l in s: raise PermissionError(f"LOCKBOX GUARD (R03 §7): refusing to open {s}")
        return path
    C3._check = c3_check
    return _allowed

def unit_of(pid, m):
    if pid in C4.UNIT_OF: return C4.UNIT_OF[pid]
    row = m[m.pair_id == pid]; assert len(row) == 1, pid; return str(row.iloc[0]["leakage_unit"])

def split_pairs(s): return [p for p in s.replace(',', ' ').split() if p]

def argparser(desc):
    ap = argparse.ArgumentParser(description=desc); ap.add_argument("--mode", required=True, choices=MODES); ap.add_argument("--pairs", required=True, help="pair ids, comma- or space-separated (sbatch --export splits on commas: use spaces there)")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4))); return ap

def start(script, mode, pairs, calls_path=None, results_path=None, log=print):
    """Guard first (R09, full), locks, pair admission, then (phase4) the allow-list. Returns (manifest, root, record).
    Manifest: rehearsal = v2.1 only (C4.load_manifest, sha256-verified); phase4 = v2.1 development rows + the sealed-row supplement (§A3)."""
    m0 = C4.load_manifest(); R9.install_guard_r09(m0); n, probes, readers, rows = R9.guard_selftest_r09(m0)
    assert n == probes == 702, (n, probes)
    rec = enforce_locks(mode, script, pairs, calls_path=calls_path, results_path=results_path, manifest=m0, log=log)
    root = ROOT[mode]; os.makedirs(f"{root}/provenance", exist_ok=True)
    if mode == "phase4": rec["allowlist"] = apply_allowlist(pairs); m0 = rec.pop("manifest")
    if "acq_run_record" in rec: rec["acq_run_record"] = {k: v for k, v in rec["acq_run_record"].items() if k != "per_pair"} | dict(per_pair=rec["acq_run_record"]["per_pair"])
    rec.update(time=datetime.datetime.now().isoformat(timespec="seconds"), host=os.uname().nodename, slurm_job=os.environ.get("SLURM_JOB_ID"), git_head=R8.git_head(), pairs=list(pairs),
               guard_probes_blocked=n, guard_probes=probes, script_sha256={sc: sha256_file(os.path.join(HERE, sc)) for sc in SCRIPTS if os.path.exists(os.path.join(HERE, sc))})
    json.dump(rec, open(f"{root}/provenance/{script}_{rec['time'].replace(':', '')}_{os.getpid()}.json", "w"), indent=1, default=str)
    log(f"[phase4_common.start] {script} mode={mode}: guard {n}/{probes} probes blocked; locks passed; pairs {list(pairs)}; root {root}", flush=True)
    return m0, root, rec
