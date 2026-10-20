"""R16 Part C2 — run one Phase 5 script UNCHANGED in rehearsal mode on the mock layout, through the functions phase5 mode uses.
Usage: python run_mock_r16.py <script> --mode rehearsal --pairs "..." [script args]
Scaffolding (this file only): phase5_common.start is wrapped. The mock not-testable id is added to SITES['rehearsal'] (the
site-list lock) and then, AFTER the real start (R09 guard 702 / 702 + Phase 5 guard self-test on the real names, the rehearsal
locks), the harness runs the phase5-mode data locks on the mock files, with the same functions:
  1. phase5_common.load_supplement(m, mock supplement, mock ledger, sites = the four mock ids): merged manifest, eligible /
     not-testable split (exactly what phase5 mode returns from enforce_locks);
  2. phase5_common.load_acq_run_record on the mock PHASE5 directory (byte copies of the ACQ-R09 rehearsal record and the freeze
     record), record_mode = 'rehearsal', with the sha256 of the supplement that record cites (supplement_rehearsal_r09.parquet,
     11d0868d...): the mock supplement cannot carry that sha256 (it holds mock paths, the FK151121 harness row and the
     not-testable row), so the record is accepted against the supplement it was written with (R16 report, interpretation 1);
  3. the calls lock (predict, score: the code-keyed mock calls file's sha256 in the mock ledger) and the single-run lock (score);
  4. the mock site tokens (pair id, cruise, HR id) join the Phase 5 tokens; every mock path is probed (open / rasterio.open /
     os.stat) and must be refused; phase5_common.apply_allowlist(eligible mock ids, sites = mock ids) admits them exactly as
     phase5 mode does; the eligible mock paths must then be allowed, while the not-testable mock site, a FAKE id under the mock
     root, the bare mock roots, an admitted site's sealed_candidates_v3 directory, the real Phase 5 names (an HR path of every
     real site, the bare real roots) and a real Phase 4 sealed path must stay refused;
  5. an opened-path recorder (append-only <ROOT>/opened_paths_<script>.log).
The record returned to the script carries the accepted acquisition record (k provenance) and the not-testable statuses."""
import builtins, io, json, os, runpy, sys, time
import r16_common as R
P5 = R.P5; R9 = P5.R9
def probe(path):
    import rasterio; out = {}
    for name, fn in (("open", lambda p: open(p, "rb")), ("rasterio.open", rasterio.open), ("os.stat", os.stat)):
        try: r = fn(path); out[name] = "allowed"; (hasattr(r, "close") and r.close())
        except PermissionError as e: out[name] = "REFUSED" if "GUARD" in str(e) else f"PermissionError: {e}"[:80]
        except Exception as e: out[name] = f"allowed ({type(e).__name__})"
    return out
def install_recorder(logpath):
    import numpy as np, pandas as pd, pyarrow.parquet as pq, rasterio
    _open0 = builtins.open
    def rec(kind, p, mode=""):
        try:
            s = str(os.fspath(p)) if not isinstance(p, int) else ""
            if s and s != logpath:
                with _open0(logpath, "a") as f: f.write(f"{os.getpid()}\t{kind}\t{mode}\t{s}\n")
        except Exception: pass
    b_open, i_open, r_open, n_load, p_read, q_read, to_pq = builtins.open, io.open, rasterio.open, np.load, pd.read_parquet, pq.read_table, pd.DataFrame.to_parquet
    def w_bopen(f, mode="r", *a, **k): rec("open", f, mode); return b_open(f, mode, *a, **k)
    def w_iopen(f, mode="r", *a, **k): rec("io.open", f, mode); return i_open(f, mode, *a, **k)
    def w_ropen(f, mode="r", *a, **k): rec("rasterio.open", f, mode); return r_open(f, mode, *a, **k)
    def w_nload(f, *a, **k): rec("numpy.load", f); return n_load(f, *a, **k)
    def w_pread(f, *a, **k): rec("pandas.read_parquet", f); return p_read(f, *a, **k)
    def w_qread(f, *a, **k): rec("pyarrow.read_table", f); return q_read(f, *a, **k)
    def w_topq(self, f, *a, **k): rec("DataFrame.to_parquet", f, "w"); return to_pq(self, f, *a, **k)
    builtins.open, io.open, rasterio.open, np.load, pd.read_parquet, pq.read_table, pd.DataFrame.to_parquet = w_bopen, w_iopen, w_ropen, w_nload, w_pread, w_qread, w_topq
def argval(flag):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else None
if __name__ == "__main__":
    script = sys.argv[1]; assert script in P5.SCRIPTS, script; assert "phase5" not in sys.argv[2:], "rehearsal only"
    path = os.path.join(R.P5_DIR, script); print(f"[run_mock_r16] {script} sha256 {P5.sha256_file(path)}; ROOT[rehearsal] = {P5.ROOT['rehearsal']}; phase5_common sha256 {P5.sha256_file(os.path.join(R.P5_DIR, 'phase5_common.py'))}", flush=True)
    P5.SITES["rehearsal"] = tuple(P5.SITES["rehearsal"]) + (R.NOT_TESTABLE_MOCK,)
    real_start = P5.start
    def start_mock(name, mode, pairs, calls_path=None, results_path=None, **kw):
        m, root, rec = real_start(name, mode, pairs, calls_path=calls_path, results_path=results_path, **kw); assert mode == "rehearsal"
        assert sorted(pairs) == sorted(R.MOCK_SITES), pairs
        m, srec = P5.load_supplement(m, R.MOCK_SUP, R.MOCK_LEDGER, sites=R.MOCK_SITES)
        acq = P5.load_acq_run_record(R.REH_SUP_SHA, records_dir=R.MOCK_PHASE5, freeze_record=R.MOCK_FREEZE, record_mode="rehearsal")
        locks = dict(supplement={k: v for k, v in srec.items()}, acq_run_record={k: v for k, v in acq.items() if k != "per_pair"})
        if name in ("phase5_predict.py", "phase5_score.py"): locks["calls_sha256"] = P5.calls_lock(calls_path, R.MOCK_LEDGER)
        if name == "phase5_score.py": P5.single_run_lock(f"{argval('--out')}/results_{argval('--tag')}.json"); locks["single_run"] = "passed"
        toks = R.mock_tokens(R.MOCK_SITES); P5.P5_TOKENS = sorted(set(P5.P5_TOKENS) | set(toks))
        elig = [p for p in pairs if p in srec["eligible"]]; nt = [p for p in pairs if p in srec["not_testable"]]
        mock_paths = {pid: [R.mock_dir(pid)] + [f"{R.mock_dir(pid)}/{f}" for f in R.FILES] + [f"{R.mock_dir(pid)}/ship_products_v2_1/ship_count.tif"] for pid in R.MOCK_SITES}
        before = {pid: {p: probe(p) for p in ps} for pid, ps in mock_paths.items()}
        al = P5.apply_allowlist(elig, sites=R.MOCK_SITES)
        after = {pid: {p: probe(p) for p in ps} for pid, ps in mock_paths.items()}
        stay = {"not-testable mock site": f"{R.mock_dir(R.NOT_TESTABLE_MOCK)}/hr.tif", "FAKE id under the mock root": f"{R.MOCK_HARM}/{R.FAKE_ID}/hr.tif", "bare mock harmonized root": R.MOCK_HARM, "bare mock phase5 root": f"{R.MOCK_DATA}/phase5",
                "bare mock sealed_candidates_v3": R.MOCK_CAND, "admitted site's sealed_candidates_v3 dir": f"{R.MOCK_CAND}/{elig[0].partition('__')[2]}", "real Phase 4 sealed path": os.path.join(P5.C4.DATA, "harmonized_confirmatory/nu01b/RC2511__MGDS_32239/hr.tif"),
                "real phase5 harmonized root": f"{P5.ACQ_OUT_ROOT}/harmonized", "real sealed_candidates_v3 root": P5.SEALED_CANDIDATES_V3}
        stay.update({f"real site {pid} hr.tif": f"{P5.ACQ_OUT_ROOT}/harmonized/{pid}/hr.tif" for pid in P5.SEALED8})
        stay_res = {k: probe(p) for k, p in stay.items()}
        ok_before = all(v[k] == "REFUSED" for d in before.values() for v in d.values() for k in v)
        ok_after = all(v[k].startswith("allowed") for pid in elig for v in after[pid].values() for k in v) and all(v[k] == "REFUSED" for pid in nt for v in after[pid].values() for k in v)
        ok_stay = all(v == "REFUSED" for d in stay_res.values() for v in d.values())
        prec = dict(script=name, mode=mode, pairs=list(pairs), eligible=elig, not_testable=srec["not_testable"], mock_tokens_added=toks, allowlist=list(al), locks=locks, probes_before_allowlist_all_refused=ok_before,
                    probes_after_allowlist_eligible_allowed_not_testable_refused=ok_after, stay_refused_all=ok_stay, stay=dict((k, dict(path=stay[k], result=stay_res[k])) for k in stay), before=before, after=after,
                    guard=[rec.get("guard_probes_blocked"), rec.get("guard_probes"), rec.get("guard_p5_probes_blocked"), rec.get("guard_p5_probes")], time=time.strftime("%Y-%m-%dT%H:%M:%S"))
        os.makedirs(f"{root}/mock_guard", exist_ok=True); json.dump(prec, open(f"{root}/mock_guard/probes_{name}_{int(time.time())}.json", "w"), indent=1, default=str)
        print(f"[run_mock_r16] locks on mock files passed (supplement {srec['sha256'][:12]}, acq record {acq['sha256'][:12]}{', calls ' + locks['calls_sha256'][:12] if 'calls_sha256' in locks else ''}); eligible {elig}; not testable {list(srec['not_testable'])}; "
              f"before allow-list all refused {ok_before}; after: eligible allowed / not-testable refused {ok_after}; FAKE / roots / real names / Phase 4 refused {ok_stay}", flush=True)
        if not (ok_before and ok_after and ok_stay): raise SystemExit("mock guard check FAILED")
        rec.update(acq_run_record=acq, not_testable=srec["not_testable"], admitted=srec["eligible"], mock_locks=locks)
        install_recorder(f"{root}/opened_paths_{name}.log"); return m, root, rec
    P5.start = start_mock; sys.argv = [path] + sys.argv[2:]; sys.path.insert(0, R.P5_DIR); runpy.run_path(path, run_name="__main__")
