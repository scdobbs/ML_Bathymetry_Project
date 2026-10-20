"""R12c Part C2 — run one Phase 4 script UNCHANGED in rehearsal mode on the mock sealed-style layout, through the same functions phase4
mode uses. Usage: python run_mock_r12c.py <script> --mode rehearsal --pairs "..." [script args]
Scaffolding (rehearsal only, this file): phase4_common.start is wrapped so that, AFTER the real start (guard installed and self-tested 702 / 702,
the rehearsal locks, pair admission), the harness
  1. treats the mock pairs as sealed: appends r12c_common.mock_tokens(pairs) to r09_common.SEALED_TOKENS and probes every mock path with
     open / rasterio.open / os.stat (all must be refused: 'mock guard' record);
  2. admits them exactly as phase4 mode does: phase4_common.apply_allowlist(pairs) (the edited rule);
  3. probes the admitted mock paths (must not be refused) and the non-admitted FAKE probe under harmonized_confirmatory/nu99/ (must be refused:
     must-hold 2), and a sealed real path (must still be refused);
  4. installs an opened-path recorder on builtins.open / io.open / rasterio.open / numpy.load / pandas.read_parquet / pyarrow.read_table /
     DataFrame.to_parquet (append-only log <ROOT>/opened_paths_<script>.log; must-hold 3).
Nothing in phase4/scripts is edited; phase4 mode is refused here."""
import builtins, io, json, os, runpy, sys, time
import r12c_common as R
P4 = R.P4; R9 = P4.R9
def probe(path):
    """(refused_by_guard, exception_name) for open / rasterio.open / os.stat on a path (the file need not exist)."""
    import rasterio; out = {}
    for name, fn in (("open", lambda p: open(p, "rb")), ("rasterio.open", rasterio.open), ("os.stat", os.stat)):
        try: r = fn(path); out[name] = "allowed"; (hasattr(r, "close") and r.close())
        except PermissionError as e: out[name] = "REFUSED" if "GUARD" in str(e) else f"PermissionError: {e}"[:80]
        except Exception as e: out[name] = f"allowed ({type(e).__name__})"
    return out
def install_recorder(logpath, script):
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
if __name__ == "__main__":
    script = sys.argv[1]; assert script in P4.SCRIPTS, script; assert "phase4" not in sys.argv[2:], "rehearsal only"
    path = os.path.join(R.P4_DIR, script); print(f"[run_mock_r12c] {script} sha256 {P4.sha256_file(path)}; ROOT[rehearsal] = {P4.ROOT['rehearsal']}; phase4_common sha256 {P4.sha256_file(os.path.join(R.P4_DIR, 'phase4_common.py'))}", flush=True)
    real_start = P4.start
    def start_mock(name, mode, pairs, **kw):
        m, root, rec = real_start(name, mode, pairs, **kw); assert mode == "rehearsal"
        mock = [p for p in pairs if p in R.MOCK_ROWS]; toks = R.mock_tokens(mock); R9.SEALED_TOKENS.extend(toks)
        if mock: m, hrec = P4.harness_rows(m, R.MOCK_SUP, mock); print(f"[run_mock_r12c] manifest rows of {mock} replaced by the mock supplement rows (R11 --rows mechanism; phase4_figures / phase4_predict take no --rows): {hrec['sha256'][:12]}", flush=True)
        mock_paths = {pid: [f"{R.mock_dir(pid)}/{f}" for f in R.FILES] + [f"{R.mock_dir(pid)}/ship_products_v2_1/ship_count.tif"] for pid in mock}
        before = {pid: {p: probe(p) for p in ps} for pid, ps in mock_paths.items()}
        sealed_real = os.path.join(P4.C4.DATA, "harmonized_confirmatory/nu01b/RC2511__MGDS_32239/hr.tif")
        al = P4.apply_allowlist(mock)                                                  # the same admit step as phase4 mode, after the locks
        after = {pid: {p: probe(p) for p in ps} for pid, ps in mock_paths.items()}; fake = probe(R.FAKE_PROBE); real = probe(sealed_real)
        ok_before = all(v[k] == "REFUSED" for d in before.values() for v in d.values() for k in v); ok_after = all(v[k].startswith("allowed") for d in after.values() for v in d.values() for k in v)
        ok_fake = all(v == "REFUSED" for v in fake.values()); ok_real = all(v == "REFUSED" for v in real.values())
        prec = dict(script=name, mode=mode, pairs=list(pairs), mock_pairs=mock, mock_tokens_added=toks, allowlist=list(al), tokens_remaining=list(R9.SEALED_TOKENS), probes_before_allowlist_all_refused=ok_before, probes_after_allowlist_all_allowed=ok_after,
                    fake_probe=dict(path=R.FAKE_PROBE, result=fake, refused=ok_fake), real_sealed_probe=dict(path=sealed_real, result=real, refused=ok_real), before=before, after=after, guard=[rec.get("guard_probes_blocked"), rec.get("guard_probes")], time=time.strftime("%Y-%m-%dT%H:%M:%S"))
        os.makedirs(f"{root}/mock_guard", exist_ok=True); json.dump(prec, open(f"{root}/mock_guard/probes_{name}_{int(time.time())}.json", "w"), indent=1)
        print(f"[run_mock_r12c] mock guard: tokens +{len(toks)}; before allow-list all refused {ok_before}; after allow-list all allowed {ok_after}; FAKE probe refused {ok_fake}; real sealed probe refused {ok_real}", flush=True)
        if not (ok_before and ok_after and ok_fake and ok_real): raise SystemExit("mock guard check FAILED")
        install_recorder(f"{root}/opened_paths_{name}.log", name); return m, root, rec
    P4.start = start_mock; sys.argv = [path] + sys.argv[2:]; sys.path.insert(0, R.P4_DIR); runpy.run_path(path, run_name="__main__")
