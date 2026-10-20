"""R04 (Phase 0) shared definitions. Importing this module INSTALLS THE GUARD (directive R04 §0.1): the R03 lockbox
guard (r03_common) is extended so that builtins.open, rasterio.open, numpy.load and pandas.read_parquet raise
PermissionError on (a) any path containing one of the 5 sealed pair ids (2 lockbox + 3 confirmatory) and (b) any path
under .../auv_ship_colocated_bathy/harmonized_confirmatory/. start() self-tests the guard (a lockbox path and a
confirmatory path must both raise), records the manifest sha256 and refuses to run if it differs from the directive's.
Frozen at phase0-freeze; the unit list is versioned separately (units-v1 = the 16 units below)."""
from __future__ import annotations
import builtins, hashlib, json, os, sys, datetime
REPO = "/home/users/scdobbs/GAN_Bathymetry_Project"
R04_SCRIPTS = f"{REPO}/grl_reviews-and-prompt-reports/directives_and_reports/R04/scripts"
R03_SCRIPTS = f"{REPO}/grl_reviews-and-prompt-reports/directives_and_reports/R03/scripts"
for p in (REPO, R03_SCRIPTS, R04_SCRIPTS):
    if p not in sys.path: sys.path.insert(0, p)
import r03_common as C3                                   # installs the R03 lockbox guard on import
ROOT = "/scratch/users/scdobbs/grl_review/R04"; R02 = C3.R02; R03 = C3.ROOT; DATA = C3.DATA
ARCHIVE = f"{DATA}/grl_review_archive"
MANIFEST = "/home/users/scdobbs/co-located-auv-ship-based-multibeam-acquisition/manifest/pairs_v2_1.parquet"
MANIFEST_SHA256 = "f57f3d8f98b320b136549de300ccac58787129af658a1c62296bab16b6c2ac35"
CONTRACT = f"{REPO}/grl_reviews-and-prompt-reports/directives_and_reports/INTERFACE_CONTRACT_v2.1.md"
CONTRACT_SHA256 = "0b4aadaf61fb04213ae18771d2a58532f39e9d49b5152ab64738d188a7e18fcb"
LOCKBOX = tuple(C3.LOCKBOX)
CONFIRMATORY = ("KN182L03__MGDS_30193", "RC2511__MGDS_32239", "SUM1004__MGDS_33090")
SEALED = LOCKBOX + CONFIRMATORY
CONFIRMATORY_DIR = "auv_ship_colocated_bathy/harmonized_confirmatory/"
E1M_THRESHOLD_M = 1.0          # §1.2 default; Steve may change it before the §6 freeze
UNITS_VERSION = "units-v1"
UNITS = {   # directive §0.1: development units (16) and their pairs (38)
    "lu00": ["RR1506__MGDS_29779"], "lu02": ["discol_so242_1"], "lu04": ["AT37-13__MGDS_31199"],
    "lu05": ["TN399__MGDS_30373", "TN399__MGDS_33068", "AT42-06__MGDS_5174"], "lu06": ["ccz_so268_1"],
    "lu07": ["EW9801__MGDS_31425", "KM0923__MGDS_31429"], "lu08": ["FK181031__MGDS_24618"], "lu09": ["tag_m127"],
    "lu10": ["NA090__MGDS_31212"], "lu11": ["NA076__MGDS_32317"],
    "lu12": ["cal_dig_morro_bay__20180426m2_Channel1000", "cal_dig_morro_bay__20180427m1_Channel700",
             "cal_dig_morro_bay__20180427m3_PockmarkNorth", "cal_dig_morro_bay__20180428m1_Cable",
             "cal_dig_morro_bay__201804_LuciaChica2m", "cal_dig_morro_bay__20190314m4_LuciaChica970m",
             "cal_dig_morro_bay__20190315m1_HeadlessCanyon", "cal_dig_morro_bay__20190316m1_BankTop",
             "cal_dig_morro_bay__20190317m1_1000mGully", "cal_dig_morro_bay__20190317m2_600mGully",
             "cal_dig_morro_bay__20190318m2_Transect601060m", "cal_dig_morro_bay__20190510m1_BankFlankHoles",
             "cal_dig_morro_bay__20190510m2_BankFlankIncipCh", "cal_dig_morro_bay__20190511m1_6thHeadlessCany",
             "cal_dig_morro_bay__20190511m2_BankTopEofCanyon3", "cal_dig_morro_bay__basin_flank",
             "cal_dig_morro_bay__luciachica_2007", "cal_dig_morro_bay__pockmark_south"],
    "lu13": ["NA080__MGDS_31290", "NA080__MGDS_31289"], "lu14": ["AR26__MGDS_31838"],
    "lu15": ["TN299__MGDS_27339", "TN299__MGDS_31253"], "nu00": ["FK151121__MGDS_22383"], "vu02": ["EX1402L2__MGDS_31073"]}
EW0207 = {"lu16": ["EW0207__MGDS_32556"]}          # reported separately, never pooled
NOT_USED_NO_K = ["KIWI10RR__MGDS_24499", "TN268__MGDS_30466", "FK181031__MGDS_24367", "EX1202L2__MGDS_20815",
                 "2009_Amundsen__MGDS_30046", "2009_Amundsen__MGDS_30047"]
UNIT_OF = {p: u for d in (UNITS, EW0207) for u, ps in d.items() for p in ps}
DEV_PAIRS = [p for ps in UNITS.values() for p in ps]
NATIVE7 = {"lu10": "NA090", "lu13": "NA080", "lu05": "TN399", "lu00": "RR1506", "lu04": "AT37-13", "lu08": "FK181031", "lu15": "TN299"}   # R02/R03 unit names
NATIVE7_PAIRS = {"NA090__MGDS_31212": "NA090", "NA080__MGDS_31290": "NA080", "TN399__MGDS_30373": "TN399", "RR1506__MGDS_29779": "RR1506",
                 "AT37-13__MGDS_31199": "AT37-13", "FK181031__MGDS_24618": "FK181031", "TN299__MGDS_27339": "TN299", "TN299__MGDS_31253": "TN299"}
R03_UNIT_NAME = {**NATIVE7, "lu16": "EW0207", "lu12": "CalDIG", "lu06": "CCZ", "lu02": "DISCOL", "lu09": "TAG", "lu07": "EW9801", "lu14": "AR26", "lu11": "NA076"}

def _check(path):
    s = os.fspath(path) if not isinstance(path, int) else ""
    s = str(s)
    for pid in SEALED:
        if pid in s: raise PermissionError(f"R04 GUARD (sealed pair): refusing to open {s}")
    if CONFIRMATORY_DIR in s or "/harmonized_confirmatory/" in s:
        raise PermissionError(f"R04 GUARD (confirmatory directory): refusing to open {s}")
    return path

_installed = False
def install_guard():
    """Extends r03_common.install_lockbox_guard() (already run on import of r03_common)."""
    global _installed
    if _installed: return
    C3.install_lockbox_guard()
    _open = builtins.open
    def guarded_open(file, *a, **k): _check(file); return _open(file, *a, **k)
    builtins.open = guarded_open
    import rasterio; _ro = rasterio.open
    def guarded_ro(fp, *a, **k): _check(fp); return _ro(fp, *a, **k)
    rasterio.open = guarded_ro
    import numpy as np; _nl = np.load
    def guarded_nl(file, *a, **k): _check(file); return _nl(file, *a, **k)
    np.load = guarded_nl
    import pandas as pd; _rp = pd.read_parquet
    def guarded_rp(path, *a, **k): _check(path); return _rp(path, *a, **k)
    pd.read_parquet = guarded_rp
    _installed = True

def guard_selftest():
    """§0.1: opening a lockbox path and a confirmatory path must both raise, through every patched entry point."""
    import rasterio, numpy as np, pandas as pd
    probes = [f"{DATA}/harmonized/{LOCKBOX[0]}/hr.tif", f"{DATA}/harmonized/{LOCKBOX[1]}/lr.tif",
              f"{DATA}/harmonized_confirmatory/nu03/{CONFIRMATORY[2]}/hr.tif", f"{DATA}/harmonized_confirmatory/nu01b/anything.tif",
              f"{DATA}/harmonized/{CONFIRMATORY[0]}/hr.tif"]
    n = 0
    for p in probes:
        for fn in (open, rasterio.open, np.load, pd.read_parquet):
            try: fn(p)
            except PermissionError: n += 1; continue
            except Exception as e: raise AssertionError(f"guard did not block {fn.__name__}({p}): {type(e).__name__}: {e}")
            raise AssertionError(f"guard did not block {fn.__name__}({p})")
    assert n == len(probes) * 4
    return n

def assert_no_sealed(pair_ids):
    bad = [p for p in pair_ids if p in SEALED]
    if bad: raise PermissionError(f"R04 GUARD: pair list contains sealed pairs {bad}")
    return list(pair_ids)

def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""): h.update(b)
    return h.hexdigest()

def load_manifest(check=True):
    """Manifest v2.1, read-only; stops if the sha256 differs from the directive header."""
    import pandas as pd
    sha = sha256_file(MANIFEST)
    if check and sha != MANIFEST_SHA256:
        raise SystemExit(f"STOP: manifest sha256 {sha} != directive {MANIFEST_SHA256}")
    m = pd.read_parquet(MANIFEST)
    m.attrs["sha256"] = sha
    return m

def manifest_dev_pairs(m):
    """Every manifest pair whose designation starts with 'development' and whose k is recoverable (not NaN)."""
    d = m[m.designation.astype(str).str.startswith("development") & m.max_recoverable_k.notna()].copy()
    d["k"] = d.max_recoverable_k.astype(int)
    return d

def verify_unit_list(m):
    """Challenge check (§0.1): the frozen UNITS (+ EW0207 handled separately) must equal the manifest's development
    pairs with recoverable k. Returns a dict with any differences (empty lists = verified)."""
    d = manifest_dev_pairs(m)
    man = {r.pair_id: r.leakage_unit for r in d.itertuples()}
    frozen = dict(UNIT_OF)
    out = dict(n_manifest_dev_with_k=len(man), n_frozen=len(frozen), n_frozen_dev_units=len(UNITS), n_frozen_dev_pairs=len(DEV_PAIRS),
               in_manifest_not_frozen=sorted(set(man) - set(frozen)), in_frozen_not_manifest=sorted(set(frozen) - set(man)),
               unit_mismatch={p: (frozen[p], man[p]) for p in set(frozen) & set(man) if frozen[p] != man[p]},
               no_k_in_manifest=sorted(m[m.designation.astype(str).str.startswith("development") & m.max_recoverable_k.isna()].pair_id),
               sealed_in_manifest=sorted(m[m.designation.isin(["lockbox", "confirmatory"])].pair_id))
    out["verified"] = not (out["in_manifest_not_frozen"] or out["in_frozen_not_manifest"] or out["unit_mismatch"])
    return out

def pair_row(m, pid):
    r = m[m.pair_id == pid]
    assert len(r) == 1, pid
    return r.iloc[0]

def products_dir(row):
    """Contract v2.1 §2: <directory containing the pair's lr.tif, from the manifest>/ship_products_v2_1/ (never from pair_id)."""
    return os.path.join(DATA, os.path.dirname(str(row["harmonized_path_lr"])), "ship_products_v2_1")

def ship_usable(row):
    v = row["ship_products_usable"]
    return bool(v is True or v == 1 or str(v).lower() == "true")

def hr_res_of(row):
    import rasterio
    with rasterio.open(os.path.join(DATA, row["harmonized_path_hr"])) as s: return float(abs(s.transform.a))

def band_edges(row, hr_res):
    """R02 T32B: f_lr = hr_res / lr_native, f_hi = min(k f_lr, 0.45), HR cycles per pixel."""
    f_lr = hr_res / float(row["lr_native_res_m"]); return f_lr, min(int(row["max_recoverable_k"]) * f_lr, 0.45)

def start(name):
    """Every R04 script calls this first: guard self-test, manifest hash, provenance record."""
    install_guard(); n = guard_selftest()
    m = load_manifest()
    rec = dict(script=name, time=datetime.datetime.now().isoformat(timespec="seconds"), host=os.uname().nodename,
               slurm_job=os.environ.get("SLURM_JOB_ID"), manifest=MANIFEST, manifest_sha256=m.attrs["sha256"], guard_probes_blocked=n,
               contract_sha256=sha256_file(CONTRACT), units_version=UNITS_VERSION, e1m_threshold_m=E1M_THRESHOLD_M)
    os.makedirs(f"{ROOT}/provenance", exist_ok=True)
    with open(f"{ROOT}/provenance/{name}_{rec['time'].replace(':', '')}.json", "w") as f: json.dump(rec, f, indent=1)
    print(f"[r04_common.start] {name}: guard self-test blocked {n}/{n} probes; manifest sha256 OK ({m.attrs['sha256'][:12]}…); contract {rec['contract_sha256'][:12]}…", flush=True)
    return m

install_guard()
