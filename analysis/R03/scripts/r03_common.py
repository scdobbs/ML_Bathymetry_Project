"""R03 shared definitions. Importing this module INSTALLS THE LOCKBOX GUARD (directive R03 §7): any attempt to open a
path containing a lockbox pair id through builtins.open, rasterio.open or numpy.load raises before any read."""
from __future__ import annotations
import builtins, os, hashlib
LOCKBOX = ("AT42-03__MGDS_32007", "TN159__MGDS_21981")          # the 2 permanent-holdout pairs (D3)
ROOT = "/scratch/users/scdobbs/grl_review/R03"; R02 = "/scratch/users/scdobbs/grl_review/R02"
DATA = "/oak/stanford/groups/hilley/auv_ship_colocated_bathy"
NATIVE7 = {"NA090": ["NA090__MGDS_31212"], "NA080": ["NA080__MGDS_31290"], "TN399": ["TN399__MGDS_30373"],
           "RR1506": ["RR1506__MGDS_29779"], "AT37-13": ["AT37-13__MGDS_31199"], "FK181031": ["FK181031__MGDS_24618"],
           "TN299": ["TN299__MGDS_27339", "TN299__MGDS_31253"]}
EW0207 = {"EW0207": ["EW0207__MGDS_32556"]}
CALDIG = {"CalDIG": ["cal_dig_morro_bay__20180426m2_Channel1000", "cal_dig_morro_bay__20180427m1_Channel700",
                     "cal_dig_morro_bay__20180427m3_PockmarkNorth", "cal_dig_morro_bay__20180428m1_Cable",
                     "cal_dig_morro_bay__201804_LuciaChica2m", "cal_dig_morro_bay__20190314m4_LuciaChica970m",
                     "cal_dig_morro_bay__20190315m1_HeadlessCanyon", "cal_dig_morro_bay__20190316m1_BankTop",
                     "cal_dig_morro_bay__20190317m1_1000mGully", "cal_dig_morro_bay__20190317m2_600mGully",
                     "cal_dig_morro_bay__20190318m2_Transect601060m", "cal_dig_morro_bay__20190510m1_BankFlankHoles",
                     "cal_dig_morro_bay__20190510m2_BankFlankIncipCh", "cal_dig_morro_bay__20190511m1_6thHeadlessCany",
                     "cal_dig_morro_bay__20190511m2_BankTopEofCanyon3", "cal_dig_morro_bay__basin_flank",
                     "cal_dig_morro_bay__luciachica_2007", "cal_dig_morro_bay__pockmark_south"]}
EXPANSION7 = {"CCZ": ["ccz_so268_1"], "DISCOL": ["discol_so242_1"], "TAG": ["tag_m127"], "EW9801": ["EW9801__MGDS_31425"],
              "AR26": ["AR26__MGDS_31838"], "NA076": ["NA076__MGDS_32317"], "Amundsen": ["2009_Amundsen__MGDS_30046"]}
STOPPED_NO_K = {"KIWI10RR__MGDS_24499": "stage_c2c_sweep status=no_curve, max_recoverable_k=None",
                "TN268__MGDS_30466": "stage_c2c_sweep status=no_curve (HR superseded), max_recoverable_k=None; shares lu16 with EW0207",
                "FK181031__MGDS_24367": "stage_c2c_sweep max_recoverable_k=None (Alarcon Rise; shares lu08 with FK181031__MGDS_24618)"}
ALL15 = {**NATIVE7, **CALDIG, **EXPANSION7}            # units for the §4 all-units analyses (EW0207 separate)
UNIT_OF = {p: u for d in (NATIVE7, EW0207, CALDIG, EXPANSION7) for u, ps in d.items() for p in ps}

def _check(path):
    s = os.fspath(path) if not isinstance(path, int) else ""
    for l in LOCKBOX:
        if l in str(s):
            raise PermissionError(f"LOCKBOX GUARD (R03 §7): refusing to open {s}")
    return path

_installed = False
def install_lockbox_guard():
    global _installed
    if _installed: return
    _open = builtins.open
    def guarded_open(file, *a, **k): _check(file); return _open(file, *a, **k)
    builtins.open = guarded_open
    try:
        import rasterio; _ro = rasterio.open
        def guarded_ro(fp, *a, **k): _check(fp); return _ro(fp, *a, **k)
        rasterio.open = guarded_ro
    except ImportError: pass
    try:
        import numpy as np; _nl = np.load
        def guarded_nl(file, *a, **k): _check(file); return _nl(file, *a, **k)
        np.load = guarded_nl
    except ImportError: pass
    _installed = True

def assert_no_lockbox(pair_ids):
    bad = [p for p in pair_ids if p in LOCKBOX]
    if bad: raise PermissionError(f"LOCKBOX GUARD (R03 §7): pair list contains {bad}")
    return list(pair_ids)

def manifest_row_sha256(row):
    return hashlib.sha256(row.astype(str).to_json().encode()).hexdigest()

install_lockbox_guard()
