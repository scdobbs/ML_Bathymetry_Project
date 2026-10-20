"""§1.1 consistency check, mechanism: on the R03 tiles of each unit (the stride-128 ones), decompose the R03-convention
A_fine (hard annulus, no taper, plane detrend, RMS over the full 256 window) into its edge and interior parts, and compare
with the same filter restricted to the central 128 and with the R04 field (taper + raised cosine, central 128)."""
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import r04_common as C, targets_v1 as T
from preprocessing import extract_clean_patches as ecp
from scipy.stats import pearsonr
m = C.start("consistency_diag"); tiles = pd.read_parquet(f"{C.ROOT}/tiles/tiles_v1.parquet")
s3 = pd.read_parquet(f"{C.R03}/s3/s3_pertile.parquet")[["pid", "key"]]; s4 = pd.read_parquet(f"{C.R03}/s4/features_all.parquet")[["pair_id", "r0", "c0"]].rename(columns={"pair_id": "pid"})
s3["r0"] = s3.key.str.extract(r"_r(\d{6})_").astype(int); s3["c0"] = s3.key.str.extract(r"_c(\d{6})").astype(int)
keys = pd.concat([s3[["pid", "r0", "c0"]], s4]).drop_duplicates(); keys = keys[(keys.r0 % 128 == 0) & (keys.c0 % 128 == 0)]
rows = []; sl = slice(64, 192)
for pid, kk in keys.groupby("pid"):
    if pid not in C.UNIT_OF: continue
    row = C.pair_row(m, pid); g = ecp.pair_setup(C.DATA, row, 256); f_lr, f_hi = C.band_edges(row, g["hr_res"]); fr = T.radial_freq(256)
    Hha = T.hard_annulus_response(fr, f_lr, f_hi); Hrc = T.raised_cosine_response(fr, f_lr, f_hi); tap = T.tukey2d()
    for r in kk.itertuples():
        r0, c0 = int(r.r0), int(r.c0); hr = g["hr_arr"][r0:r0+256, c0:c0+256].copy(); hv = g["hrv_arr"][r0:r0+256, c0:c0+256] & np.isfinite(hr); hr = np.where(hv, hr, np.nan)
        if g["joint"][r0:r0+256, c0:c0+256].mean() < 0.5 or hv.mean() < 0.5 or np.nanstd(hr) < 0.05: continue
        lr, lv = ecp.tile_ship_side(g, r0, c0, 256); inp = ecp.build_inputs(lr, lv, 256, 0.5)
        if inp is None: continue
        tgt = ecp.build_target(hr, hv, inp); j = tgt["joint"].astype(bool); hr0 = np.where(j, hr, 0.0) - tgt["delta"]
        rms = lambda b, mm: float(np.sqrt(np.mean(b[mm] ** 2))) if mm.any() else np.nan
        b_ha = T.bandpass(hr0, j, Hha, None); b_rc_t = T.bandpass(hr0, j, Hrc, tap)
        edge = np.zeros_like(j); edge[:16, :] = edge[-16:, :] = edge[:, :16] = edge[:, -16:] = True; jc = np.zeros_like(j); jc[sl, sl] = j[sl, sl]
        # interior holes: joint px within 8 px of an invalid px inside the window
        from scipy.ndimage import distance_transform_edt
        dist = distance_transform_edt(j); near_hole = j & (dist <= 8) & ~edge
        rows.append(dict(unit=C.UNIT_OF[pid], pair=pid, r0=r0, c0=c0, frac_joint=float(j.mean()), ha256=rms(b_ha, j), ha_edge16=rms(b_ha, j & edge), ha_interior=rms(b_ha, j & ~edge), ha128=rms(b_ha, jc), rc_t128=rms(b_rc_t, jc),
                         ha_nearhole=rms(b_ha, near_hole), frac_nearhole=float(near_hole.sum() / max(1, j.sum())), rc_t128_nearhole=rms(b_rc_t, near_hole & jc)))
    print(pid, len(rows), flush=True); del g
d = pd.DataFrame(rows); d.to_parquet(f"{C.ROOT}/tiles/consistency_diag.parquet"); out = []
for u, s in d.groupby("unit"):
    L = lambda a, b: float(pearsonr(np.log10(s[a]), np.log10(s[b]))[0]) if len(s) > 3 else np.nan
    out.append(dict(unit=u, name=C.R03_UNIT_NAME.get(u, u), n=len(s), med_ha256=s.ha256.median(), med_edge16=s.ha_edge16.median(), med_interior=s.ha_interior.median(), med_ha128=s.ha128.median(), med_rc_t128=s.rc_t128.median(),
                    ratio_ha128_ha256=float((s.ha128 / s.ha256).median()), ratio_rc128_ha128=float((s.rc_t128 / s.ha128).median()), r_log_ha256_ha128=L("ha256", "ha128"), r_log_ha128_rc128=L("ha128", "rc_t128"), r_log_ha256_rc128=L("ha256", "rc_t128"),
                    frac_nearhole_med=s.frac_nearhole.median(), ratio_nearhole_interior=float((s.ha_nearhole / s.ha_interior).median())))
o = pd.DataFrame(out); o.to_csv(f"{C.ROOT}/tiles/consistency_diag_units.csv", index=False); pd.set_option("display.width", 300); print(o.round(3).to_string())
