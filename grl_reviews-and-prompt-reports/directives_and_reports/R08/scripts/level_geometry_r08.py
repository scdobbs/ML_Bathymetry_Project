"""R08 §4 (site level across units) and §5 (acquisition geometry).
§4 per pair: median tile A_fine (m) with p10-p90, median tile s_lr (m; tiles_v1_1.s_lr = the R02 256-px statistic =
   ref_s_lr), k, lr_native, hr_res, band width log2 k (and the effective width log2(f_hi / f_lr), which is smaller where
   f_hi is clipped at 0.45), band-normalised level A_fine / sqrt(log2 k). Unit value = median over the unit's tiles.
   Cross-unit Spearman rho (n = 16) with a 2,000-resample bootstrap over units (seed 20261009): level vs s_lr (raw,
   normalised); GBT_ship unit-median prediction vs observed level (raw, normalised). Two figures (raw, normalised).
§5 on the 12 units with usable ship products, per pair at tile scale: partial Spearman rho given log s_lr between each
   of {log A_fine, log GBT_ship, log V2 seed-averaged} and each of {xtrack_mean_128, beam_mean_128, logcount_med_128}
   (the frozen feature values over the tile's 128-px window). Unit = median over pairs; median with bootstrap CI over units."""
from __future__ import annotations
import json, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r08_common as R8, r04_common as C4
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#F0E442", "#000000"]   # fixed categorical order

def level_figure(U, EW, col_level, col_s, fname, ylabel):
    ks = sorted(set(U.k_median.astype(int)) | set(EW.k_median.astype(int))); kcol = {k: OKABE_ITO[i % len(OKABE_ITO)] for i, k in enumerate(ks)}
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for k in ks:
        d = U[U.k_median.astype(int) == k]
        if len(d): ax.scatter(d[col_s], d[col_level], s=46, color=kcol[k], edgecolor="white", linewidth=0.8, label=f"k = {k}", zorder=3)
    OFF = {"vu02": (5, -9), "lu10": (-24, 3), "lu15": (5, 3)}   # fixed label offsets where two units sit together
    for r in U.itertuples(): ax.annotate(r.unit, (getattr(r, col_s), getattr(r, col_level)), textcoords="offset points", xytext=OFF.get(r.unit, (5, 3)), fontsize=7.5, color="#333333")
    for r in EW.itertuples():
        ax.scatter(getattr(r, col_s), getattr(r, col_level), s=90, marker="*", facecolor="none", edgecolor="#333333", linewidth=1.2, label="EW0207 (separate)", zorder=4)
        ax.annotate("EW0207", (getattr(r, col_s), getattr(r, col_level)), textcoords="offset points", xytext=(-40, -11), fontsize=7.5, color="#333333")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("median tile s_lr (m)"); ax.set_ylabel(ylabel)
    ax.grid(True, which="major", color="#e6e6e6", linewidth=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8, title="pair k (unit median)", title_fontsize=8); fig.tight_layout()
    fig.savefig(f"{R8.FIGURES}/{fname}.pdf"); fig.savefig(f"{R8.FIGURES}/{fname}.png", dpi=160); plt.close(fig)

if __name__ == "__main__":
    t0 = time.time(); m = R8.start("level_geometry_r08"); os.makedirs(R8.OUT_LEVEL, exist_ok=True); os.makedirs(R8.FIGURES, exist_ok=True); units = list(C4.UNITS)
    P, rules = R8.load_pred_tile(); tiles = R8.load_tiles(); feats = R8.load_features(); ft = feats[feats.block_id < 0].set_index("tile_id")
    PE = pd.read_parquet(R8.PRED_TILE_EW_R05)
    T = tiles.merge(pd.concat([P[["tile_id", "GBT_ship_A_fine", "V2_seedavg", "ref_s_lr"]], PE[["tile_id", "GBT_ship_A_fine", "ref_s_lr"]].assign(V2_seedavg=np.nan)], ignore_index=True), on="tile_id", how="left")
    assert np.allclose(T.s_lr.values, T.ref_s_lr.values, rtol=0, atol=0), "tiles_v1_1.s_lr != ref_s_lr"
    T["log2k"] = np.log2(T.k.astype(float)); T["oct_eff"] = np.log2(T.f_hi / T.f_lr); T["A_norm"] = T.A_fine / np.sqrt(T.log2k); T["GBT_norm"] = T.GBT_ship_A_fine / np.sqrt(T.log2k)
    # ---- §4 per pair
    rows = []
    for pid, d in T.groupby("pair", sort=False):
        rows.append(dict(unit=C4.UNIT_OF[pid], pair=pid, n_tiles=len(d), A_fine_median_m=float(d.A_fine.median()), A_fine_p10_m=float(d.A_fine.quantile(0.1)), A_fine_p90_m=float(d.A_fine.quantile(0.9)),
                         s_lr_median_m=float(d.s_lr.median()), k=int(d.k.iloc[0]), lr_native_m=float(d.lr_native.iloc[0]), hr_res_m=float(d.hr_res.iloc[0]), log2k=float(d.log2k.iloc[0]), oct_eff=float(d.oct_eff.iloc[0]), f_hi_clipped=bool(d.f_hi.iloc[0] < d.k.iloc[0] * d.f_lr.iloc[0] - 1e-12),
                         A_norm_median=float(d.A_norm.median()), GBT_ship_median_m=float(d.GBT_ship_A_fine.median()), GBT_norm_median=float(d.GBT_norm.median())))
    PP = pd.DataFrame(rows); PP.to_csv(f"{R8.OUT_LEVEL}/level_per_pair.csv", index=False)
    # ---- §4 per unit (median over the unit's tiles)
    urows = []
    for u, d in T.groupby("unit"):
        urows.append(dict(unit=u, n_pairs=d.pair.nunique(), n_tiles=len(d), A_fine_median_m=float(d.A_fine.median()), s_lr_median_m=float(d.s_lr.median()), A_norm_median=float(d.A_norm.median()), GBT_ship_median_m=float(d.GBT_ship_A_fine.median()), GBT_norm_median=float(d.GBT_norm.median()),
                          k_median=float(d.groupby("pair").k.first().median()), k_values=sorted(set(d.k.astype(int))), lr_native_values=sorted(set(np.round(d.lr_native, 2))), log2k_median=float(d.groupby("pair").log2k.first().median()), low_amplitude=(u in R8.LOW_AMP)))
    UU = pd.DataFrame(urows); UU.to_csv(f"{R8.OUT_LEVEL}/level_per_unit.csv", index=False)
    U = UU[UU.unit.isin(units)].reset_index(drop=True); EW = UU[UU.unit == R8.EW_UNIT]
    corr = dict(level_raw_vs_s_lr=R8.spearman_boot(U.A_fine_median_m, U.s_lr_median_m), level_norm_vs_s_lr=R8.spearman_boot(U.A_norm_median, U.s_lr_median_m),
                GBT_raw_vs_level_raw=R8.spearman_boot(U.GBT_ship_median_m, U.A_fine_median_m), GBT_norm_vs_level_norm=R8.spearman_boot(U.GBT_norm_median, U.A_norm_median),
                k_vs_s_lr=R8.spearman_boot(U.k_median, U.s_lr_median_m), log2k_vs_level_raw=R8.spearman_boot(U.log2k_median, U.A_fine_median_m))
    R8.dump(dict(n_units=len(U), unit_value="median over the unit's tiles", correlations=corr, n_boot=R8.N_BOOT, seed=R8.SEED, k_range=[int(U.k_median.min()), int(U.k_median.max())], lr_native_range_m=[float(T[T.unit.isin(units)].lr_native.min()), float(T[T.unit.isin(units)].lr_native.max())],
                 pairs_with_f_hi_clipped=PP[PP.f_hi_clipped].pair.tolist()), f"{R8.OUT_LEVEL}/level_correlations.json")
    level_figure(U, EW, "A_fine_median_m", "s_lr_median_m", "R08_fig_level_raw", "median tile A_fine (m)")
    level_figure(U, EW, "A_norm_median", "s_lr_median_m", "R08_fig_level_bandnorm", "median tile A_fine / sqrt(log2 k)  (m per sqrt-octave)")
    print("§4 " + "; ".join(f"{k} rho {v['rho']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]" for k, v in corr.items()), flush=True)
    # ---- §5 acquisition geometry
    usable_feat = sorted(u for u, d in feats[feats.block_id < 0].groupby("unit") if d.ship_usable.all() and u in units); assert usable_feat == sorted(R8.USABLE_UNITS), (usable_feat, R8.USABLE_UNITS)
    G = P[P.unit.isin(R8.USABLE_UNITS)].copy()
    for c in ("xtrack_mean_128", "beam_mean_128", "logcount_med_128"): G[c] = ft[c].reindex(G.tile_id).values
    TARGETS = {"log_A_fine": np.log10(G.A_fine.values + 1e-9), "log_GBT_ship": np.log10(G.GBT_ship_A_fine.values + 1e-9), "log_V2_seedavg": np.log10(G.V2_seedavg.values + 1e-9)}
    GEOM = {"xtrack_mean_128": G.xtrack_mean_128.values, "beam_angle_mean_128": G.beam_mean_128.values, "log_count_median_128": G.logcount_med_128.values}
    z = np.log10(G.ref_s_lr.values); prows = []
    for pid in sorted(set(G.pair)):
        s = (G.pair.values == pid); r = dict(unit=C4.UNIT_OF[pid], pair=pid, n_tiles=int(s.sum()), n_xtrack_defined=int(np.isfinite(GEOM["xtrack_mean_128"][s]).sum()))
        for tn, tv in TARGETS.items():
            for gn, gv in GEOM.items(): r[f"{tn}|{gn}"] = R8.partial_spearman(tv[s], gv[s], z[s])
        prows.append(r)
    PR = pd.DataFrame(prows); PR.to_csv(f"{R8.OUT_LEVEL}/geometry_partial_rho_per_pair.csv", index=False)
    keys = [c for c in PR.columns if "|" in c]; UR = PR.groupby("unit")[keys].median().reindex(list(R8.USABLE_UNITS)); UR["n_pairs"] = PR.groupby("unit").size().reindex(UR.index)
    UR.reset_index().to_csv(f"{R8.OUT_LEVEL}/geometry_partial_rho_per_unit.csv", index=False)
    summ = {k: R8.boot_median_ci(UR[k].values) for k in keys}
    R8.dump(dict(units=list(R8.USABLE_UNITS), n_units=len(R8.USABLE_UNITS), conditioning="log10 ref_s_lr (tiles_v1_1 s_lr, 256-px R02 statistic)", method="partial Spearman: Pearson r of the residuals of rank(a), rank(b) regressed on rank(z)",
                 note_vu02="EX1402L2__MGDS_31073 (vu02) has no ship_xtrack_frac feature (R05 override): its xtrack rows are NaN and the xtrack medians are over 11 units", per_unit=UR.reset_index().to_dict("records"), summary=summ, n_boot=R8.N_BOOT, seed=R8.SEED), f"{R8.OUT_LEVEL}/geometry_summary.json")
    print("§5 " + "; ".join(f"{k} {v['value']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] n={v['n']}" for k, v in summ.items()), flush=True)
    print(f"done ({time.time()-t0:.0f} s)")
