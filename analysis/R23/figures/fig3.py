"""R23 Figure 3: (a) per-unit rho of the spectral extrapolation against s_lr on the 64-cell ship window (R08, all tiles);
(b) unit-median A_fine against Goff (2020) predicted abyssal-hill RMS height (R15, 10 units); (c) the same y against
unit-median s_lr, all 16 units. Values read from frozen tables; only the Spearman checks are recomputed from the plotted points."""
from __future__ import annotations
import json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import fig_style as S
font = S.apply()
import matplotlib.pyplot as plt

SCORES = f"{S.AR}/R08/wide_window/scores.json"
UNITS_CSV = f"{S.AR}/R15/out_ruling/units.csv"
meta = dict(directive="R23", figure=3, assertions=[]); rows = []
def check(name, ok, got, want):
    meta["assertions"].append(dict(name=name, passed=bool(ok), got=got, want=want)); print(f"  assert {name}: {'PASS' if ok else 'FAIL'} got={got} want={want}")

U = pd.read_csv(UNITS_CSV); U = U[U.unit != "lu16"].set_index("unit")
sc = json.load(open(SCORES))["scores_per_unit"]; x_a = pd.Series(sc["s_lr_W64"]); y_a = pd.Series(sc["shat_fine_W64"])
units16 = [u for u in U.index if u in x_a.index]
check("16 units in (a)", len(units16) == 16 and len(x_a) == 16, (len(units16), len(x_a)), 16)
xa, ya = x_a.loc[units16], y_a.loc[units16]
check("median s_lr rho (64-cell window, all tiles) = 0.350", round(float(np.median(xa)), 3) == 0.350, round(float(np.median(xa)), 3), 0.350)
check("median spectral rho = 0.09", round(float(np.median(ya)), 2) == 0.09, round(float(np.median(ya)), 2), 0.09)
real = [u for u in units16 if U.loc[u, "group"] == "real-relief"]; art = [u for u in units16 if U.loc[u, "group"] == "artifact-affected"]
G = U[U.domain_class.isin(["computed", "possibly interpolated"])]; comp = list(G[G.domain_class == "computed"].index); interp = list(G[G.domain_class == "possibly interpolated"].index)
check("10 units with Goff defined, 4 computed", (len(G) == 10) and (len(comp) == 4), (len(G), len(comp)), (10, 4))
rho_bg = spearmanr(G.goff_b, G.A_fine_median_m).statistic; rho_c10 = spearmanr(G.s_lr_median_m, G.A_fine_median_m).statistic; rho_c16 = spearmanr(U.s_lr_median_m, U.A_fine_median_m).statistic
check("Spearman rho(Goff, A_fine) = -0.31 on the 10 plotted points", round(float(rho_bg), 2) == -0.31, round(float(rho_bg), 3), -0.31)
check("Spearman rho(s_lr, A_fine) = +0.64 on the 10 black points", round(float(rho_c10), 2) == 0.64, round(float(rho_c10), 3), 0.64)
check("Spearman rho(s_lr, A_fine) = 0.70 on all 16", round(float(rho_c16), 2) == 0.70, round(float(rho_c16), 3), 0.70)
meta["spearman"] = dict(goff_vs_Afine_10=float(rho_bg), s_lr_vs_Afine_10=float(rho_c10), s_lr_vs_Afine_16=float(rho_c16))

FIG_H = 61.0; fig = S.fig_new(S.WIDTH_MM, FIG_H)
axa = S.ax_mm(fig, 13.0, 5.0, 45.0, 45.0); axb = S.ax_mm(fig, 72.0, 5.0, 44.0, 45.0); axc = S.ax_mm(fig, 122.0, 5.0, 44.0, 45.0)
for ax in (axa, axb, axc): S.style_axes(ax)
K = dict(linestyle="none", markersize=S.MS, markeredgewidth=0.5, zorder=3)
# (a)
lim_lo = float(np.floor((min(xa.min(), ya.min()) - 0.05) * 10) / 10); lim_hi = float(np.ceil((max(xa.max(), ya.max()) + 0.05) * 10) / 10)
l11, = axa.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color=S.GRAY_ZERO, linewidth=S.LW_THIN, linestyle=(0, (3, 2)), zorder=0.5); l11.set_gid("fig3a_one_to_one")
S.zero_line(axa, "x", "fig3a_zero_x"); S.zero_line(axa, "y", "fig3a_zero_y")
h, = axa.plot(xa.loc[real], ya.loc[real], marker="o", color=S.OKABE["black"], markerfacecolor=S.OKABE["black"], **K); h.set_gid("fig3a_real_relief")
h, = axa.plot(xa.loc[art], ya.loc[art], marker="o", color=S.OKABE["black"], markerfacecolor="white", **K); h.set_gid("fig3a_artifact_affected")
axa.set_xlim(lim_lo, lim_hi); axa.set_ylim(lim_lo, lim_hi); axa.set_aspect("equal")
axa.set_xlabel("ρ($s_{\\mathrm{lr}}$), 64-cell window", fontsize=S.FS); axa.set_ylabel("ρ(spectral extrapolation)", fontsize=S.FS)
for u in units16: rows.append(dict(panel="a", unit=u, group=U.loc[u, "group"], x=float(xa[u]), y=float(ya[u]), source_file=SCORES, source_column="scores_per_unit.s_lr_W64 (x), scores_per_unit.shat_fine_W64 (y)"))
# (b)
h, = axb.plot(G.loc[comp].goff_b, G.loc[comp].A_fine_median_m, marker="o", color=S.OKABE["black"], markerfacecolor=S.OKABE["black"], **K); h.set_gid("fig3b_computed")
h, = axb.plot(G.loc[interp].goff_b, G.loc[interp].A_fine_median_m, marker="o", color=S.OKABE["black"], markerfacecolor="white", **K); h.set_gid("fig3b_possibly_interpolated")
axb.set_yscale("log"); axb.set_xlabel("Goff (2020) RMS height (m)", fontsize=S.FS); axb.set_ylabel("unit-median $A_{\\mathrm{fine}}$ (m)", fontsize=S.FS)
gx = G.goff_b; axb.set_xlim(0, 400.0)   # 400 m leaves the upper right free for the legend (max Goff value 256 m)
for u in G.index: rows.append(dict(panel="b", unit=u, group=U.loc[u, "domain_class"], x=float(G.loc[u, "goff_b"]), y=float(G.loc[u, "A_fine_median_m"]), source_file=UNITS_CSV, source_column="goff_b (x), A_fine_median_m (y), domain_class"))
# (c) shared y with (b)
others = [u for u in units16 if u not in G.index]
h, = axc.plot(U.loc[comp].s_lr_median_m, U.loc[comp].A_fine_median_m, marker="o", color=S.OKABE["black"], markerfacecolor=S.OKABE["black"], **K); h.set_gid("fig3c_goff_computed")
h, = axc.plot(U.loc[interp].s_lr_median_m, U.loc[interp].A_fine_median_m, marker="o", color=S.OKABE["black"], markerfacecolor="white", **K); h.set_gid("fig3c_goff_possibly_interpolated")
h, = axc.plot(U.loc[others].s_lr_median_m, U.loc[others].A_fine_median_m, marker="s", color=S.GRAY_LIGHT, markerfacecolor=S.GRAY_LIGHT, **K); h.set_gid("fig3c_goff_undefined")
axc.set_xscale("log"); axc.set_yscale("log"); axc.set_xlabel("unit-median $s_{\\mathrm{lr}}$ (m)", fontsize=S.FS)
for u in units16: rows.append(dict(panel="c", unit=u, group=("Goff defined: " + U.loc[u, "domain_class"]) if u in G.index else "Goff undefined", x=float(U.loc[u, "s_lr_median_m"]), y=float(U.loc[u, "A_fine_median_m"]), source_file=UNITS_CSV, source_column="s_lr_median_m (x), A_fine_median_m (y)"))
ylo = float(U.A_fine_median_m.min()) / 1.8; yhi = float(U.A_fine_median_m.max()) * 12.0   # headroom for the panel-b legend
for ax in (axb, axc): ax.set_ylim(ylo, yhi)
axc.tick_params(labelleft=False); axc.tick_params(axis="x", which="minor", labelbottom=False)
from matplotlib.ticker import FixedLocator, FixedFormatter
xt = [2, 5, 10, 20, 50]; axc.xaxis.set_major_locator(FixedLocator(xt)); axc.xaxis.set_major_formatter(FixedFormatter([str(v) for v in xt]))
for ax in (axa, axb, axc): ax.tick_params(axis="both", which="both", labelsize=S.FS)
from matplotlib.lines import Line2D
LK = dict(linestyle="none", markersize=S.MS, markeredgewidth=0.5, color=S.OKABE["black"])
la = axa.legend(handles=[Line2D([], [], marker="o", markerfacecolor=S.OKABE["black"], label="real-relief unit", **LK), Line2D([], [], marker="o", markerfacecolor="white", label="artifact-affected unit", **LK)], loc="lower left", frameon=False, fontsize=S.FS, handletextpad=0.3, borderaxespad=0.3, labelspacing=0.25); la.set_gid("fig3a_legend")
lb = axb.legend(handles=[Line2D([], [], marker="o", markerfacecolor=S.OKABE["black"], label="Goff computed", **LK), Line2D([], [], marker="o", markerfacecolor="white", label="Goff possibly interpolated", **LK)], loc="upper right", frameon=False, fontsize=S.FS, handletextpad=0.3, borderaxespad=0.3, labelspacing=0.25); lb.set_gid("fig3b_legend")
lc = axc.legend(handles=[Line2D([], [], marker="s", markerfacecolor=S.GRAY_LIGHT, markeredgecolor=S.GRAY_LIGHT, label="no Goff value", linestyle="none", markersize=S.MS, markeredgewidth=0.5, color=S.GRAY_LIGHT)], loc="lower right", frameon=False, fontsize=S.FS, handletextpad=0.3, borderaxespad=0.3); lc.set_gid("fig3c_legend")
S.panel_letter(fig, axa, "a", dx_mm=-11.0, dy_mm=0.5); S.panel_letter(fig, axb, "b", dx_mm=-8.0, dy_mm=0.5); S.panel_letter(fig, axc, "c", dx_mm=-4.0, dy_mm=0.5)
S.finish(fig, 3, font, meta, rows)
