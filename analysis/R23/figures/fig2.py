"""R23 Figure 2: (a) per-unit s_lr rho on interior tiles with 95 % block-bootstrap CIs (R22) and the pooled row (14 pairs,
stratified CI, seed 20261013); (b) rho(model) - rho(s_lr) per unit for GBT_ship, V2 and P-V1 on the same tiles.
All values read from frozen tables; nothing recomputed."""
from __future__ import annotations
import json
import numpy as np, pandas as pd
import fig_style as S
font = S.apply()
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R22 = f"{S.ACQ}/reports_post_grl_review/R22/table_S2_unit_CIs.csv"
TRIPLES = f"{S.AR}/R10b/rehearsal/triples.json"
R09 = f"{S.AR}/R09/scores/scores_r09.json"
UNITS_CSV = f"{S.AR}/R15/out_ruling/units.csv"
TABLE_S1 = f"{S.ACQ}/reports_post_grl_review/ACQ-R16/TableS1_pairs.csv"
# short region names: Table S1 'region' where it is specific; where Table S1 says only "Pacific" the manifest site name (R15 units.csv 'site') is used
LABELS = {"lu00": "Kermadec Arc", "lu04": "Costa Rica Rift", "lu05": "EPR 9°N", "lu07": "Loihi Seamount", "lu08": "Pescadero Basin",
          "lu09": "TAG, MAR 26°N", "lu13": "Davidson Seamount", "lu15": "Gorda / Cascadia", "nu00": "Izu-Bonin-Mariana arc"}
MODELS = [("GBT_ship", "GBT$_{\\mathrm{ship}}$", "D", S.OKABE["vermillion"], +0.24), ("V2", "V2", "^", S.OKABE["green"], 0.0), ("P-V1", "P-V1", "s", S.OKABE["blue"], -0.24)]

meta = dict(directive="R23", figure=2, assertions=[], labels=LABELS); rows = []
def check(name, ok, got, want):
    meta["assertions"].append(dict(name=name, passed=bool(ok), got=got, want=want)); print(f"  assert {name}: {'PASS' if ok else 'FAIL'} got={got} want={want}")

U = pd.read_csv(UNITS_CSV); ten = set(U[U.group == "real-relief"].unit)
A = pd.read_csv(R22); A = A[(A.panel == "A") & A.unit.isin(ten) & (A.n_tiles > 0)]
s = A[A.predictor == "s_lr"].set_index("unit"); units = list(s.sort_values("rho", ascending=False).index)
check("9 units in (a)", len(units) == 9, len(units), 9)
piv = A.pivot(index="unit", columns="predictor", values="rho").loc[units]
delta = piv[["GBT_ship", "V2", "P-V1"]].sub(piv["s_lr"], axis=0)
tr = json.load(open(TRIPLES))["D1_ten_C1"]; whole = tr["whole"]; print("pooled 'whole' keys:", list(whole.keys()))
pm = float(whole["mean_rho"]); plo, phi = [float(v) for v in whole["ci95"]]; pseed = whole.get("seed"); npairs = int(tr["n_pairs"])
r9 = json.load(open(R09))["median"]["ten"]["intD1_Afine"]["ref_s_lr"]
check("median s_lr rho = 0.390", round(float(np.median(s.rho)), 3) == 0.390, round(float(np.median(s.rho)), 3), 0.390)
check("frozen CI of the median [0.165, 0.588] (R09 scores_ten, interior D1)", [round(v, 3) for v in r9["ci95"]] == [0.165, 0.588], [round(v, 3) for v in r9["ci95"]], [0.165, 0.588])
check("min s_lr rho = -0.09 at TAG (lu09)", (round(float(s.rho.min()), 2) == -0.09) and (s.rho.idxmin() == "lu09"), (round(float(s.rho.min()), 2), s.rho.idxmin()), (-0.09, "lu09"))
check("max s_lr rho = 0.63", round(float(s.rho.max()), 2) == 0.63, (round(float(s.rho.max()), 2), s.rho.idxmax()), 0.63)
check("pooled row 0.392 [0.317, 0.430], 14 pairs, seed 20261013", (round(pm, 3), round(plo, 3), round(phi, 3), npairs, pseed) == (0.392, 0.317, 0.430, 14, 20261013), (round(pm, 3), round(plo, 3), round(phi, 3), npairs, pseed), (0.392, 0.317, 0.430, 14, 20261013))
check("median GBT_ship rho = 0.370", round(float(np.median(piv["GBT_ship"])), 3) == 0.370, round(float(np.median(piv["GBT_ship"])), 3), 0.370)
check("median V2 - s_lr = +0.028", round(float(np.median(delta["V2"])), 3) == 0.028, round(float(np.median(delta["V2"])), 3), 0.028)
n_above_full = int((delta["P-V1"] > 0).sum()); n_above_2dp = int((piv["P-V1"].round(2) > piv["s_lr"].round(2)).sum())
check("P-V1 above s_lr at exactly 5 of 9 (full precision)", n_above_full == 5, n_above_full, 5)
check("P-V1 above s_lr at exactly 5 of 9 (two-decimal values, as tabulated)", n_above_2dp == 5, n_above_2dp, 5)
meta["pv1_minus_s_lr"] = {u: float(delta.loc[u, "P-V1"]) for u in units}

# ---------------------------------------------------------------- figure
FIG_H = 75.0; fig = S.fig_new(S.WIDTH_MM, FIG_H)
ypos = {u: float(len(units) - i) for i, u in enumerate(units)}; YPOOL = -0.75
axa = S.ax_mm(fig, 36.0, 4.0, 56.0, 62.0); S.style_axes(axa)
axb = S.ax_mm(fig, 108.0, 4.0, 58.0, 62.0); S.style_axes(axb)
for ax in (axa, axb):
    ax.set_ylim(YPOOL - 0.7, len(units) + 0.7); ax.set_yticks([ypos[u] for u in units] + [YPOOL]); S.zero_line(ax, "x")
axa.set_yticklabels([LABELS[u] for u in units] + [f"Pooled ({npairs} pairs)"], fontsize=S.FS)
axb.tick_params(labelleft=False)
# (a)
for u in units:
    r = s.loc[u]; eb = axa.errorbar(r.rho, ypos[u], xerr=[[r.rho - r.ci_lo], [r.ci_hi - r.rho]], fmt="o", color=S.OKABE["black"], markersize=S.MS, markeredgewidth=0.5, elinewidth=0.6, capsize=1.5, capthick=0.5, zorder=3)
    eb.lines[0].set_gid(f"fig2a_{u}")
    rows.append(dict(panel="a", unit=u, label=LABELS[u], series="s_lr rho (interior D1)", x=float(r.rho), ci_lo=float(r.ci_lo), ci_hi=float(r.ci_hi), n_tiles=int(r.n_tiles), source_file=R22, source_column="rho, ci_lo, ci_hi (panel A, predictor s_lr)"))
eb = axa.errorbar(pm, YPOOL, xerr=[[pm - plo], [phi - pm]], fmt="D", color=S.OKABE["black"], markersize=S.MS + 0.5, markeredgewidth=0.5, elinewidth=0.6, capsize=1.5, capthick=0.5, zorder=3); eb.lines[0].set_gid("fig2a_pooled")
rows.append(dict(panel="a", unit="pooled", label=f"Pooled ({npairs} pairs)", series="mean of 14 per-pair rho, unit-stratified block bootstrap", x=pm, ci_lo=plo, ci_hi=phi, n_tiles=None, source_file=TRIPLES, source_column="D1_ten_C1.whole mean_rho, ci95"))
axa.set_xlabel("ρ($s_{\\mathrm{lr}}$)", fontsize=S.FS)
xlo = min(float(s.ci_lo.min()), plo); xhi = max(float(s.ci_hi.max()), phi); axa.set_xlim(np.floor((xlo - 0.05) * 10) / 10, np.ceil((xhi + 0.05) * 10) / 10)
# (b)
hs = []
for key, lab, mk, col, off in MODELS:
    xs = [float(delta.loc[u, key]) for u in units]; ys = [ypos[u] + off for u in units]
    h, = axb.plot(xs, ys, marker=mk, linestyle="none", color=col, markerfacecolor=col, markeredgecolor=col, markersize=S.MS, markeredgewidth=0.5, zorder=3); h.set_gid(f"fig2b_{key}")
    hs.append(Line2D([], [], marker=mk, linestyle="none", color=col, markerfacecolor=col, markersize=S.MS, markeredgewidth=0.5, label=lab))
    for u, x in zip(units, xs):
        rows.append(dict(panel="b", unit=u, label=LABELS[u], series=f"rho({key}) - rho(s_lr)", x=x, ci_lo=None, ci_hi=None, n_tiles=int(s.loc[u].n_tiles), source_file=R22, source_column=f"rho (panel A, predictor {key}) minus rho (panel A, predictor s_lr)"))
axb.set_xlabel("ρ(model) − ρ($s_{\\mathrm{lr}}$)", fontsize=S.FS)
dlo, dhi = float(delta.values.min()), float(delta.values.max()); axb.set_xlim(np.floor((dlo - 0.05) * 10) / 10, np.ceil((dhi + 0.05) * 10) / 10)
leg = axb.legend(handles=hs, loc="lower left", frameon=False, fontsize=S.FS, handletextpad=0.3, borderaxespad=0.2, labelspacing=0.25); leg.set_gid("fig2b_legend")
S.panel_letter(fig, axa, "a", dx_mm=-33.0, dy_mm=0.5); S.panel_letter(fig, axb, "b", dx_mm=-5.0, dy_mm=0.5)
S.finish(fig, 2, font, meta, rows)
