"""R10b §B5 — the pooled row in development, using phase4_score.pooled_row (imported). Pools: the pairs of the ten real-relief
units in the C1 set at D1 and at 125 m, each with and without AR26. Per pool: the whole-pool row; then every unordered triple of
pairs, reusing each pair's 2,000 block resamples drawn once per pool (one default_rng(20261013), pairs in ascending pair-id
order): number of triples, fraction with CI lower bound > 0, min / median / max of the pooled mean, the same for triples whose
pairs come from three different units, and the cross-table of the triple's C1 label against lower bound > 0. Writes triples.json."""
import itertools, json, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "phase4", "scripts"))
import phase4_common as P4, phase4_score as PS
RB, C4 = P4.R9.RB, P4.C4; ROOT = P4.ROOT["rehearsal"]; SI = f"{ROOT}/score_inputs"
def views_for(tag, pairs):
    T = pd.read_parquet(f"{SI}/inputs_{tag}.parquet"); pp = pd.read_csv(f"{ROOT}/scores/per_pair_{tag}.csv").set_index("pair"); units = []
    for pid in pairs:
        t = T[T.pair == pid]; t = t[np.isfinite(t.A_fine.values)]; view = pp.loc[pid, "view"]
        tab, tgt = (t[t.interior.values.astype(bool)], "A_fine") if view == "interior" else (t[np.isfinite(t.A_fine_int.values)], "A_fine_int")
        units.append(dict(pair=pid, unit=C4.UNIT_OF[pid], pred=tab.s_lr.values.astype(float), target=tab[tgt].values.astype(float), blk=PS.blocks_of(tab), c1=bool(pp.loc[pid, "c1_condition"])))
    return units
def label(us): n = sum(u["c1"] for u in us); return "confirmed" if n == len(us) else ("not confirmed" if n == 0 else "partially confirmed")
if __name__ == "__main__":
    t0 = time.time(); pairs_all = list(C4.DEV_PAIRS) + [P4.R8.EW_PAIR]; m, root, rec = P4.start("triples_r10b.py", "rehearsal", pairs_all); ten = [p for u in RB.TEN for p in C4.UNITS[u]]; out = {}
    for tag in ("D1", "D125"):
        pp = pd.read_csv(f"{ROOT}/scores/per_pair_{tag}.csv"); inset = sorted(pp[pp.pair.isin(ten) & pp.status.str.startswith("in C1")].pair)
        for name, pool in ((f"{tag}_ten_C1", inset), (f"{tag}_ten_C1_noAR26", [p for p in inset if p != "AR26__MGDS_31838"])):
            units = views_for(tag, pool); whole, R = PS.pooled_row(units); rows = []
            for tri in itertools.combinations(sorted(units, key=lambda u: u["pair"]), 3):
                row, _ = PS.pooled_row(list(tri), resamples=R); rows.append(dict(pairs=[u["pair"] for u in tri], units=[u["unit"] for u in tri], distinct_units=len({u["unit"] for u in tri}) == 3, mean_rho=row["mean_rho"], ci95=row["ci95"], estimable=row["estimable"], lower_gt0=row["lower_gt0"], n_dropped=row["n_dropped"], label=label(tri)))
            D = pd.DataFrame(rows); D3 = D[D.distinct_units]
            def summ(d): return dict(n=int(len(d)), frac_lower_gt0=float(d.lower_gt0.mean()) if len(d) else None, mean_rho_min=float(d.mean_rho.min()) if len(d) else None, mean_rho_median=float(d.mean_rho.median()) if len(d) else None, mean_rho_max=float(d.mean_rho.max()) if len(d) else None, n_not_estimable=int((~d.estimable).sum()) if len(d) else 0)
            cross = {lab: {"lower_gt0": int(((D.label == lab) & D.lower_gt0).sum()), "lower_le0": int(((D.label == lab) & ~D.lower_gt0).sum())} for lab in ("confirmed", "partially confirmed", "not confirmed")}
            out[name] = dict(pool=pool, n_pairs=len(pool), whole=whole, triples=summ(D), triples_distinct_units=summ(D3), cross_label_vs_lower_gt0=cross, seconds=time.time() - t0)
            print(f"  {name}: {len(pool)} pairs; whole mean {whole['mean_rho']:+.3f} [{whole['ci95'][0]:+.3f}, {whole['ci95'][1]:+.3f}] lower>0 {whole['lower_gt0']}; triples {len(D)}: lower>0 {D.lower_gt0.mean():.3f}, mean rho {D.mean_rho.min():+.2f}/{D.mean_rho.median():+.2f}/{D.mean_rho.max():+.2f}; distinct-unit triples {len(D3)}: lower>0 {D3.lower_gt0.mean():.3f}; cross {cross} ({time.time()-t0:.0f} s)", flush=True)
    json.dump(out, open(f"{ROOT}/triples.json", "w"), indent=1, default=float); print("done")
