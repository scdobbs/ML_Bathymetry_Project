"""R01b Task 2.2 — meters transfer regression, EW0207 excluded and included, N=7/8 native and 14/15 expanded.
Reuses review_stats.py Part 1 machinery verbatim (frozen dir44 task3 conventions), reading the RESTORED patch set."""
import sys, json
sys.path.insert(0, "/scratch/users/scdobbs/grl_review"); import review_stats as rs
sys.path.insert(0, "/scratch/users/scdobbs/dir44"); import task3_covariates as t3
rs.PATCHES = "/scratch/users/scdobbs/grl_review/R01/patches_restored/patches_native_phase2"; t3.PATCHES = rs.PATCHES
import numpy as np
out = {}
by_all = rs.collect_with_std([p for v in {**rs.NATIVE_UNITS, **rs.NEW_UNITS}.values() for p in v])
for excl in (True, False):
    nat = {u: v for u, v in rs.NATIVE_UNITS.items() if not (excl and u == "EW0207")}
    for tag, units in (("native", nat), ("expanded", {**nat, **rs.NEW_UNITS})):
        bu = rs.merge_units(by_all, units)
        for meters in (False, True):
            key = f"{tag}{len(bu)}/{'exclEW0207' if excl else 'inclEW0207'}/{'meters' if meters else 'normalized'}"
            res = rs.louo(bu, "base", meters); out[key] = res; s = res["summary"]
            print(f"[{key}] pooled={res['pooled_r2']:.3f} | R2raw mean/med {s['r2_raw']['mean']:.2f}/{s['r2_raw']['median']:.2f} | R2dm {s['r2_demeaned']['mean']:.2f}/{s['r2_demeaned']['median']:.2f} | R2dm+scale(r^2) {s['r2_demeaned_rescaled']['mean']:.2f}/{s['r2_demeaned_rescaled']['median']:.2f} | r {s['pearson']['mean']:.2f}/{s['pearson']['median']:.2f} ({s['pearson']['n_positive']}/{res['n_units']}) | rho {s['spearman']['mean']:.2f}/{s['spearman']['median']:.2f} ({s['spearman']['n_positive']}/{res['n_units']})", flush=True)
json.dump(out, open("/scratch/users/scdobbs/grl_review/R01b/task2_transfer_meters.json", "w"), indent=1, default=float)
