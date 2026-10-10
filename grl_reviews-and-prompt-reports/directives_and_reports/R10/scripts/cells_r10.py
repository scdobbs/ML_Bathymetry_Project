"""R10 §A6.5 — ship cells per 256-px window: cells_across = 256 x hr_res / lr_native for every development pair and EW0207
(from tiles_v1.1), beside the pair's interior-tile rho for s_lr (R10 per-pair rehearsal scores at D1); for the five sealed
pairs from the MANIFEST ROW ONLY (hr_native_res_m, lr_native_res_m) -> COULD NOT CHECK where absent. Rule: s_lr_W64 is in
force if any confirmatory pair's cells_across is below the development minimum; HOLD if a confirmatory value could not be checked."""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "phase4", "scripts"))
import phase4_common as P4
R8, C4 = P4.R8, P4.C4
if __name__ == "__main__":
    m0 = C4.load_manifest(); P4.R9.install_guard_r09(m0); tiles = R8.load_tiles(); out = f"{P4.ROOT['rehearsal']}/cells"; os.makedirs(out, exist_ok=True)
    g = tiles.groupby("pair").agg(unit=("unit", "first"), hr_res=("hr_res", "first"), lr_native=("lr_native", "first")).reset_index(); g["cells_across"] = 256 * g.hr_res / g.lr_native
    pp = f"{P4.ROOT['rehearsal']}/scores/per_pair_D1.csv"
    if os.path.exists(pp):
        s = pd.read_csv(pp).set_index("pair"); g["rho_s_lr_D1"] = s.rho_s_lr.reindex(g.pair).values; g["view_D1"] = s.view.reindex(g.pair).values; g["rho_s_lr_W64_D1"] = s.rho_s_lr_W64.reindex(g.pair).values if "rho_s_lr_W64" in s else np.nan
    dev_min = float(g[g.pair != R8.EW_PAIR].cells_across.min()); g.to_csv(f"{out}/cells_development.csv", index=False)
    sealed = []
    for pid in P4.SEALED5:
        row = m0[m0.pair_id == pid].iloc[0]; hr, lr = row["hr_native_res_m"], row["lr_native_res_m"]; ok = isinstance(hr, (int, float)) and isinstance(lr, (int, float)) and np.isfinite(hr) and np.isfinite(lr)
        sealed.append(dict(pair=pid, role="confirmatory" if pid in P4.CONFIRMATORY else "lockbox", hr_native_res_m=float(hr) if ok else None, lr_native_res_m=float(lr) if isinstance(lr, (int, float)) and np.isfinite(lr) else None, cells_across=float(256 * hr / lr) if ok else None, status="from manifest row" if ok else "COULD NOT CHECK (manifest row holds no HR grid spacing)"))
    conf = [s for s in sealed if s["role"] == "confirmatory"]; unknown = [s["pair"] for s in conf if s["cells_across"] is None]; below = [s["pair"] for s in conf if s["cells_across"] is not None and s["cells_across"] < dev_min]
    rule = dict(dev_min_cells_across=dev_min, dev_min_pair=g.loc[g[g.pair != R8.EW_PAIR].cells_across.idxmin(), "pair"], confirmatory_below_dev_min=below, confirmatory_could_not_check=unknown,
                in_force=(len(below) > 0), hold=(len(unknown) > 0 and len(below) == 0), options_if_hold=["include s_lr_W64 regardless", "Steve supplies the HR grid spacing from the dataset's public record"])
    json.dump(dict(development=g.to_dict("records"), sealed=sealed, rule=rule), open(f"{out}/cells.json", "w"), indent=1, default=float); print(json.dumps(dict(dev_min=dev_min, sealed=sealed, rule=rule), indent=1, default=float))
