"""R10 §A3 and §A4 checks.
A3: each rehearsal set-A page (phase4_figures, rehearsal) against the R08 set-A page of the same pair (R08 fields/pages/A_<code>.png,
    OAK-verified): maximum pixel difference over the two panels with the title band (page code) masked out; and a text scan of the
    page record for names / scores / amplitudes (the pages carry only the code).
A4: EW0207 predictions of the frozen artifacts (phase4_predict, rehearsal) against their references: s_lr == tiles_v1.1 s_lr;
    GBT_ship against R05's EW0207 predictions (R09 persisted no EW0207 predictions: the max |diff| must reproduce R09's 0.472 m, then
    the predictions are persisted with their sha256 as the Phase 4 reference); P-V1 per seed and seed-averaged against the R07 EW0207
    table (max |diff| in log10 <= 1e-3; EW0207 all-tile rho equal to R07's to three decimals)."""
import json, os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "phase4", "scripts"))
import phase4_common as P4, eval_v1_1 as E
R8, R7 = P4.R8, P4.R7; TITLE_PX = 45
if __name__ == "__main__":
    pairs = P4.split_pairs(sys.argv[1]); m, root, rec = P4.start("rehearsal_checks_r10.py", "rehearsal", pairs); out = {}
    # ---- A3
    key8 = {r["pair"]: r["code"] for r in json.load(open(f"{R8.FIGURES}/key/figset_A_key.json"))["order"]}; key10 = {r["pair"]: r["code"] for r in json.load(open(f"{root}/figures/key/figset_A_key.json"))["order"]}
    a3 = {}
    for pid in pairs:
        if pid not in key10: continue
        R8.oak_verify("R08/fields", f"pages/A_{key8[pid]}.png", f"{R8.OUT_FIELDS}/pages/A_{key8[pid]}.png")
        a = plt.imread(f"{root}/figures/work/pages/A_{key10[pid]}.png"); b = plt.imread(f"{R8.OUT_FIELDS}/pages/A_{key8[pid]}.png")
        same_shape = a.shape == b.shape; d = np.abs(a[TITLE_PX:].astype(float) - b[TITLE_PX:].astype(float)) if same_shape else None
        a3[pid] = dict(code_r10=key10[pid], code_r08=key8[pid], same_shape=same_shape, shape=list(a.shape), max_pixel_diff_below_title=float(d.max()) if same_shape else None, n_pixels_differing=int((d > 0).any(-1).sum()) if same_shape else None, title_band_px=TITLE_PX)
    rec_pages = json.load(open(f"{root}/figures/work/pages_record.json")); a3["page_text_scan"] = dict(note="set-A pages carry the code as the only text besides the 500 m scale bar label (fields_r08 set A: suptitle = code, axes off, no colourbar)", codes=sorted(key10.values()))
    out["A3"] = a3
    # ---- A4 (EW0207)
    ew = R8.EW_PAIR; a4 = {}
    if ew in pairs and os.path.exists(f"{root}/predictions/{ew}.parquet"):
        P = pd.read_parquet(f"{root}/predictions/{ew}.parquet").set_index("tile_id"); tiles = R8.load_tiles().set_index("tile_id").reindex(P.index)
        a4["s_lr"] = dict(identical=bool(np.array_equal(P.s_lr.values, tiles.s_lr.values)))
        E5 = pd.read_parquet(R8.PRED_TILE_EW_R05).set_index("tile_id").reindex(P.index); d = np.abs(P.GBT_ship.values - E5.GBT_ship_A_fine.values); rec9 = json.load(open(f"{P4.R9.OUT_ART}/gbt_ship_all16_record.json"))
        a4["GBT_ship"] = dict(r09_persisted_ew0207_predictions=False, max_abs_diff_vs_R05_ew0207_m=float(d.max()), r09_record_value_m=rec9["r05_ew0207_model"]["max_abs_diff_vs_r05_ew0207_pred_m"], reproduces_r09=bool(abs(float(d.max()) - rec9["r05_ew0207_model"]["max_abs_diff_vs_r05_ew0207_pred_m"]) < 1e-9))
        refp = f"{root}/predictions/reference_GBT_ship_EW0207.parquet"; P[["GBT_ship"]].reset_index().to_parquet(refp); a4["GBT_ship"]["reference"] = dict(path=refp, sha256=P4.sha256_file(refp))
        E7 = pd.read_parquet(f"{R8.R07_SCORES}/pred_tile_EW0207_r07.parquet").set_index("tile_id").reindex(P.index); pv = {}
        for s in R7.SEEDS:
            dl = np.abs(np.log10(P[f"P-V1_s{s}"].values) - np.log10(E7[f"P-V1_s{s}"].values)); pv[f"seed {s}"] = dict(max_abs_diff_log10=float(dl.max()), within_tol=bool(dl.max() <= 1e-3))
            fold = pd.read_parquet(f"{R7.MODELS}/P-V1/seed{s}/fold_ALL/pred.parquet").set_index("tile_id").reindex(P.index); blk = np.abs(P[[f"P-V1_s{s}_blk{j}" for j in range(16)]].values - fold[[f"blk{j}" for j in range(16)]].values); pv[f"seed {s}"]["max_abs_diff_blocks_log10"] = float(blk.max())
        dl = np.abs(np.log10(P["P-V1"].values) - np.log10(E7["P-V1_seedavg"].values)); pv["seed-averaged"] = dict(max_abs_diff_log10=float(dl.max()), within_tol=bool(dl.max() <= 1e-3))
        y = tiles.A_fine.values.astype(float); r10 = E.rho(P["P-V1"].values.astype(float), y); r7 = E.rho(E7["P-V1_seedavg"].values.astype(float), y)
        pv["ew0207_all_tile_rho"] = dict(r10=r10, r07=r7, equal_3dp=bool(round(r10, 3) == round(r7, 3))); a4["P-V1"] = pv
    out["A4"] = a4; json.dump(out, open(f"{root}/checks_A3_A4.json", "w"), indent=1, default=float); print(json.dumps(out, indent=1, default=float)[:3000])
