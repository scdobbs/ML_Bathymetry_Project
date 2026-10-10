"""R10b (from R10 §A6; the 64-cell row is removed per ruling H1; hr_res and lr_native added for the cells-per-window row; EW0207 predictions from the R10 OAK copy) — assemble the rehearsal scoring inputs from FROZEN tables only (§0.3: held-out predictions of the R08b master
table for the 38 development pairs; EW0207 takes its s_lr from tiles_v1.1 and, once §A4 has run, GBT_ship / P-V1 from the
rehearsal predictions of the frozen artifacts): tile_id, pair, unit, role, group, r0, c0, A_fine, A_fine_int, interior,
s_lr, GBT_ship, P-V1, s_lr_W64, at D1 and at the 125 m distance (R09 tiles_int). Also the joined rehearsal calls table
(R08 calls + key; ab_whole_survey = yes wherever the criteria include a or b: rehearsal convention)."""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "phase4", "scripts"))
import phase4_common as P4
R8, R9, RB, C4 = P4.R8, P4.R9, P4.R9.RB, P4.C4
OUT = f"{P4.ROOT['rehearsal']}/score_inputs"
if __name__ == "__main__":
    pairs = list(C4.DEV_PAIRS) + [R8.EW_PAIR]; m, root, rec = P4.start("rehearsal_inputs_r10b.py", "rehearsal", pairs); os.makedirs(OUT, exist_ok=True)
    R8.oak_verify("R08b/scores", "master_tile.parquet", f"{RB.OUT_SCORES}/master_tile.parquet"); R8.oak_verify("R09/targets_int", "tiles_int.parquet", f"{R9.OUT_INT}/tiles_int.parquet")
    M = pd.read_parquet(f"{RB.OUT_SCORES}/master_tile.parquet")[["tile_id", "pair", "unit", "A_fine", "ref_s_lr", "GBT_ship_A_fine", "P-V1_seedavg"]].rename(columns={"ref_s_lr": "s_lr", "GBT_ship_A_fine": "GBT_ship", "P-V1_seedavg": "P-V1"})
    tiles = R8.load_tiles(); TI = pd.read_parquet(f"{R9.OUT_INT}/tiles_int.parquet").set_index("tile_id")
    ew = tiles[tiles.pair == R8.EW_PAIR][["tile_id", "pair", "unit", "A_fine", "s_lr"]].copy(); ew["GBT_ship"] = np.nan; ew["P-V1"] = np.nan
    pp = f"{P4.R10_OAK}/rehearsal/predictions/{R8.EW_PAIR}.parquet"; R8.oak_verify("R10/rehearsal", f"predictions/{R8.EW_PAIR}.parquet", pp)
    if os.path.exists(pp):
        P = pd.read_parquet(pp).set_index("tile_id"); ew["GBT_ship"] = P.GBT_ship.reindex(ew.tile_id).values; ew["P-V1"] = P["P-V1"].reindex(ew.tile_id).values; src = dict(path=pp, sha256=P4.sha256_file(pp))
    else: src = None
    A = pd.concat([M, ew], ignore_index=True); t = tiles.set_index("tile_id"); A["r0"] = t.r0.reindex(A.tile_id).values; A["c0"] = t.c0.reindex(A.tile_id).values
    A["hr_res"] = t.hr_res.reindex(A.tile_id).values; A["lr_native"] = t.lr_native.reindex(A.tile_id).values; A["role"] = np.where(A.pair == R8.EW_PAIR, "EW0207", "development"); A["group"] = A.unit.map(lambda u: "ten" if u in RB.TEN else ("six" if u in RB.SIX else "EW0207"))
    for tag, fl, fi in (("D1", "interior_D1", "A_fine_int_D1"), ("D125", "interior_D125", "A_fine_int_D125")):
        X = A.copy(); X["interior"] = TI[fl].reindex(X.tile_id).values.astype(bool); X["A_fine_int"] = TI[fi].reindex(X.tile_id).values; X.to_parquet(f"{OUT}/inputs_{tag}.parquet")
    key = {r["code"]: r for r in json.load(open(f"{R8.FIGURES}/key/figset_A_key.json"))["order"]}; calls = pd.read_csv(f"{P4.DR}/R08b/R08_figset_A_calls_steve_2026-10-09.csv").fillna("")
    calls["pair"] = calls.code.map(lambda c: key[c]["pair"]); calls["criteria"] = calls.apply(lambda r: ",".join(k for k, v in (("a", r.a_stripes), ("b", r.b_ribbing), ("c", r.c_halos)) if v == "yes") or "none", axis=1)
    calls["ab_whole_survey"] = calls.criteria.map(lambda c: "yes" if ("a" in c.split(",") or "b" in c.split(",")) else "no"); calls["note"] = calls.call_text
    calls[["code", "pair", "relief", "criteria", "dominant", "ab_whole_survey", "note"]].to_csv(f"{OUT}/calls_rehearsal.csv", index=False)
    stats = {}
    for pid in pairs:
        X = A.copy(); X["interior"] = TI["interior_D1"].reindex(X.tile_id).values.astype(bool); X["A_fine_int"] = TI["A_fine_int_D1"].reindex(X.tile_id).values; x = X[X.pair == pid]
        vi = x[x.interior]; vf = x[np.isfinite(x.A_fine_int)]; stats[pid] = dict(views=dict(interior=dict(amplitude_m=float(vi.A_fine.median()) if len(vi) else np.nan), fallback=dict(amplitude_m=float(vf.A_fine_int.median()) if len(vf) else np.nan)))
    json.dump(stats, open(f"{OUT}/stats_D1.json", "w"), indent=1)
    json.dump(dict(n_rows=int(len(A)), pairs=pairs, ew0207_predictions=src, sources=dict(master="R08b/scores/master_tile.parquet", tiles_int="R09/targets_int/tiles_int.parquet", calls="R08b/R08_figset_A_calls_steve_2026-10-09.csv + R08 key"), ab_convention="ab_whole_survey = yes wherever the R08 criteria include a or b (rehearsal only)"), open(f"{OUT}/inputs_record.json", "w"), indent=1)
    print(f"inputs: {len(A)} rows, {A.pair.nunique()} pairs; EW0207 artifact predictions {'joined' if src else 'NOT yet available (s_lr only)'}; calls {len(calls)} rows", flush=True)
