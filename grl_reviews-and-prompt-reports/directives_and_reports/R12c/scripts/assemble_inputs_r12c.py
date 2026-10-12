"""R12c Part C2 steps 3 and 5 — the mock calls file and the scoring inputs for the mock pairs, from the mock builds and predictions under ROOT:
  calls_mock.csv   one row per page code of the mock figure set (key read: rehearsal, development data), relief yes, criteria none,
                   dominant relief, ab_whole_survey no (the joined form phase4_score reads: pair column added), sha256 recorded;
  inputs_D1.parquet  tile_id, pair, unit, role, r0, c0, hr_res, lr_native, A_fine, A_fine_int, interior, s_lr, GBT_ship, P-V1;
  stats.json       {pair: pair_stats} for the amplitude assert and the no-k status.
Written under <ROOT>/score_inputs/. Development data only; nothing sealed."""
import hashlib, json, os
import numpy as np, pandas as pd
import r12c_common as R
P4 = R.P4; ROOT = R.ROOT; OUT = f"{ROOT}/score_inputs"
if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); key = json.load(open(f"{ROOT}/figures/key/figset_A_key.json"))
    calls = pd.DataFrame([dict(code=o["code"], pair=o["pair"], relief="yes", criteria="none", dominant="relief", ab_whole_survey="no", note="R12c mock call (rehearsal scaffolding)") for o in key["order"]])
    cp = f"{OUT}/calls_mock.csv"; calls.to_csv(cp, index=False); rows = []; stats = {}
    for pid, spec in R.MOCK_ROWS.items():
        st = json.load(open(f"{ROOT}/build/{pid}/pair_stats.json")); stats[pid] = st
        if st["status"] != "built": continue
        t = pd.read_parquet(f"{ROOT}/build/{pid}/tiles_v1_1.parquet").set_index("tile_id"); ti = pd.read_parquet(f"{ROOT}/build/{pid}/tiles_int.parquet").set_index("tile_id"); P = pd.read_parquet(f"{ROOT}/predictions/{pid}.parquet").set_index("tile_id")
        X = pd.DataFrame(dict(tile_id=t.index, pair=pid, unit=t.unit.values, role=spec["style"], r0=t.r0.values, c0=t.c0.values, hr_res=t.hr_res.values, lr_native=t.lr_native.values, A_fine=t.A_fine.values, A_fine_int=ti.A_fine_int_D1.reindex(t.index).values,
                              interior=ti.interior_D1.reindex(t.index).values.astype(bool), s_lr=P.s_lr.reindex(t.index).values, GBT_ship=P.GBT_ship.reindex(t.index).values)); X["P-V1"] = P["P-V1"].reindex(t.index).values; rows.append(X)
    A = pd.concat(rows, ignore_index=True); ip = f"{OUT}/inputs_D1.parquet"; A.to_parquet(ip); sp = f"{OUT}/stats.json"; json.dump(stats, open(sp, "w"), indent=1, default=float)
    rec = dict(calls=dict(path=cp, sha256=hashlib.sha256(open(cp, "rb").read()).hexdigest(), n_rows=len(calls), codes=list(calls.code)), inputs=dict(path=ip, sha256=hashlib.sha256(open(ip, "rb").read()).hexdigest(), n_tiles=int(len(A)), pairs={p: int((A.pair == p).sum()) for p in A.pair.unique()}), stats=dict(path=sp, statuses={p: s["status"] for p, s in stats.items()}))
    json.dump(rec, open(f"{OUT}/record.json", "w"), indent=1); print(json.dumps(rec, indent=1))
