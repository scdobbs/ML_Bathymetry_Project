"""R16 Part C2 step 5 (scaffolding, as R12c / R13) — the scoring inputs phase5_score.py reads, from the mock builds and predictions
under ROOT: inputs_D1.parquet (tile_id, pair, unit, role = phase5, r0, c0, hr_res, lr_native, A_fine, A_fine_int (D1), interior (D1),
s_lr, GBT_ship, P-V1) for the built (eligible) sites, and stats.json ({pair: pair_stats}). The not-testable site has neither (its
build directory does not exist). No calls are joined here (phase5_score.py joins the code-keyed file itself). Development data only."""
import hashlib, json, os
import numpy as np, pandas as pd
import r16_common as R
ROOT = R.ROOT; OUT = f"{ROOT}/score_inputs"
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); rows = []; stats = {}
    for pid in R.ELIGIBLE_MOCK:
        st = json.load(open(f"{ROOT}/build/{pid}/pair_stats.json")); stats[pid] = st
        if st["status"] != "built": continue
        t = pd.read_parquet(f"{ROOT}/build/{pid}/tiles_v1_1.parquet").set_index("tile_id"); ti = pd.read_parquet(f"{ROOT}/build/{pid}/tiles_int.parquet").set_index("tile_id"); P = pd.read_parquet(f"{ROOT}/predictions/{pid}.parquet").set_index("tile_id")
        assert set(P.index) == set(t.index), pid
        X = pd.DataFrame(dict(tile_id=t.index, pair=pid, unit=t.unit.values, role="phase5", r0=t.r0.values, c0=t.c0.values, hr_res=t.hr_res.values, lr_native=t.lr_native.values, A_fine=t.A_fine.values, A_fine_int=ti.A_fine_int_D1.reindex(t.index).values,
                              interior=ti.interior_D1.reindex(t.index).values.astype(bool), s_lr=P.s_lr.reindex(t.index).values, GBT_ship=P.GBT_ship.reindex(t.index).values)); X["P-V1"] = P["P-V1"].reindex(t.index).values; rows.append(X)
    A = pd.concat(rows, ignore_index=True); ip = f"{OUT}/inputs_D1.parquet"; A.to_parquet(ip); sp = f"{OUT}/stats.json"; json.dump(stats, open(sp, "w"), indent=1, default=float)
    rec = dict(inputs=dict(path=ip, sha256=sha(ip), n_tiles=int(len(A)), pairs={p: int((A.pair == p).sum()) for p in A.pair.unique()}), stats=dict(path=sp, statuses={p: s["status"] for p, s in stats.items()}),
               not_testable_build_dir_exists=os.path.exists(f"{ROOT}/build/{R.NOT_TESTABLE_MOCK}"))
    json.dump(rec, open(f"{OUT}/record.json", "w"), indent=1); print(json.dumps(rec, indent=1))
