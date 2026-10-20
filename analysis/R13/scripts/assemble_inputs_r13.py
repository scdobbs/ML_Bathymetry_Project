"""R13 Part D scaffolding (after the calls are committed and the predictions exist) — the three inputs phase4_score.py requires, assembled
from the Phase 4 build and prediction outputs under grl_review_archive/phase4 (read through phase4_common.start in phase4 mode: guard,
every lock, allow-list; a provenance record is written):
  score_inputs/inputs_D1.parquet   tile_id, pair, unit, role (confirmatory | lockbox), r0, c0, hr_res, lr_native, A_fine, A_fine_int (D1),
                                   interior (D1), s_lr, GBT_ship, P-V1 — from build/<pair>/tiles_v1_1.parquet, tiles_int.parquet and
                                   predictions/<pair>.parquet (the no-k pair has no rows);
  score_inputs/phase4_calls_joined.csv   Steve's committed calls (R13/phase4_calls.csv, sha256 asserted = ae791021...) joined to the pairs
                                   through figures/key/figset_A_key.json (code -> pair); every column of Steve's file carried unchanged,
                                   `pair` added. Its sha256 is printed so that it can be recorded in the ledger (the calls lock of
                                   phase4_score.py hashes the file it is given);
  score_inputs/stats.json          {pair: pair_stats} of the five pairs (amplitude assert in score_pair; no-k status).
Written under grl_review_archive/phase4/score_inputs/. No value is printed."""
import hashlib, json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); DR = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, f"{DR}/phase4/scripts")
import phase4_common as P4
CALLS = f"{DR}/R13/phase4_calls.csv"; CALLS_SHA = "ae7910210b75d341539642213585652f72b2459d7787cce2bf8c1abc3a06a55f"
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
if __name__ == "__main__":
    assert sha(CALLS) == CALLS_SHA, sha(CALLS)
    pairs = list(P4.SEALED5); m, root, rec = P4.start("assemble_inputs_r13.py", "phase4", pairs); out = f"{root}/score_inputs"; os.makedirs(out, exist_ok=True)
    key = json.load(open(f"{root}/figures/key/figset_A_key.json")); code2pair = {o["code"]: o["pair"] for o in key["order"]}
    calls = pd.read_csv(CALLS, dtype=str, keep_default_na=False); assert list(calls.columns) == ["code", "relief", "criteria", "dominant", "ab_whole_survey", "note"], list(calls.columns)
    assert set(calls.code) == set(code2pair), (sorted(calls.code), sorted(code2pair))
    joined = calls.copy(); joined.insert(1, "pair", joined.code.map(code2pair)); jp = f"{out}/phase4_calls_joined.csv"; joined.to_csv(jp, index=False)
    back = pd.read_csv(jp, dtype=str, keep_default_na=False); assert back.drop(columns=["pair"]).equals(calls), "joined file does not carry Steve's values unchanged"
    rows = []; stats = {}
    for pid in pairs:
        st = json.load(open(f"{root}/build/{pid}/pair_stats.json")); stats[pid] = st; role = "confirmatory" if pid in P4.CONFIRMATORY else "lockbox"
        if st["status"] != "built": continue
        t = pd.read_parquet(f"{root}/build/{pid}/tiles_v1_1.parquet").set_index("tile_id"); ti = pd.read_parquet(f"{root}/build/{pid}/tiles_int.parquet").set_index("tile_id"); P = pd.read_parquet(f"{root}/predictions/{pid}.parquet").set_index("tile_id")
        assert set(P.index) == set(t.index), pid
        fin = {c: int(np.isfinite(P[c].values.astype(float)).sum()) for c in ("s_lr", "GBT_ship", "P-V1")}; finite_rec = globals().setdefault("FINITE", {}); finite_rec[pid] = dict(n_tiles=int(len(P)), finite=fin, all_finite=bool(all(v == len(P) for v in fin.values())))
        X = pd.DataFrame(dict(tile_id=t.index, pair=pid, unit=t.unit.values, role=role, r0=t.r0.values, c0=t.c0.values, hr_res=t.hr_res.values, lr_native=t.lr_native.values, A_fine=t.A_fine.values, A_fine_int=ti.A_fine_int_D1.reindex(t.index).values,
                              interior=ti.interior_D1.reindex(t.index).values.astype(bool), s_lr=P.s_lr.reindex(t.index).values, GBT_ship=P.GBT_ship.reindex(t.index).values)); X["P-V1"] = P["P-V1"].reindex(t.index).values; rows.append(X)
    A = pd.concat(rows, ignore_index=True); ip = f"{out}/inputs_D1.parquet"; A.to_parquet(ip); sp = f"{out}/stats.json"; json.dump(stats, open(sp, "w"), indent=1, default=float)
    r = dict(calls=dict(path=CALLS, sha256=CALLS_SHA), joined_calls=dict(path=jp, sha256=sha(jp), n_rows=int(len(joined)), codes=list(joined.code), pairs=list(joined.pair)), key=dict(path=f"{root}/figures/key/figset_A_key.json", sha256=sha(f"{root}/figures/key/figset_A_key.json")),
             inputs=dict(path=ip, sha256=sha(ip), n_tiles=int(len(A)), pairs={p: int((A.pair == p).sum()) for p in A.pair.unique()}, columns=list(A.columns)), stats=dict(path=sp, sha256=sha(sp), statuses={p: s["status"] for p, s in stats.items()}), provenance_record=rec.get("time"),
             predictions_finite=globals().get("FINITE", {}), predictions_absent_for_no_k=(not os.path.exists(f"{root}/predictions/KN182L03__MGDS_30193.parquet")))
    json.dump(r, open(f"{out}/record.json", "w"), indent=1); print(json.dumps(r, indent=1))
