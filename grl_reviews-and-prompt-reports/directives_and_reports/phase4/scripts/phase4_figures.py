"""Phase 4 figure set A (pre-registration v2, B3 step 3) for the listed pairs, through the R08 as-run implementation
fields_r08.one_pair UNCHANGED (sha256-verified in phase4_common): hillshade of HR and F_hr at +/- 3 RMS, 500 m bar, no
names, scores or amplitudes; pages shuffled with numpy.random.default_rng(20261011) and coded Q01.., key written to a
separate file (<root>/figures/key/figset_A_key.json), the blind PDF to <root>/figures/figset_A_blind.pdf.
Thin wrapper (no frozen file is edited): fields_r08 reads the field mosaics and writes pages through module-level roots
(R8.R04_FIELDS, R8.OUT_FIELDS, R8.FIGURES), re-pointed to <root>; its one_pair also renders a labelled set-B page, which
needs a prediction column: it is given an all-NaN table, and the set-B pages and the metrics are left in
<root>/figures/work/ (never assembled, never delivered). C4.UNIT_OF is extended for pairs outside the development list."""
from __future__ import annotations
import json, os, sys, time
import numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase4_common as P4
import r04_common as C4, fields_r08 as FR

if __name__ == "__main__":
    ap = P4.argparser("Phase 4 figure set A"); a = ap.parse_args(); pairs = sorted(P4.split_pairs(a.pairs))
    m, root, rec = P4.start("phase4_figures.py", a.mode, pairs); t0 = time.time()
    for pid in pairs: C4.UNIT_OF.setdefault(pid, P4.unit_of(pid, m))
    FR.R8.R04_FIELDS = f"{root}/fields"; FR.R8.OUT_FIELDS = f"{root}/figures/work"; os.makedirs(f"{root}/figures/key", exist_ok=True); os.makedirs(FR.R8.OUT_FIELDS, exist_ok=True)
    tiles = pd.concat([pd.read_parquet(f"{root}/build/{pid}/tiles_v1_1.parquet") for pid in pairs], ignore_index=True)
    rng = np.random.default_rng(P4.FIG_SEED[a.mode]); order = [pairs[i] for i in rng.permutation(len(pairs))]; code = {p: f"{P4.FIG_CODE[a.mode]}{i+1:02d}" for i, p in enumerate(order)}
    key = dict(mode=a.mode, seed=P4.FIG_SEED[a.mode], order=[dict(code=code[p], pair=p, unit=C4.UNIT_OF[p]) for p in order]); json.dump(key, open(f"{root}/figures/key/figset_A_key.json", "w"), indent=1)
    FR.D.update(m=m, tiles=tiles, pred=pd.DataFrame(dict(tile_id=tiles.tile_id.values, GBT_ship_A_fine=np.nan)), code=code)
    with Pool(min(a.workers, 4)) as pool: mets = pool.map(FR.one_pair, pairs, chunksize=1)
    by = {r["pair"]: r for r in mets}; FR.assemble_pdf([by[p]["page_A"] for p in order], f"{root}/figures/figset_A_blind.pdf")
    json.dump([{k: v for k, v in r.items() if k in ("pair", "page_A", "seconds", "window", "hillshade")} for r in mets], open(f"{root}/figures/work/pages_record.json", "w"), indent=1, default=str)
    print(f"figure set A: {len(pairs)} pages -> {root}/figures/figset_A_blind.pdf (key in figures/key/); set-B pages and metrics left in figures/work/, not delivered ({time.time()-t0:.0f} s)", flush=True)
