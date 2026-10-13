"""Phase 5 figure set A (pre-registration v3, step 2, blind pages): a parameterised copy of phase4_figures.py, through the R08
as-run implementation fields_r08.one_pair UNCHANGED (sha256-verified): hillshade of HR and F_hr at +/- 3 RMS, 500 m bar, no
names, scores or amplitudes; pages shuffled with numpy.random.default_rng(20261011) and coded Q01.., key written to a separate
file (<root>/figures/key/figset_A_key.json; the calls stay keyed by Q-code, §B5), the blind PDF to <root>/figures/figset_A_blind.pdf.
Changes from Phase 4: phase5_common.start; a not-testable row (eligible = False, §B1) gets no page and is listed in the key's
`no_page` with its status, read from its supplement row (no file of that site is opened). Set-B pages and metrics stay in
<root>/figures/work/ (never assembled, never delivered), as in Phase 4."""
from __future__ import annotations
import json, os, sys, time
import numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase5_common as P5
import r04_common as C4, fields_r08 as FR

if __name__ == "__main__":
    ap = P5.argparser("Phase 5 figure set A"); a = ap.parse_args(); pairs = sorted(P5.split_pairs(a.pairs))
    m, root, rec = P5.start("phase5_figures.py", a.mode, pairs); t0 = time.time()
    nt = {pid: st for pid in pairs if (st := P5.row_status(C4.pair_row(m, pid))).startswith(P5.NOT_TESTABLE)}; pairs = [p for p in pairs if p not in nt]   # §B1: no page for a not-testable site
    for pid in pairs: C4.UNIT_OF.setdefault(pid, P5.unit_of(pid, m))
    skipped = [pid for pid in pairs if json.load(open(f"{root}/build/{pid}/pair_stats.json")).get("status") == P5.P4.NO_K]; pairs = [p for p in pairs if p not in skipped]   # no page for a no-k pair
    FR.R8.R04_FIELDS = f"{root}/fields"; FR.R8.OUT_FIELDS = f"{root}/figures/work"; os.makedirs(f"{root}/figures/key", exist_ok=True); os.makedirs(FR.R8.OUT_FIELDS, exist_ok=True)
    tiles = pd.concat([pd.read_parquet(f"{root}/build/{pid}/tiles_v1_1.parquet") for pid in pairs], ignore_index=True)
    rng = np.random.default_rng(P5.P4.FIG_SEED["phase4"]); order = [pairs[i] for i in rng.permutation(len(pairs))]; code = {p: f"{P5.P4.FIG_CODE['phase4']}{i+1:02d}" for i, p in enumerate(order)}
    key = dict(mode=a.mode, seed=P5.P4.FIG_SEED["phase4"], order=[dict(code=code[p], pair=p, unit=C4.UNIT_OF[p]) for p in order], no_page=[dict(pair=p, status=P5.P4.NO_K) for p in skipped] + [dict(pair=p, status=s) for p, s in nt.items()]); json.dump(key, open(f"{root}/figures/key/figset_A_key.json", "w"), indent=1)
    FR.D.update(m=m, tiles=tiles, pred=pd.DataFrame(dict(tile_id=tiles.tile_id.values, GBT_ship_A_fine=np.nan)), code=code)
    with Pool(min(a.workers, 4)) as pool: mets = pool.map(FR.one_pair, pairs, chunksize=1)
    by = {r["pair"]: r for r in mets}; FR.assemble_pdf([by[p]["page_A"] for p in order], f"{root}/figures/figset_A_blind.pdf")
    json.dump([{k: v for k, v in r.items() if k in ("pair", "page_A", "seconds", "window", "hillshade")} for r in mets], open(f"{root}/figures/work/pages_record.json", "w"), indent=1, default=str)
    print(f"figure set A: {len(pairs)} pages ({len(skipped)} no-k pairs and {len(nt)} not-testable sites without a page) -> {root}/figures/figset_A_blind.pdf (key in figures/key/); set-B pages and metrics left in figures/work/, not delivered ({time.time()-t0:.0f} s)", flush=True)
