"""Phase 4 build (pre-registration v2, B3 step 2) for a list of pair ids, through the FROZEN builders unchanged:
  1. R04 build_tiles_targets.process_pair: windows (R02 rules, stride 128), tiles_v1, the band-passed fields F_hr / F_res,
     bicubic_m and joint_v1 mosaics (written under <root>/fields/<pair>/), tile and block targets (targets_v1);
  2. R05 tiles_v1.1 selection (tile kept iff >= 50 % of its central 128 px are joint-valid; blocks of dropped tiles dropped);
  3. R04 build_features (features_v1 at 128 / 224 px for every tile centre and kept block centre) + the driver's META columns;
  4. R09 targets_v1_int: the distance map, the R08b interior-tile flag at D1 and A_fine_int at D1 (>= 2,048 interior px);
  5. R06 build_inputs_r06.process_pair: the P-V1 input arrays (x, lrv, cond; products not needed for V1) under <root>/inputs/<pair>/;
  6. the B4 amplitude statistic per view (interior tiles / fallback), with the non-empty 4 x 4-tile block counts, written BEFORE any prediction.
Thin wrappers (no frozen file is edited): the R04 / R06 builders write to module-level roots (FIELDS, OUT, R.FIELDS) which are
re-pointed to <root>; C4.UNIT_OF is extended with the pair's leakage unit for pairs outside the frozen development list
(the sealed pairs at Phase 4). A manifest row without lr_native_res_m, max_recoverable_k or the harmonized paths is refused.
Product features (R11 §A1): the frozen builder's rule is applied unchanged — features_v1.pair_context / contract_features read the row's
`ship_products_usable` through r04_common.ship_usable (true / 1 / "true"); when it is not true the 14 contract columns are NaN. The
decision is recorded in pair_stats (ship_products_usable, product_features = used | masked) and, in phase4 mode, pair_stats also carries
the k provenance of the acquisition run record (k, its source, the per-k verdicts; §A3)."""
from __future__ import annotations
import json, os, sys, time
import numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase4_common as P4
import r04_common as C4, r05_common as C5, r06_common as R6, targets_v1 as T, targets_v1_int as TI
import build_tiles_targets as BTT, build_features as BF, build_inputs_r06 as BI, bar_v1_1 as B

def require_row(row, pid):
    """§A4: a row without spacings or harmonized paths is refused; a row with them but no max_recoverable_k is NO_K (nothing is built)."""
    st = P4.row_status(row)
    if st == "refused": raise P4.Refusal(f"manifest row of {pid} lacks a spacing or a harmonized path: Phase 4 step 1 (acquisition repo) must fill them first")
    return st

def block_stats(tiles, flag):
    t = tiles[flag]; n = len(t); nb = int(len({(int(r0) // T.STRIDE // 4, int(c0) // T.STRIDE // 4) for r0, c0 in zip(t.r0, t.c0)})) if n else 0
    return n, nb

def build_pair(m, pid, root, workers, log=print, acq_rec=None):
    t0 = time.time(); row = C4.pair_row(m, pid); st = require_row(row, pid); unit = P4.unit_of(pid, m); C4.UNIT_OF.setdefault(pid, unit); kp = P4.k_provenance(pid, row, acq_rec)
    out = f"{root}/build/{pid}"; os.makedirs(out, exist_ok=True); os.makedirs(f"{root}/fields", exist_ok=True)
    if st == P4.NO_K:
        stats = dict(pair=pid, unit=unit, status=P4.NO_K, lr_native=float(row["lr_native_res_m"]), ship_products_usable=C4.ship_usable(row), product_features="not built", k_provenance=kp, note="no recoverable k in the manifest row: no fine band; nothing built, no figure page, outside C1 (pre-registration v2 B4 rule 0)", seconds=time.time() - t0)
        json.dump(stats, open(f"{out}/pair_stats.json", "w"), indent=1, default=float); log(f"  {pid} ({unit}): {P4.NO_K}; nothing built", flush=True); return stats
    # 1. R04 windows / fields / targets_v1
    BTT.FIELDS = f"{root}/fields"; tiles1, blocks1, summ, chk = BTT.process_pair(m, pid, workers)
    # 2. R05 selection
    keep = tiles1.frac_joint >= C5.TILE_FLOOR; tiles = tiles1[keep].reset_index(drop=True); blocks = blocks1[blocks1.tile_id.isin(tiles.tile_id)].reset_index(drop=True)
    tiles.to_parquet(f"{out}/tiles_v1_1.parquet"); blocks.to_parquet(f"{out}/blocks_v1_1.parquet"); tiles1.to_parquet(f"{out}/tiles_v1.parquet")
    # 3. R04 features (+ META as in build_features.__main__)
    ctx, row2, g = BF.build_ctx(m, pid); BF.CTX["ctx"] = ctx; J = BF.jobs_for(tiles, blocks, pid)
    with Pool(workers) as pool: frows = pool.map(BF.work, J, chunksize=16)
    meta = dict(unit=unit, pair=pid, lr_native=float(row["lr_native_res_m"]), k=int(row["max_recoverable_k"]), hr_res=float(g["hr_res"]), ship_usable=C4.ship_usable(row)); assert bool(ctx["usable"]) == meta["ship_usable"], (pid, ctx["usable"], meta["ship_usable"])
    s256 = tiles.set_index("tile_id").s_lr
    for o in frows: o.update(meta); o["s_lr_256"] = float(s256[o["tile_id"]]); o["depth"] = o["mu_lr_128"]; o["log_depth_over_lr"] = float(np.log10(abs(o["mu_lr_128"]) / meta["lr_native"])) if np.isfinite(o["mu_lr_128"]) and o["mu_lr_128"] != 0 else np.nan
    feats = pd.DataFrame(frows); feats.to_parquet(f"{out}/features.parquet"); del BF.CTX["ctx"], ctx, g
    # 4. targets_v1_int at D1 + interior-tile flag (R08b rule)
    import rasterio
    with rasterio.open(f"{root}/fields/{pid}/F_hr.tif") as s: F = s.read(1); hr_res = float(abs(s.transform.a))
    with rasterio.open(f"{root}/fields/{pid}/joint_v1.tif") as s: Jm = (s.read(1) == 1) & np.isfinite(F)
    D1 = TI.d_px(float(row["lr_native_res_m"]), hr_res); dist = TI.distance_map(Jm); rows = []
    for r in tiles.itertuples():
        sl = (slice(r.tr0, r.tr0 + TI.TILE), slice(r.tc0, r.tc0 + TI.TILE)); j, d, f = Jm[sl], dist[sl], F[sl]
        i1, f1 = TI.interior_tile(j, d, D1); a1, n1 = TI.afine_int(f, j, d, D1)
        rows.append(dict(tile_id=r.tile_id, pair=pid, unit=unit, r0=int(r.r0), c0=int(r.c0), A_fine=r.A_fine, s_lr=r.s_lr, n_joint=int(j.sum()), frac_ge_D1=f1, interior_D1=i1, n_int_px_D1=n1, A_fine_int_D1=a1))
    tint = pd.DataFrame(rows); tint.to_parquet(f"{out}/tiles_int.parquet"); del F, Jm, dist
    # 5. R06 inputs for P-V1
    R6.FIELDS = f"{root}/fields"; ichk = BI.process_pair(m, pid, tiles, blocks, workers, f"{root}/inputs")
    # 6. amplitude statistic per view (B4), before any prediction
    n_i, b_i = block_stats(tint, tint.interior_D1.values.astype(bool)); fb = tint.A_fine_int_D1.notna().values; n_f, b_f = block_stats(tint, fb)
    views = dict(interior=dict(n_tiles=n_i, n_blocks=b_i, amplitude_m=float(tint[tint.interior_D1].A_fine.median()) if n_i else np.nan),
                 fallback=dict(n_tiles=n_f, n_blocks=b_f, amplitude_m=float(tint.loc[fb, "A_fine_int_D1"].median()) if n_f else np.nan), all=dict(n_tiles=int(len(tint)), amplitude_m=float(tint.A_fine.median())))
    for bmin in (P4.B_MIN_DEFAULT, 12, 16):
        v = "interior" if (n_i >= P4.MIN_TILES and b_i >= bmin) else ("fallback" if (n_f >= P4.MIN_TILES and b_f >= bmin) else "not_scorable"); views[f"view_B{bmin}"] = v; views[f"amplitude_B{bmin}_m"] = views[v]["amplitude_m"] if v != "not_scorable" else np.nan
    cells = 256 * hr_res / float(row["lr_native_res_m"]); dev_min = P4.dev_min_cells()
    contract_nan = float(np.isnan(feats[B.CONTRACT].values.astype(float)).mean()) if len(feats) else np.nan
    stats = dict(pair=pid, unit=unit, status="built", hr_res=hr_res, lr_native=float(row["lr_native_res_m"]), k=int(row["max_recoverable_k"]), D1_px=D1, cells_across=cells, dev_min_cells=dev_min, thinner_than_development=bool(cells < dev_min), n_tiles_v1=int(len(tiles1)), n_tiles_v1_1=int(len(tiles)), n_blocks_kept=int(blocks.kept.sum()), n_feature_rows=int(len(feats)),
                 ship_products_usable=meta["ship_usable"], product_features=("used" if meta["ship_usable"] else "masked"), contract_columns_frac_nan=contract_nan, k_provenance=kp,
                 r04_summary={k: v for k, v in summ.items() if k != "rejected"} | dict(rejected=summ["rejected"]), r04_grid_check=chk, r06_inputs=ichk, views=views, seconds=time.time() - t0)
    json.dump(stats, open(f"{out}/pair_stats.json", "w"), indent=1, default=float)
    log(f"  built {pid} ({unit}): tiles {len(tiles1)} -> {len(tiles)}; D1 {D1} px; interior {n_i} tiles / {b_i} blocks; fallback {n_f} / {b_f}; view(B8) {views['view_B8']}; amplitude {views['amplitude_B8_m']:.4f} m ({time.time()-t0:.0f} s)", flush=True)
    return stats

if __name__ == "__main__":
    ap = P4.argparser("Phase 4 build"); ap.add_argument("--rows", default=None, help="rehearsal only (R11 §A2.3): parquet whose rows replace the listed pairs' manifest rows (harness supplement)"); ap.add_argument("--out_root", default=None, help="rehearsal only: output root instead of ROOT[rehearsal]"); a = ap.parse_args(); pairs = P4.split_pairs(a.pairs)
    if a.mode == "phase4" and (a.rows or a.out_root): raise P4.Refusal("--rows and --out_root are rehearsal-only (phase4 mode takes the rows from the supplement and writes to ROOT[phase4])")
    m, root, rec = P4.start("phase4_build.py", a.mode, pairs); hrec = None
    if a.rows: m, hrec = P4.harness_rows(m, a.rows, pairs)
    if a.out_root: root = a.out_root; os.makedirs(f"{root}/provenance", exist_ok=True); json.dump(rec | dict(out_root=root, harness_rows=hrec), open(f"{root}/provenance/phase4_build.py_{rec['time'].replace(':', '')}_{os.getpid()}.json", "w"), indent=1, default=str)
    allst = [build_pair(m, pid, root, a.workers, acq_rec=rec.get("acq_run_record")) for pid in pairs]
    json.dump(dict(mode=a.mode, pairs=allst, harness_rows=hrec), open(f"{root}/build/build_summary_{int(time.time())}.json", "w"), indent=1, default=float); print("done", flush=True)
