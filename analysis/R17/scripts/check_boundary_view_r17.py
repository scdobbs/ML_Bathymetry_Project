"""R17 §2.2 follow-up, COUNTS ONLY: for the drift sites (Escanaba, Chile margin), the kept tiles whose target-joint pixels fall inside a lattice tile
that ACQ-R13 zeroed (detector of check_build_r17.boundary), split by view membership (interior at D1 = primary view; A_fine_int defined = fallback
view), and the share of such tiles' target-joint pixels that lie in zeroed tiles, binned. No target, amplitude or raster value is printed.
Writes R17/tables/boundary_view_check.json."""
import json, os, sys
import numpy as np, pandas as pd, rasterio
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import check_build_r17 as CB
P5 = CB.P5; C4 = CB.C4; ROOT = CB.ROOT
SITES = ("Escanaba__MGDS_30368", "MV1003__MGDS_31860")
if __name__ == "__main__":
    m, root, rec = P5.start("check_boundary_view_r17.py", "phase5", list(P5.SEALED8)); out = {}
    for pid in SITES:
        row = C4.pair_row(m, pid); paths = CB.resolve_pair_paths(C4.DATA, row)
        jf = CB.compute_joint_valid(paths)[0].astype(bool); jd = CB.compute_joint_valid(dict(paths, joint_valid=None))[0].astype(bool); H, W = jf.shape
        Z = np.zeros((H, W), bool)
        for r0 in range(0, H - CB.WIN + 1, CB.WIN):
            for c0 in range(0, W - CB.WIN + 1, CB.WIN):
                s = (slice(r0, r0 + CB.WIN), slice(c0, c0 + CB.WIN))
                if not jf[s].any() and jd[s].any(): Z[s] = True
        del jd
        with rasterio.open(f"{ROOT}/fields/{pid}/joint_v1.tif") as s: J = s.read(1) == 1
        ti = pd.read_parquet(f"{ROOT}/build/{pid}/tiles_int.parquet", columns=["r0", "c0", "interior_D1", "A_fine_int_D1"])
        hit = []
        for r in ti.itertuples():
            s = (slice(int(r.r0) + CB.MARGIN, int(r.r0) + CB.MARGIN + CB.TILE), slice(int(r.c0) + CB.MARGIN, int(r.c0) + CB.MARGIN + CB.TILE)); j = J[s]; n = int((j & Z[s]).sum())
            if n: hit.append(dict(interior=bool(r.interior_D1), fallback=bool(pd.notna(r.A_fine_int_D1)), share=n / max(int(j.sum()), 1)))
        h = pd.DataFrame(hit) if hit else pd.DataFrame(columns=["interior", "fallback", "share"])
        bins = [0, 0.1, 0.25, 0.5, 0.75, 1.0001]
        out[pid] = dict(kept_tiles=int(len(ti)), tiles_with_target_px_in_zeroed=int(len(h)), of_which_interior_D1=int(h.interior.sum()) if len(h) else 0,
                        of_which_fallback_defined=int(h.fallback.sum()) if len(h) else 0, interior_view_tiles_total=int(ti.interior_D1.sum()),
                        share_of_target_px_in_zeroed_hist={f"[{a},{b if b <= 1 else 1})": int(((h.share >= a) & (h.share < b)).sum()) for a, b in zip(bins[:-1], bins[1:])} if len(h) else {})
        print(pid, json.dumps(out[pid]), flush=True)
    json.dump(out, open(f"{CB.DR}/R17/tables/boundary_view_check.json", "w"), indent=1)
