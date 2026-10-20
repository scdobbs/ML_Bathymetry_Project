"""R17 H1 ruling (Steve, 2026-10-14; option (b) = the ACQ-R13 rule that tiles crossing a quadrant boundary are dropped): the FIXED exclusion list.
A kept tile (tiles_v1_1) is excluded iff its target-joint pixels (joint_v1 over the tile's central 128 px) fall inside a lattice tile that ACQ-R13
zeroed (detector of check_build_r17.boundary). Applied to every eligible site (only Escanaba is expected to have any). Writes tile ids and their
tile-grid coordinates only (no value) to R17/exclusions_phase5_r17.json; also counts, for information, accepted-but-not-kept tiles (tiles_v1 only)
whose field pixels fall inside zeroed tiles (they are not on the list; the figure masks the listed tiles only)."""
import json, os, sys
import numpy as np, pandas as pd, rasterio
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import check_build_r17 as CB
P5 = CB.P5; C4 = CB.C4; ROOT = CB.ROOT
if __name__ == "__main__":
    m, root, rec = P5.start("make_exclusions_r17.py", "phase5", list(P5.SEALED8)); excl = []; info = {}
    for pid in rec["admitted"]:
        row = C4.pair_row(m, pid); paths = CB.resolve_pair_paths(C4.DATA, row)
        jf = CB.compute_joint_valid(paths)[0].astype(bool); jd = CB.compute_joint_valid(dict(paths, joint_valid=None))[0].astype(bool); H, W = jf.shape
        Z = np.zeros((H, W), bool)
        for r0 in range(0, H - CB.WIN + 1, CB.WIN):
            for c0 in range(0, W - CB.WIN + 1, CB.WIN):
                s = (slice(r0, r0 + CB.WIN), slice(c0, c0 + CB.WIN))
                if not jf[s].any() and jd[s].any(): Z[s] = True
        del jd, jf
        with rasterio.open(f"{ROOT}/fields/{pid}/joint_v1.tif") as s: J = s.read(1) == 1
        def hits(t):
            out = []
            for r in t.itertuples():
                s = (slice(int(r.tr0), int(r.tr0) + CB.TILE), slice(int(r.tc0), int(r.tc0) + CB.TILE))
                if (J[s] & Z[s]).any(): out.append(dict(tile_id=r.tile_id, pair=pid, r0=int(r.r0), c0=int(r.c0), tr0=int(r.tr0), tc0=int(r.tc0)))
            return out
        kept = pd.read_parquet(f"{ROOT}/build/{pid}/tiles_v1_1.parquet", columns=["tile_id", "r0", "c0", "tr0", "tc0"])
        allv1 = pd.read_parquet(f"{ROOT}/build/{pid}/tiles_v1.parquet", columns=["tile_id", "r0", "c0", "tr0", "tc0"])
        h = hits(kept); excl += h; dropped_only = [x for x in hits(allv1) if x["tile_id"] not in set(kept.tile_id)]
        info[pid] = dict(zeroed_lattice_tiles=int(Z[::CB.WIN, ::CB.WIN].sum()), excluded_kept_tiles=len(h), not_kept_v1_tiles_with_px_in_zeroed=len(dropped_only))
        print(pid, json.dumps(info[pid]), flush=True)
    out = dict(rule="R17 H1 ruling (Steve, 2026-10-14): a kept tile (tiles_v1_1) whose target-joint px fall inside an ACQ-R13 boundary-zeroed stride-256 lattice tile is excluded from the figures and from every R18 row",
               source=dict(root=ROOT, acq_run_record=rec["acq_run_record"]["sha256"], supplement=rec["supplement"]["sha256"]), per_site=info, n_tiles=len(excl), tiles=sorted(excl, key=lambda d: d["tile_id"]))
    json.dump(out, open(f"{CB.DR}/R17/exclusions_phase5_r17.json", "w"), indent=1); print("n_tiles", len(excl), flush=True)
