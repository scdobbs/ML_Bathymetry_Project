"""R03 §6.1 — raw-swath availability audit from the acquisition repo's persisted records + manifest. No downloads.
Lockbox pairs excluded (guard installed by r03_common)."""
import sys, os, json, re, glob, collections, numpy as np, pandas as pd
sys.path.insert(0, "/scratch/users/scdobbs/grl_review/R03"); import r03_common as C
sys.path.insert(0, "/home/users/scdobbs/GAN_Bathymetry_Project"); from preprocessing.colocated_utils import load_manifest
ACQ = "/home/users/scdobbs/co-located-auv-ship-based-multibeam-acquisition/reports"; OUT = f"{C.ROOT}/s6"; os.makedirs(OUT, exist_ok=True)
m = load_manifest(C.DATA); fl = pd.read_csv(f"{ACQ}/stage_b_fetch_list_2026-06-22.csv").set_index("cruise")
plan = pd.read_csv(f"{ACQ}/discovery/stage_b_file_plan_2026-06-22.csv"); log = pd.read_csv(f"{ACQ}/discovery/stage_b_fetch_log.csv")
sizes = pd.read_csv(f"{ACQ}/discovery/stage_a_raw_lr_sizes_2026-06-22.csv").set_index("cruise")
plan["fmt"] = plan.filename.str.extract(r"\.mb(\d+)(?:\.gz)?$")[0]; plan["gb"] = pd.to_numeric(plan.advertised, errors="coerce") / 1e9
RAW_LR = "/scratch/groups/hilley/auv_ship_colocated_bathy/raw_lr"; GRIDDED = "/scratch/groups/hilley/auv_ship_colocated_bathy/raw_lr_gridded"
rows = []
for _, r in m.iterrows():
    pid = r.pair_id
    if pid in C.LOCKBOX: continue
    unit = C.UNIT_OF.get(pid, "(stopped/no k)" if pid in C.STOPPED_NO_K else "?"); cruise = r.lr_cruise
    if str(r.raw_path_lr).startswith("raw_lr/"):          # NCEI swath, gridded by our pipeline (stage_c_full: mbgrid -A2 -G3 -F1 -C0 -M)
        p = plan[plan.cruise == cruise]; lg = log[log.cruise == cruise]
        sc = glob.glob(f"{ACQ}/discovery/stage_c_full/{cruise}.json"); scj = json.load(open(sc[0])) if sc else []
        hit = [e for e in scj if e.get("hr_id", "").replace(":", "_") in pid] or scj[:1]
        n_disk = len([f for f in glob.glob(f"{RAW_LR}/{cruise}/*") if os.path.isfile(f)]); n_grd = len(glob.glob(f"{GRIDDED}/{cruise}__*.grd"))
        rows.append(dict(pair_id=pid, unit=unit, cruise=cruise, source="NCEI swath (data.ngdc.noaa.gov), gridded by our pipeline", mb_format=",".join(sorted(p.fmt.dropna().unique())) or "?",
                         n_files_planned=int(len(p)), gb_planned=round(float(p.gb.sum()), 2), fetch_mode=fl.loc[cruise, "fetch_mode"] if cruise in fl.index else "?",
                         n_files_fetched_ok=int((lg.integrity.astype(str).str.contains("verified|ok", case=False)).sum()) if len(lg) else 0,
                         on_disk_now=f"{n_disk} swath files in raw_lr/{cruise}; {n_grd} .grd in raw_lr_gridded" + (" (PURGED)" if n_disk == 0 else ""),
                         retrievable="yes: URLs persisted in stage_b_file_plan_2026-06-22.csv", grid_provenance=(f"mbgrid union grid, cell {hit[0].get('achieved_cell_m')} m, fill {hit[0].get('fill_fraction')}, mode {hit[0].get('process_mode')}, qc {hit[0].get('qc')}" if hit else "stage_c_full record missing"),
                         sd_count_grids="produced by mbgrid -M at gridding time (stage_c_full.py:189-190) but not persisted; purged with raw_lr_gridded",
                         lr_native_m=r.lr_native_res_m, hr_id=pid.split("__")[-1]))
    else:                                                  # provider grid (anchor pairs)
        rows.append(dict(pair_id=pid, unit=unit, cruise=cruise, source=f"provider grid: {r.lr_doi}", mb_format="n/a (gridded product)", n_files_planned=1, gb_planned=round(os.path.getsize(os.path.join(C.DATA, r.raw_path_lr)) / 1e9, 3) if os.path.exists(os.path.join(C.DATA, r.raw_path_lr)) else np.nan,
                         fetch_mode="provider", n_files_fetched_ok=1, on_disk_now=f"OAK {r.raw_path_lr}", retrievable="raw swath not acquired; would need the provider's swath release (PANGAEA/USGS) — not assessed",
                         grid_provenance="provider-gridded (not produced from swath by us)", sd_count_grids="none", lr_native_m=r.lr_native_res_m, hr_id=pid))
df = pd.DataFrame(rows); df.to_csv(f"{OUT}/s6_audit.csv", index=False)
cols = ["pair_id", "unit", "cruise", "source", "mb_format", "n_files_planned", "gb_planned", "fetch_mode", "n_files_fetched_ok", "on_disk_now", "grid_provenance", "sd_count_grids"]
md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(str(x) for x in r) + " |" for r in df[cols].values]
open(f"{OUT}/s6_audit.md", "w").write("\n".join(md)); print(df[["pair_id", "unit", "cruise", "mb_format", "n_files_planned", "gb_planned", "fetch_mode", "on_disk_now"]].to_string()); print("total planned GB (NCEI pairs):", round(df[df.fetch_mode != "provider"].drop_duplicates("cruise").gb_planned.sum(), 1))
