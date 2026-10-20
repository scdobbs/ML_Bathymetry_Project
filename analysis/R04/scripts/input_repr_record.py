"""R04 §7 — input-representation record (report only, no decisions): what the R02 clean pipeline (extract_clean_patches.py
+ het_clean.py) feeds the network, quoted from the code, plus per development unit the LR cell size, LR cells per 256-px
window and the range of the ratio HR px / LR cell (from the manifest and the harmonized raster headers)."""
import json, os, sys, re
import numpy as np, pandas as pd, rasterio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import r04_common as C
m = C.start("input_repr_record"); rows = []
for u, ps in list(C.UNITS.items()) + list(C.EW0207.items()):
    for pid in ps:
        r = C.pair_row(m, pid)
        with rasterio.open(os.path.join(C.DATA, r.harmonized_path_hr)) as h, rasterio.open(os.path.join(C.DATA, r.harmonized_path_lr)) as l:
            hr_res = abs(h.transform.a); cell = abs(l.transform.a)
        lr_native = float(r.lr_native_res_m); lr_px = max(2, int(round(256 * hr_res / lr_native)))
        rows.append(dict(unit=u, pair=pid, hr_res_m=hr_res, lr_cell_m=cell, lr_native_m=lr_native, cells_per_256px_window=256 * hr_res / lr_native, lr_px_cells_R02=lr_px, hr_px_per_lr_cell=lr_native / hr_res, k=int(r.max_recoverable_k)))
d = pd.DataFrame(rows); d.to_csv(f"{C.ROOT}/input_repr_per_pair.csv", index=False)
g = d.groupby("unit").agg(pairs=("pair", "size"), lr_cell_m=("lr_cell_m", lambda s: "/".join(f"{x:.2f}" for x in s)), lr_native_m=("lr_native_m", lambda s: "/".join(f"{x:.2f}" for x in s)),
                          cells_per_256=("cells_per_256px_window", lambda s: "/".join(f"{x:.1f}" for x in s)), ratio_min=("hr_px_per_lr_cell", "min"), ratio_max=("hr_px_per_lr_cell", "max"), k=("k", lambda s: "/".join(str(x) for x in s)))
src = open(f"{C.REPO}/preprocessing/extract_clean_patches.py").read(); het = open(f"{C.R02}/het_clean.py").read()
quotes = dict(lr_representation="tile_ship_side: LR area-averaged (Resampling.average) onto lr_px x lr_px cells over the exact 256-px HR window footprint, lr_px = max(2, round(256 * hr_res / lr_native)); validity nearest; invalid cells <- mu_lr; bicubic (scipy zoom order 3) to 256 x 256 HR px (build_inputs)",
              normalization="per window, ship-side statistics only: mu_lr = mean(LR | valid), s_lr = max(std(LR | valid), 0.5 m); x = (bicubic(filled LR) - mu_lr) / s_lr; target y = (HR - delta - mu_lr) / s_lr with delta = median(HR - bicubic) over joint px (label only)",
              channels="in_ch = 2: channel 1 = x (normalized bicubic ship surface), channel 2 = lr_valid_hr (LR validity nearest-upsampled to 256 px, floor(i*L/H)); the model output is (mean, log-variance) per pixel (HetUNetClean, residual mean)",
              augmentation="none (het_clean.py docstring: 'no augmentation'; recipe = het_louo.py)",
              scale_conditioning="scale = [log2(lr_native), log2(hr_res)] from the manifest, fed through FiLM (2-vector) in every block; lr_native and hr_res enter only this way; k does not enter the model (it sets the band edges of the evaluation only); depth enters only through mu_lr normalization (removed from x) - not as an input",
              window="256 x 256 HR px window; training stride 64 px (train pairs) / 128 px (val-test pairs) in R02; R04 tiles_v1 uses stride 128 everywhere")
ev = {k: bool(re.search(p, src + het)) for k, p in dict(area_average=r"Resampling\.average", bicubic_zoom=r"bicubic_upsample|zoom\(", s_floor=r"s_floor", in_ch2=r"in_ch=2|in_ch = 2", no_aug=r"no augmentation", film=r"_FiLM|FiLM", scale_log2=r"log2\(lr_native\)").items()}
md = ["# §7 Input-representation record (R02 clean pipeline; factual, no proposal)\n", "| item | as implemented |", "|---|---|"] + [f"| {k} | {v} |" for k, v in quotes.items()]
md += ["", f"Code evidence (regex hits in extract_clean_patches.py + het_clean.py): {ev}", "", "| unit | pairs | LR cell (m) | lr_native (m) | LR cells per 256-px window | HR px per LR cell (min-max) | k |", "|---|---|---|---|---|---|---|"]
for u, r in g.iterrows(): md.append(f"| {u} | {r.pairs} | {r.lr_cell_m} | {r.lr_native_m} | {r.cells_per_256} | {r.ratio_min:.1f}-{r.ratio_max:.1f} | {r.k} |")
open(f"{C.ROOT}/input_repr_record.md", "w").write("\n".join(md) + "\n"); json.dump(dict(quotes=quotes, evidence=ev, per_pair=rows), open(f"{C.ROOT}/input_repr_record.json", "w"), indent=1)
print("\n".join(md))
