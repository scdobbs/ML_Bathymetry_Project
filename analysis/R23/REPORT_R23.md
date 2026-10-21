# REPORT R23 — Figures 1 to 3 for the GRL revision (2026-10-21)

**Directive:** `reports_post_grl_review/R23_figures.md` (acquisition repo). **Where things went:** the directive names `grl_review/`; this repository's review folder is `analysis/` (renamed from `grl_reviews-and-prompt-reports/directives_and_reports/` on 2026-10-20), so the outputs are under `analysis/R23/figures/` and this report is `analysis/R23/REPORT_R23.md`, following the per-directive layout of R01…R22. Read only on every frozen output (OAK archive `grl_review_archive/`, written `A/` below; R21/R22 tables; manifests); writes only under `analysis/R23/`. No score was recomputed: every plotted number is read from a frozen CSV, JSON or report table, the rasters of Fig. 1b are the frozen harmonized grids and the frozen `F_hr` mosaic, and the spectra of Fig. 1c call the frozen `stage_c5_spectral.radial_psd` routine unchanged. Only the Spearman checks of Fig. 3 are recomputed from the plotted points, as the directive asks. Committed by explicit path, not pushed; Zenodo untouched; manuscript, captions and Claude Docs untouched.

Jobs (dev partition, 4 CPUs, 16 GB): 48817527 (failed on my own pdffonts parser, see §6), 48817808, 48818434, 48818683, 48818924. Runner: `analysis/R23/figures/run_r23.sh` (`srun -p dev -c 4 --mem=16G -t 0:40:00 bash run_r23.sh`).

## 1. Files written

| file | MB | sha256 |
|---|---|---|
| `analysis/R23/figures/Dobbs_Figure1.eps` | 42.13 | `12e2c9d2995378819c242e220a5f2b9c3d4e4e00767f8f922203f660ab020834` |
| `analysis/R23/figures/Dobbs_Figure1.pdf` | 2.49 | `b57413724c0054403a9e7019fa4a8d5990a6dd81fd43d82e58d1bedd5ff1e683` |
| `analysis/R23/figures/Dobbs_Figure1.svg` | 3.97 | `6132ddd9e45f522e2a0ad3035fa1da9234b8f3a12e7f92c2c1c36961e29a9304` |
| `analysis/R23/figures/Dobbs_Figure1_data.csv` | 0.15 | `8b84c2dc12512201c20750c8c9f67f24a74dc3d709057bb1d7f72b96e603a916` |
| `analysis/R23/figures/Dobbs_Figure1_preview.png` | 1.12 | `4c917d345ed261afbd0b6c4461caaed1a6fc21a9900d320a00e63f71d7ef4caa` |
| `analysis/R23/figures/Dobbs_Figure2.eps` | 0.07 | `bc6f0cb316e36a1572587556207c644733ed6d71bed0f952d03886fecc1761a2` |
| `analysis/R23/figures/Dobbs_Figure2.pdf` | 0.02 | `8f3c6d0dd0e8fffb103f63262c4500c88ac0984afc40b6dd137c779cac442a5f` |
| `analysis/R23/figures/Dobbs_Figure2.svg` | 0.03 | `f3e8964ac779fe7aa9af080d678ca970671f68937721c1a5fb664f13305f1767` |
| `analysis/R23/figures/Dobbs_Figure2_data.csv` | 0.01 | `3a3318e3ce113661bab7f016f7dfd3feeb35e5f967b3eecf93cf0147b7e854f2` |
| `analysis/R23/figures/Dobbs_Figure2_preview.png` | 0.09 | `227be6933e89025de9c4c0e357404a17632003c37cb5aaba7661a0f40c956850` |
| `analysis/R23/figures/Dobbs_Figure3.eps` | 0.08 | `25c0fef423676b2a849e868143e4dfc76cae7912716b04f3cadc57b667368e21` |
| `analysis/R23/figures/Dobbs_Figure3.pdf` | 0.02 | `ed2d60b426cc1d4f3820f687f12b5ae8901c8e4eed9c5d34b955dda50a7a80ef` |
| `analysis/R23/figures/Dobbs_Figure3.svg` | 0.05 | `3eb899364dde004369e39c7e291e1a3b6266873e7f5db98c8dfb392a03ca6d68` |
| `analysis/R23/figures/Dobbs_Figure3_data.csv` | 0.01 | `b503853f9443d78376802341ea7b34bce3e502d954f5e11b839434141678a480` |
| `analysis/R23/figures/Dobbs_Figure3_preview.png` | 0.11 | `028d4d22d2fd60dab8643f0767d490e8edac50593107a618039c7f7ccfefaf32` |
| `analysis/R23/figures/fig1.py` | 0.02 | `0d360d2f05d93d0ee0f4dc22e9ae070fd79131d6ca2196700869efabf64a0e7c` |
| `analysis/R23/figures/fig1_meta.json` | 0.01 | `29e7f00ca016e5e3ec71ce65edf20bd4bd125a4c24b5de82edaffe394dae3e51` |
| `analysis/R23/figures/fig2.py` | 0.01 | `7f04f9a78708e1fa4a084f5345c4fd773b5b965bedd2c1d4fd97edc7d1490300` |
| `analysis/R23/figures/fig2_meta.json` | 0.00 | `fc55d105b0a93ef8b145e70d0f242d91ca5bda33fd7124973a5ab560b8d9880f` |
| `analysis/R23/figures/fig3.py` | 0.01 | `0a23ff8fde8672f67e94ee6256a5ab5d7ade567ff4f0758abdcb56d25e0709c8` |
| `analysis/R23/figures/fig3_meta.json` | 0.00 | `c4f2774e4bd81f1e73a55cd8e69fb07a43d48f4a0ff8dba6f0adf1b584cf138e` |
| `analysis/R23/figures/fig_style.py` | 0.01 | `ffb7f05c2e5e0b03e9906abb318554df54ae799aa9317e70bbe6c95600eae245` |
| `analysis/R23/figures/run_r23.sh` | 0.00 | `c6ef2ae26c30148d2eb37777fc1721c33f741d902bd3f184e0821d7eb61661c2` |

`Dobbs_Figure1.eps` is 42 MB (four 600-dpi raster panels plus the basemap, stored uncompressed as EPS requires). It is on disk but **not committed** (the repository keeps tracked files under 10 MB); say the word and I add it, or drop the EPS from the deliverable since the PDF is the primary. Everything else in the table is committed.

## 2. Font

`fc-list` on Sherlock finds only DejaVu (9 entries); **Arial is not available**, so **Liberation Sans** (Arial-metric) is used, as the directive provides. It was not installed either: Liberation Fonts 2.1.5 (SIL OFL 1.1) were fetched from the upstream release (`https://github.com/liberationfonts/liberation-fonts/files/7261482/liberation-fonts-ttf-2.1.5.tar.gz`, sha256 `7191c669bf38899f73a2094ed00f7b800553364f90e2637010a69c0e268f25d0`) and the four LiberationSans ttf files plus the licence were placed in the user font directory `~/.local/share/fonts/` (user-level only, `fc-cache` on that directory). `fig_style.py` registers these files with matplotlib explicitly and also maps the mathtext font set (rm/it/bf/sf/cal/tt) to Liberation Sans with `mathtext.fallback = None`, so no other font can enter the PDFs. Every PDF is checked in code with `pdffonts` (module `system poppler/0.47.0`): all fonts embedded (Type 42 subsets, reported as "CID TrueType") and named LiberationSans; page width 170 mm ± 0.5 mm via `pdfinfo`; and the SVG is parsed to assert no text below 6 pt.

### pdffonts, `Dobbs_Figure1.pdf` (170.000 × 131.000 mm; SVG smallest text 6.0 pt over 118 text elements, families ['Liberation Sans'])
```
name                                 type              encoding         emb sub uni object ID
------------------------------------ ----------------- ---------------- --- --- --- ---------
FRZNSF+LiberationSans-Bold           CID TrueType      Identity-H       yes yes yes     30  0
FRZNSF+LiberationSans                CID TrueType      Identity-H       yes yes yes     37  0
```
### pdffonts, `Dobbs_Figure2.pdf` (170.000 × 75.000 mm; SVG smallest text 6.0 pt over 58 text elements, families ['Liberation Sans'])
```
name                                 type              encoding         emb sub uni object ID
------------------------------------ ----------------- ---------------- --- --- --- ---------
FRZNSF+LiberationSans-Bold           CID TrueType      Identity-H       yes yes yes     18  0
FRZNSF+LiberationSans                CID TrueType      Identity-H       yes yes yes     25  0
```
### pdffonts, `Dobbs_Figure3.pdf` (170.000 × 61.000 mm; SVG smallest text 6.0 pt over 114 text elements, families ['Liberation Sans'])
```
name                                 type              encoding         emb sub uni object ID
------------------------------------ ----------------- ---------------- --- --- --- ---------
FRZNSF+LiberationSans-Bold           CID TrueType      Identity-H       yes yes yes     22  0
FRZNSF+LiberationSans                CID TrueType      Identity-H       yes yes yes     29  0
```

## 3. Assertions

### Figure 1 (all pass; the figure would have stopped otherwise)
| assertion | result | got | want |
|---|---|---|---|
| 16 development units | PASS | `16` | `16` |
| 10 real-relief (filled) | PASS | `10` | `10` |
| 6 artifact-affected (open) | PASS | `6` | `6` |
| 2 blind-test-1 triangles | PASS | `2` | `2` |
| 5 blind-test-2 squares | PASS | `5` | `5` |
| 14 scorable real-relief development pairs | PASS | `14` | `14` |
| chosen pair rho equals frozen (per_pair_D1 rho_s_lr) | PASS | `0.3902422411318533` | `0.3902422411318533` |
| chosen pair k equals frozen (manifest max_recoverable_k) | PASS | `16` | `manifest` |
| chosen pair lr_native equals frozen (per_pair_D1 lr_native) | PASS | `65.99` | `65.99` |
| chosen pair hr_res equals frozen (per_pair_D1 hr_res) | PASS | `1.00027363181973` | `1.00027363181973` |
| interior tiles of the chosen pair equal frozen n_interior | PASS | `523` | `523` |
| band edge f_lr (cycles/m) = (hr_res/lr_native)/hr_res | PASS | `0.015153811183512654` | `0.015153811183512654` |
| band edge f_hi (cycles/m) = min(k f_lr, 0.45)/hr_res | PASS | `0.24246097893620247` | `0.24246097893620247` |

### Figure 2
| assertion | result | got | want |
|---|---|---|---|
| 9 units in (a) | PASS | `9` | `9` |
| median s_lr rho = 0.390 | PASS | `0.39` | `0.39` |
| frozen CI of the median [0.165, 0.588] (R09 scores_ten, interior D1) | PASS | `[0.165, 0.588]` | `[0.165, 0.588]` |
| min s_lr rho = -0.09 at TAG (lu09) | PASS | `[-0.09, 'lu09']` | `[-0.09, 'lu09']` |
| max s_lr rho = 0.63 | PASS | `[0.63, 'lu08']` | `0.63` |
| pooled row 0.392 [0.317, 0.430], 14 pairs, seed 20261013 | PASS | `[0.392, 0.317, 0.43, 14, 20261013]` | `[0.392, 0.317, 0.43, 14, 20261013]` |
| median GBT_ship rho = 0.370 | PASS | `0.37` | `0.37` |
| median V2 - s_lr = +0.028 | PASS | `0.028` | `0.028` |
| P-V1 above s_lr at exactly 5 of 9 (full precision) | **FAIL** | `6` | `5` |
| P-V1 above s_lr at exactly 5 of 9 (two-decimal values, as tabulated) | PASS | `5` | `5` |

**The one failure.** "P-V1 above s_lr at exactly 5 of 9 units" holds for the two-decimal values of Table S2 (5 of 9) but not at the full precision of the frozen per-unit values (6 of 9): at lu13 (Davidson Seamount) P-V1 is +0.5929 against s_lr +0.5877, a difference of +0.0053 that rounds to 0.59 for both. The figure plots the full-precision differences, so the lu13 P-V1 marker sits on the zero line (visually indistinguishable from zero). Per-unit P-V1 − s_lr: lu08 -0.054, lu13 +0.005, lu00 +0.079, lu07 +0.069, nu00 -0.060, lu04 -0.029, lu05 +0.064, lu15 +0.086, lu09 +0.275. Nothing was changed; Steve may want the caption or text to say "5 of 9 at the tabulated precision" or "6 of 9".

### Figure 3 (all pass)
| assertion | result | got | want |
|---|---|---|---|
| 16 units in (a) | PASS | `[16, 16]` | `16` |
| median s_lr rho (64-cell window, all tiles) = 0.350 | PASS | `0.35` | `0.35` |
| median spectral rho = 0.09 | PASS | `0.09` | `0.09` |
| 10 units with Goff defined, 4 computed | PASS | `[10, 4]` | `[10, 4]` |
| Spearman rho(Goff, A_fine) = -0.31 on the 10 plotted points | PASS | `-0.309` | `-0.31` |
| Spearman rho(s_lr, A_fine) = +0.64 on the 10 black points | PASS | `0.636` | `0.64` |
| Spearman rho(s_lr, A_fine) = 0.70 on all 16 | PASS | `0.697` | `0.7` |

Spearman values recomputed from the plotted points: ρ(Goff, A_fine) = -0.3091 (10 units), ρ(s_lr, A_fine) = +0.6364 (the 10 black points) and +0.6971 (all 16), matching R15 `r_correlations.md` (−0.309, +0.636) and the R15 must-hold value (0.70).

## 4. Figure 1b: the example pair

Rule applied (§1 of the directive): of the 14 scorable real-relief development pairs (`A/R10b/rehearsal/scores/per_pair_D1.csv`, role development, ten real-relief units, view ≠ not_scorable), keep those scored on the interior view with ≥ 50 interior tiles and rigid co-registration (Table S1 `coreg case`; every development pair is rigid), and take the pair whose s_lr ρ is closest to the pooled +0.392. Eleven pairs survive the filters (dropped: TN399__MGDS_33068 and AR26, fallback view; KM0923, 24 interior tiles). The five closest:

| rank | pair | unit | ρ(s_lr), interior | interior tiles | k | lr_native (m) | hr_res (m) | distance to 0.392 |
|---|---|---|---|---|---|---|---|---|
| 1 | FK151121__MGDS_22383 | nu00 | +0.3902 | 523 | 16 | 65.99 | 1.00027 | 0.0018 |
| 2 | AT37-13__MGDS_31199 | lu04 | +0.3841 | 86 | 8 | 23.2 | 1.00184 | 0.0079 |
| 3 | RR1506__MGDS_29779 | lu00 | +0.5065 | 2157 | 8 | 20.3 | 1.00014 | 0.1145 |
| 4 | EW9801__MGDS_31425 | lu07 | +0.5458 | 55 | 2 | 50.4 | 2.06656 | 0.1538 |
| 5 | NA080__MGDS_31290 | lu13 | +0.5863 | 2430 | 16 | 57.4 | 1.00000 | 0.1943 |

**Chosen: `FK151121__MGDS_22383`** (unit nu00, Izu-Bonin-Mariana arc, R/V Falkor FK151121 ship multibeam grid at 65.99 m against the AUV Sentry 1 m grid, MGDS:22383; k = 16; interior view; Table S1 status "in C1"). For the caption: footprint centre **18.1973°N, 144.7117°E** (144°42.7′E, 18°11.8′N), water depth about 3,740 m; harmonized grid 4,459 × 5,719 px of 1.00027 m (UTM 55N), 523 interior tiles. Override by setting `pid` in `fig1.py` if the second candidate (Costa Rica Rift, ρ +0.384, 86 tiles) is preferred for its continental-margin setting.

Panels: (1) ship grid, the harmonized `lr.tif` drawn at its native 65.9 m cells (nearest-neighbour, no smoothing), (2) AUV grid `hr.tif`, both as depth = −elevation with the shared `batlow` limits 3600.0,–4020.0 m (1st–99th percentiles of the ship grid, rounded to 10 m); (3) the frozen band-passed fine field `A/R04/fields/FK151121__MGDS_22383/F_hr.tif` (targets_v1 §1.1, same grid as `hr.tif`, asserted) on `vik` with symmetric limits ±7.5 m (99th percentile of |F_hr| over the interior tiles); (4) per-tile s_lr (`A/R08b/scores/master_tile.parquet`, column `ref_s_lr`, `interior_D1 == True`, 523 tiles of 128 px placed at r0 + 64, c0 + 64 from the tile id) on `batlow`, limits 1.0,–40.0 m. One 1 km scale bar on panel 1, no tick labels, north up. The blocky edge of panel 2 is the harmonized AUV footprint (joint-valid mask), not a rendering artefact.

Fig. 1c: the frozen S3.2 routine (`src/discovery/stage_c5_spectral.py`: planar detrend, Hann window, 2-D FFT, 48 log-spaced radial bins, 1024 × 1024 cap) applied exactly as the k sweep did: `load_to_common` of `lr.tif` and `hr.tif` onto the AUV-native grid (1.00027 m, bounds 144.6906, 18.1714, 144.7330, 18.2234), `_cap_pair` to 1024 × 1024, then `radial_psd` for each grid. The ship curve is therefore the bilinearly upsampled ship grid, which is why it falls steeply above f_lr. The routine's mean |F|² per bin is multiplied by one constant (dx² / Σw², Welch normalisation, 6.799e-06) so the y axis reads as a power spectral density in m⁴ (m² per (cycle m⁻¹)²); the raw routine values are in the data csv (`psd_raw`). Band: f_lr = 1/65.99 = 0.01515 cycles m⁻¹ to f_hi = 16 f_lr = 0.24246 cycles m⁻¹ (k·f_lr = 0.2425 cycles per HR pixel < 0.45), light gray, unlabeled.

## 5. GMRT

**Reached.** GMRT GridServer (`https://www.gmrt.org/services/GridServer?north=80&south=-80&west=-180&east=180&mresolution=25000&layer=topo&format=geotiff`; the service refuses |lat| ≥ 90 and whole-globe requests at `resolution=low` or finer as "too many nodes"): 2,560 × 1,986 nodes, EPSG:4326, 0.141° × 0.081° spacing, sha256 `744fc7b01aa6b4428dc3a46ea4fe594b2ed8c5a19863a5394e7ae31b1ec430be`, kept on scratch (`/scratch/users/scdobbs/tmp_r23/gmrt_25km_topo.tif`, not in the repository). Rendered as a light grayscale (ocean 0.93 → 0.69 from 0 to 8,000 m depth, land 0.66 → 0.46 from 0 to 5,000 m), extended from ±80° to the poles by repeating the edge rows so the Equal Earth outline is complete (the only part of the basemap that is not GMRT data). Cite Ryan et al. (2009) in the caption. Equal Earth with central meridian 130°W (sites span 145°E to 45°W), graticule every 60° longitude and 30° latitude with labels.

**Inset (added, not in the directive).** Seven sites lie within a few degrees of each other on the north-east Pacific margin (lu08 Pescadero Basin, lu10, lu11, lu12 Cal DIG, lu13 Davidson, lu15 Gorda/Cascadia, Escanaba) and overlapped completely at the global scale, so a 22 mm inset (130–104°W, 21–47°N, same basemap, box drawn on the main map, no labels) was added in the lower right of the map. Remove the `INSET` block in `fig1.py` if unwanted; the global markers are unchanged.

## 6. Plumbing changes, one line each

- `pip install --only-binary=:all: cartopy` (0.24.1) and `cmcrameri` (pip 1.10; the package reports `__version__` 1.8, the Scientific colour maps version) into `~/bathy_diffusion_env`; the source build of cartopy failed on the login node and was not attempted in a job because the wheel installs cleanly.
- Liberation Sans 2.1.5 installed in `~/.local/share/fonts/` (§2).
- `fig_style.py` sets `matplotlib._mathtext.SHRINK_FACTOR = 6/7` so sub/superscripts of 7 pt labels (s_lr, m⁻¹, 10ⁿ ticks) render at 6.0 pt instead of matplotlib's default 4.9 pt; verified by the SVG check (smallest text 6.0 pt in all three files).
- Depth colour bars show depth positive down (−elevation of the harmonized grids), label "depth (m)".
- Fig. 2 y labels: Table S1 `region` where it is specific (Kermadec Arc, Costa Rica Rift, EPR 9°N, Pescadero Basin, TAG / Mid-Atlantic Ridge 26°N, Davidson Seamount, Gorda / Cascadia); where Table S1 says only "Pacific" (lu07, nu00) the manifest site name is used (Loihi Seamount, Izu-Bonin-Mariana arc). Table S1 could carry these names.
- Fig. 2b legend (three models cannot be shown on an axis) sits in the empty pooled row of panel b, no frame; Fig. 1a legend (three marker types) sits below the map in one row, no frame.
- Fig. 3c: the ten Goff-defined units keep their panel-b marker (filled computed, open possibly interpolated) in black; the six Goff-undefined units are light-gray squares, so the distinction does not rely on colour alone.
- Fig. 3a / Fig. 2: unit values are the frozen per-unit numbers (R08 `wide_window/scores.json` for 3a; R22 `table_S2_unit_CIs.csv` panel A for 2a/2b; `A/R10b/rehearsal/triples.json` `D1_ten_C1.whole` for the pooled row; `A/R09/scores/scores_r09.json` for the frozen CI of the median; `A/R15/out_ruling/units.csv` for Goff, A_fine and s_lr medians and the 16 unit centroids; footprint.geojson centroids for the seven blind-test sites).
- The first job (48817527) failed only because my `pdffonts` parser misread the "CID TrueType" column; the parser was fixed and the three figures rebuilt; later rebuilds adjusted layout only (map margins so the latitude labels are not clipped, legend moved out of the 60°S label, spectrum labels moved off the lines, Fig. 3 left margin and panel-c tick labels, inset).
- Legibility pass on the 300-dpi previews: no remaining text collisions; residual marker touching on the global map (TAG and RC2511, 2.6° apart; KN161L04 nearby) and in Fig. 3 (lu07/lu10 in panel a, lu10/vu02 in panel c) is partial and the shapes stay distinguishable.

## 7. Data csv contents

`Dobbs_Figure1_data.csv` (646 rows): 23 site coordinates with sources; 4 raster rows with file, column and colour limits; 523 per-tile s_lr values with tile ids; 96 spectrum bins (frequency, normalised PSD, raw routine value). `Dobbs_Figure2_data.csv`: 9 unit rows with ρ and CI, the pooled row, and 27 model-minus-s_lr rows. `Dobbs_Figure3_data.csv`: 16 + 10 + 16 rows with x, y, group and source column. Meta files `fig{1,2,3}_meta.json` hold the assertion records, candidates, pdffonts output and file hashes.

## 8. Revision after Steve's notes (2026-10-21, same day; jobs 48820498–48820999 on dev)

1. **Fig. 1a legend** now has four entries: filled circle "Development unit, real relief", open circle "Development unit, artifact-affected", triangle "Blind test 1", square "Blind test 2" (two rows under the map; the panel row of Fig. 1b moved down 3 mm, figure height 131 mm).
2. **Fig. 1b colour maps** changed from batlow: depth panels now use Crameri **navia** (reversed; pale yellow shallow → green → dark blue deep, colour-vision safe, perceptually uniform), the per-tile s_lr panel uses Crameri **lajolla** (reversed; pale yellow smooth → dark red rough); the fine-field panel keeps **vik**. This departs from the directive's "batlow or oleron" at Steve's request; oleron's ocean half was tried first and was too low in contrast for the 400 m depth range. One line in `fig1.py` (`cmap_d`, `cmap_s`) switches them back.
3. **Fig. 1c ship spectrum.** The gray band is the target band [f_lr, k·f_lr] = [0.0152, 0.2425] cycles m⁻¹ (ship sampling frequency 1/65.99 m to 16 times it), not the Nyquist frequency. In the first version the ship curve ran through the band because the frozen k-sweep routine first resamples the ship grid bilinearly onto the 1 m AUV grid, and that interpolation carries steeply decaying power above f_lr (interpolation leakage, no ship information). The ship spectrum is now computed with the same routine (`radial_psd`, 48 log bins, Hann window) on the ship grid at its native 65.91 m cell (87 × 67 cells), so it ends at the ship Nyquist 0.0076 cycles m⁻¹ = f_lr/2, left of the band; where the two spectra overlap (10⁻³ to 7.6 × 10⁻³ cycles m⁻¹) they agree, which is the co-registration check one would hope to see. The low-frequency end of the ship curve is noisy because the 87 × 67 ship grid gives only a few FFT cells per log bin there. The AUV spectrum is unchanged (routine on the AUV-native 1024 × 1024 common grid). Data csv series renamed "Ship PSD (native grid)"; Welch factor per grid.
4. **Fig. 3 legends** added, no frames: (a) filled "real-relief unit" / open "artifact-affected unit", lower left; (b) filled "Goff computed" / open "Goff possibly interpolated", upper right (x axis widened to 0–400 m and the shared y axis given headroom to 10¹ m so the legend sits clear of the points); (c) gray square "no Goff value", lower right. Marker shapes unchanged.
5. **Fig. 3 panel (c), text for the caption** (the current caption folds (b) and (c) together): "(c) The same unit-median fine-band amplitude as in (b) against unit-median s_lr for all 16 development units, on logarithmic axes; the ten units with a Goff (2020) value are drawn as in (b) and the six without one (off oceanic crust, or no value within 0.5° of the footprint: Kermadec Arc, Gulf of California margin NA090, Southern California, eastern North American margin, Izu-Bonin-Mariana arc, Gulf of Mexico) as gray squares. Spearman ρ(s_lr, A_fine) = +0.64 over the ten and +0.70 over all 16, against ρ(Goff, A_fine) = −0.31 in (b)." (The six Goff-undefined units are lu00, lu10, lu11, lu14, nu00, vu02; their `domain_class` in R15 `units.csv` is "no value".)
6. Fig. 2 unchanged. The P-V1 5-of-9 note in §3 stands.

