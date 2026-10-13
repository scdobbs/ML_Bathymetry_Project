# REPORT R15: site-level comparison with Goff's predicted abyssal hill RMS heights. §1 and §2 done; §3 stopped at its must-hold (HOLD H1)

**Directive:** `directives_and_reports/R15_directive_goff_abyssal_hill_comparison_2026-10-13.md`
**Executor:** Claude Code (Opus 5.5) on Sherlock, 2026-10-13. CPU only. **Status: DESCRIPTIVE, reviewer request, not pre-registered; changes no Phase 4 result.** Only the 16 development units and EW0207 (lu16, separate) were located and sampled. No sealed, lockbox or second-test candidate site was sampled. Every process ran `phase4_common.start` in rehearsal mode first (R09 guard installed and self-tested **702 / 702**, rehearsal locks passed, 38 development pairs + EW0207 admitted), then re-hashed the five R15 inputs.
**Compute:** serc CPU jobs, `goff_r15.py`: 47662670 (stopped at the §3 must-hold by design, rc 1); 47662876 (must-hold diagnostic added; failed on a seamount-count assertion, the header comment vs the file, see `tables/grid_facts.md`); 47663205 (complete through §2, rc 0, 94 s, 4 CPUs). Tables were generated on the login node in seconds by `report_tables_r15.py`.
**Outputs:** `/scratch/users/scdobbs/grl_review/R15/` (`out/grid_facts.json`, `must_hold.json`, `units.csv`, `units_record.json`, `stats.json`; `provenance/`; `logs/`).
**Certainty labels:** CONFIRMED / CONTRADICTED / COULD NOT CHECK. Statistics: Spearman ρ over units with a 2,000-resample percentile bootstrap over units, `default_rng(20261014)` (the R08 helpers `spearman_boot` / `partial_spearman`).

---

## 1. Bottom line (8 bullets)

1. **Goff grid facts (CONFIRMED; §1.1).** netCDF-3 classic (GMT 4.5.15), a single variable `z` (float32) on a gridline-registered geographic grid with 4-arcmin (0.0667°) spacing: x −180 to 179.933 (5,400 columns, full 360° with no duplicated seam), y −75 to 74.933 (2,250 rows). No CRS or units attribute. **No NaN node:** missing nodes are **−999** (7,347,438 of 12,150,000, 0.605), although `_FillValue` = NaN is declared. R15 treats −999 as missing. There are **5,533 negative values** among the valid nodes (minimum −16.2 m); none falls in any unit neighbourhood. The internal `title` attribute reads `gravhp28_hs.grd`. **There is no computed / interpolated flag or other layer**, so every class comes from the §2.3 proxies. sha256 of the grid `07421ead…d2f6`, of the SI document `6089367c…3004`.
2. **Proxies (CONFIRMED; §1.3).** Crustal age: EarthByte `age.2020.1.GTS2012.6m.nc`, the latest grid on the EarthByte server. Its README cites **Seton et al. (2020)**, with Müller as second author, not a "Müller et al." first-author grid. sha256 `3abeed4f…a39e`. Seamounts: Kim and Wessel (2011) KWSMTS v0.1, sha256 `91c93302…ea9a`, **24,643** entries. The file's header comment says 24,646; the 24 plate-segment headers sum to 24,643. Radius was taken as **major axis / 2**, because columns 4 and 5 are the basal ellipse's major and minor axes. URLs are in `tables/grid_facts.md`.
3. **Unit location (CONFIRMED, with a source note; §2.1).** The manifest's `footprint_wkt` is not WKT. It holds a stale `/scratch/groups/...` path for 33 of the 39 rows (that path no longer exists) and None for the other 6. Every pair has `<harmonized_path>/footprint.geojson` on the OAK data root (one polygon each, sha256 recorded per pair), and R15 used those. Centroid and area come from the union in a unit-centred Lambert equal-area projection. Depth is the median of the valid harmonized ship grid (`lr.tif` where `lr_valid`), pooled over the unit's pairs and rounded to 10 m.
4. **Domain classes (CONFIRMED as applied; §2.3).** **computed: lu02, lu06, lu07, lu08, lu15 (5)**. possibly interpolated: lu04, lu13 (seamount within 2r), lu05, lu09 (crust < 1 Myr; lu09 also has a seamount within 2r) (4). **no value: lu00, lu10, lu11, lu12, lu14, nu00, vu02 (7)**. EW0207: possibly interpolated (0.15 Myr; seamount within 2r), Goff (b) 56.2 m. Of the ten real-relief units, 3 are computed (lu07, lu08, lu15) and 7 are in the secondary set. Of the six artifact-affected units, lu02 and lu06 are computed and the other four have no value.
5. **The primary set sits exactly at the §3.1 floor of 5, and two of its members depend on a rule choice (CONFIRMED; `tables/sets.md`, `tables/multi_pair_units.md`).** lu15 is `computed` only through its union centroid. That point lies on 3.14 Myr crust 65 km from the Gorda pair and 175 km from the Cascadia pair. Sampled per pair, the Gorda pair is on 0.11 Myr crust with Goff (b) 86.0 m, and the Cascadia pair is off oceanic crust with no value (HOLD H2). lu07 changes class if radius = major axis, because KW-16547 lies 30.8 km from the footprint with a 15.9 km major axis (HOLD H3). Under either alternative the primary set is 4 units, below the floor (table only, no correlation). Under both it is 3.
6. **§3 must-hold NOT met (CONTRADICTED; `tables/must_hold.md`).** Item 2.3 on all 16 units with the directive's seed gives ρ = **+0.697 [+0.328, +0.872]** against R08 §4's +0.697 [+0.305, +0.866]. ρ is identical, but the CI lower bound differs by 0.023, beyond the 0.01 tolerance. The unit medians match the OAK-verified R08 `level_per_unit.csv` to ≤ 1e-12, and the same computation with R08's seed 20261009 reproduces R08 exactly. Across 100 seeds the lower bound ranges from +0.279 to +0.372, and only 20 of 100 seeds fall within 0.01 on both bounds. Per §3, R15 **stopped**: no correlation (§3.2, §3.3), no magnitude ratio (§3.4) and no figure (§3.5) were computed.
7. **Bandwidth difference (stated, not a test; §3.4).** Goff predicts the RMS height of abyssal hills about 2 to 10 km wide, resolved at about 1°. Our fine band runs from lr_native / k to lr_native, about **2.5 to 87 m** in wavelength across the development units (per unit in `tables/units.md`, e.g. lu07 5.4–86.9 m, lu12 5.0–10.0 m). The two bands do not overlap.
8. **Reading (§4.4): not given.** The two-sentence reading rests on the §3.2 correlations, which were not computed under the must-hold stop. The Goff (b) values and the unit medians appear side by side in the unit table only as §2 outputs. Preservation and record are in section 7 of this report.

**HOLDs for Steve (3).**

- **H1. The §3 must-hold cannot be met with seed 20261014 except by chance.** The directive asks for a new seed (20261014) and requires reproducing R08's bootstrap CI (seed 20261009) within 0.01. ρ reproduces exactly; the CI is Monte-Carlo-limited at n = 16 and matched within 0.01 for only 20 of 100 seeds. Options: **(a)** run the must-hold with R08's seed 20261009 (exact reproduction, 0.000 / 0.000) and keep 20261014 for every new R15 statistic; (b) apply the 0.01 tolerance to ρ only; (c) run all of R15 with 20261009. Recommendation: (a). It tests what the must-hold is for (same units, same values, same code) without making the new statistics depend on R08's seed. Once ruled, `python goff_r15.py` reruns §3 and writes the figure in about 2 minutes.
- **H2. lu15's union centroid represents neither of its sites.** The two pairs (TN299 Gorda, TN299 Cascadia) are 240 km apart. §2.1–§2.3 applied literally sample a point between them, which makes lu15 `computed` with Goff (b) 110.1 m. That value is neither the Gorda value (86.0 m, crust 0.11 Myr → possibly interpolated) nor the Cascadia value (no value, off oceanic crust). The unit-median fine relief and depth also pool the two sites: the pooled depth of 1,130 m is close to Cascadia's 1,087 m, against Gorda's 3,034 m. Options: (a) keep the literal union centroid (primary n = 5); **(b)** sample per pair, class the unit from its pairs that have a value (worst of those classes) and take the unit's Goff value as the median of the pair values with a value (lu15 → possibly interpolated, 86.0 m; primary n = 4, secondary n = 9); (c) drop lu15 from §3. Recommendation: (b), because the union centroid lies on neither site. Any multi-pair unit spanning more than about 0.5° would need the same rule; only lu15 does. lu07's pairs are 21 km apart, and lu12's 18 pairs lie within 54 km of their centroid, all off oceanic crust with no value.
- **H3. The meaning of "radius" in §2.3.** R15 used radius = semi-major axis (major / 2), the reading consistent with KWSMTS columns 4 and 5 being basal-ellipse axis lengths. The alternatives are radius = major axis (more conservative) or the mean of the semi-axes. Only lu07 is sensitive (30.8 km vs 2 × 7.95 = 15.9 km or 2 × 15.9 = 31.8 km). Under radius = major axis lu07 is possibly interpolated and the primary set is 4. Whether the census includes the subaerial Hawaiian edifices near lu07 (Lōʻihi / Kīlauea flank): COULD NOT CHECK from the file itself. Recommendation: keep major / 2 and report the sensitivity row, as done.

**Not pursued (one line each):** Goff's other parameters, the 2010 (v1) grid, and spreading-rate or age dependence (directive §7). No within-site comparison and no refitting on Goff values (§6).

---

## 2. Grid and proxy facts (§1.1, §1.3)

<!-- TABLE: grid_facts.md -->

## 3. Unit table (§4.2)

Ratios (§3.4) are withheld under H1. "fine band" = the union over the unit's pairs of [lr_native / k, lr_native] in metres (hr_res / f_hi to hr_res / f_lr). Interior D1 medians are from the R08b interior tile list (`tiles_interior.parquet`, OAK-verified) joined to the R05 tile table; lu14 has no interior tile.

<!-- TABLE: units.md -->

## 4. Set sizes (§3.1) and the must-hold (§3)

<!-- TABLE: sets.md -->

<!-- TABLE: must_hold.md -->

**Correlation table (§4.3): not computed** (must-hold not met; H1).
**Figure (§4.5): not produced** (§3.5 is part of the stopped comparison). Path once H1 is ruled: `R15/figures/R15_fig_goff_vs_fine_relief.{pdf,png}`.

## 5. Multi-pair units: pair-level spread (evidence for H2)

<!-- TABLE: multi_pair_units.md -->

## 6. Method notes

1. **Unit values (§1.4).** The unit medians of A_fine and s_lr were recomputed from the frozen R05 tile table (the R08 §4 code path: median over the unit's tiles) and asserted equal, to ≤ 1e-12, to the OAK-verified R08 `level_geometry/level_per_unit.csv`. The R08b interior tile A_fine was asserted equal to the R05 tile A_fine on every tile.
2. **Goff sampling (§2.2).** (a) Bilinear on the four surrounding nodes, NaN if any is missing. (b) The median of the nodes with |Δlon| ≤ 0.5° and |Δlat| ≤ 0.5° of the centroid (a 1° × 1° box of 15 × 15 = 225 nodes; longitude wraps at ±180°, which matters for lu00 at −179.03°). The grid orientation was checked: continental nodes are −999 and open-ocean nodes are valid.
3. **Proxies (§2.3).** Age is bilinear at the centroid on the 6-arcmin grid (NaN if any of the four nodes is NaN). Seamount distance is measured from each K&W centre to the footprint union (0 inside), in the unit's projection, for candidates within 6° of the centroid (the largest major axis is 116.5 km).
4. **Depth (§2.1).** The ship grids were read at native resolution (decimation 1 on every pair, ≤ 87 m cells). Pooling weights each pair by its valid cell count.

## 7. Preservation and record (§5)

Commit (explicit paths): `R15/scripts/`, `R15/tables/`, `R15/REPORT_R15_template.md`, this report, `R15/inputs/goff/grl60637-sup-0001-2020gl088162-si.docx`, `R15/.gitignore` (ignores the Goff grid and `inputs/proxies/`: the age grid, its README and the census are not committed), `CHANGE_LEDGER.md` row 44. OAK `grl_review_archive/R15/`: `inputs/` (with the grid, the age grid and the census), `out/`, `provenance/`, `logs/`, `tables/`, `scripts/`, `documents/`, each with `SHA256SUMS` verified at the destination.
