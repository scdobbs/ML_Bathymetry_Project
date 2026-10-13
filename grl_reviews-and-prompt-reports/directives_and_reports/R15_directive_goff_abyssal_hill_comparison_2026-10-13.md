# DIRECTIVE R15 (CNN repo, CPU only): site-level comparison with Goff's predicted abyssal hill RMS heights

**Date:** 2026-10-13
**Repo:** CNN repo, `grl_reviews-and-prompt-reports/directives_and_reports/`
**Report to:** `R15/REPORT_R15_<date>.md`
**Certainty labels:** CONFIRMED / CONTRADICTED / COULD NOT CHECK.
**Report length:** at most 8 bullets, plus the unit table and at most 3 HOLDs.
**Status of the analysis:** DESCRIPTIVE, requested by the reviewer, not pre-registered. It changes no Phase 4 result.

## 0. Purpose

The reviewer asked for a cross-site comparison with Goff (2010) gravity-derived abyssal hill roughness. Goff's
prediction is resolved to about 1 degree and estimates abyssal hill RMS height (hills 2 to 10 km wide). Our target
is fine-band relief within a site (metres to about 100 m). The comparison is therefore at site level only: across
development units, does Goff's predicted RMS height track our unit-median fine relief, and does it add anything
beyond unit-median s_lr? It cannot test within-site ranking.

**Development units only.** The 16 development units of manifest v2.1 (lu00, lu02, lu04, lu05, lu06, lu07, lu08,
lu09, lu10, lu11, lu12, lu13, lu14, lu15, nu00, vu02), with the ten real-relief units and the six
artifact-affected units labelled as ruled. EW0207 is reported in a separate row. No sealed, lockbox or second-test
candidate site is sampled. The R09 guard is installed and self-tested (702 / 702) in every process.

**Challenge clause.** If any instruction here is wrong or underspecified against the files or the code, stop at
that point, give the evidence, and report it as a HOLD. Do not work around it.

## 1. Inputs

1. **Goff grid.** Steve placed `R15/inputs/goff/abyssal_hill_rms_prediction_v2.0.grd` and
   `R15/inputs/goff/grl60637-sup-0001-2020gl088162-si.docx` in the repo. The grid is Data Set S1 of Goff (2020,
   GRL, 10.1029/2020GL088162), the update of Goff (2010) built on marine gravity v28.1, water depth from Sandwell
   et al. (2014) and GlobSed sediment thickness; it is also distributed on EarthByte.
   - Record the sha256 of both files. Commit the SI document only. Do not commit the grid: add its path to
     `.gitignore` and copy it to OAK `grl_review_archive/R15/inputs/` with `SHA256SUMS` verified.
   - Read the grid header and report: format, node registration, node spacing, extent, CRS, variable name, units,
     NaN convention, and any additional layers (for example a mask or a computed / interpolated flag).
2. **Masking context.** The SI states the prediction excludes large seamounts and mid-ocean ridges by mask, and is
   interpolated across gaps smaller than 4 degrees. A unit inside a masked feature can therefore carry an
   interpolated value rather than a computed one. If the grid carries a layer that separates computed from
   interpolated nodes, use it. If not, classify each unit with the objective proxies of §2.3.
3. **Proxy data (public, small, for §2.3 only).** The latest Muller et al. global crustal age grid (EarthByte) and
   the Kim and Wessel (2011) global seamount census. Record source URL, version and sha256 for each.
4. **Our unit values.** Unit-median fine relief and unit-median s_lr exactly as used in R08 §4 (the site-level
   result, ρ = 0.70 [0.31, 0.87] across 16 units), read from the R08 tables or recomputed from the frozen inputs
   with the R08 code. Also the R08b interior-tile unit medians for the ten real-relief units.

## 2. Sampling and classification

1. **Unit location.** For each unit, the union of its pairs' high-resolution footprints from the manifest, its
   centroid, area and median depth (from the frozen ship grids, rounded to 10 m).
2. **Goff value.** Per unit: (a) the bilinear value at the centroid; (b) the median of grid nodes within 0.5
   degrees of the centroid (one 1-degree cell), with the count of non-NaN nodes. Use (b) as the primary value.
3. **Domain class** (stated before sampling; applied mechanically):
   - `computed`: the grid's own flag says so, or, without a flag, all of: on oceanic crust (age grid not NaN),
     crustal age at least 1 Myr, and no Kim and Wessel seamount whose centre lies within twice its radius of the
     unit footprint;
   - `possibly interpolated`: a value exists but the unit fails one of those proxies (give which);
   - `no value`: the 0.5-degree neighbourhood has no non-NaN node.
   Report crustal age, distance to the nearest seamount and its radius, and on or off oceanic crust for every unit.

## 3. Comparison

All statistics across units, Spearman ρ, with a bootstrap over units (2,000 resamples, `default_rng(20261014)`),
95 % percentile CI.

1. **Primary set:** units classed `computed`. **Secondary set:** `computed` plus `possibly interpolated`. Report
   each set's size first. If a set has fewer than 5 units, report its table only, with no correlation.
2. On each set:
   1. Goff value against unit-median fine relief;
   2. Goff value against unit-median s_lr;
   3. unit-median s_lr against unit-median fine relief, recomputed on the same units (for comparison with 1);
   4. partial ρ of Goff with fine relief given s_lr.
3. Repeat 2.1 to 2.3 on the ten real-relief units with the R08b interior-tile medians, where the set allows.
4. **Magnitudes:** per unit, Goff RMS height (m), unit-median fine relief (m) and their ratio, descriptive only.
   State the bandwidth difference beside the table (Goff: abyssal hills, about 2 to 10 km; ours: the fine band of
   each pair, given as its wavelength range in metres from f_lr and k).
5. **One figure:** Goff value (x) against unit-median fine relief (y), log axes, units marked by domain class and
   by real-relief or artifact-affected, with unit labels. Write it as a PDF and a PNG.

**Must hold:** item 2.3 on all 16 units reproduces R08 §4 (ρ = 0.70 [0.31, 0.87]) within 0.01, or stop and report.

## 4. Report

1. **Grid facts** from §1.1, and whether a computed / interpolated flag exists.
2. **Unit table:** unit, real-relief or artifact-affected, setting, centroid, depth, crustal age, nearest seamount,
   domain class, Goff value (both samplings), unit-median fine relief, unit-median s_lr, ratio.
3. **Correlation table** for §3.2 and §3.3, with set sizes.
4. **A two-sentence reading** in the terms of §3 only: whether Goff tracks fine relief across the units where it
   is defined, and whether it adds to s_lr. No other interpretation.
5. The figure's path.

## 5. Preservation and record

Commit scripts, tables, the figure, the SI document and the report by explicit path; never `git add -A`. One
`CHANGE_LEDGER.md` row ("descriptive, reviewer request; not pre-registered"). Copy inputs (with the grid), tables,
figure, logs and report to OAK `grl_review_archive/R15/` with `SHA256SUMS` verified.

## 6. Out of scope

- Any sealed, lockbox, Phase 4 or second-test candidate site.
- Any change to frozen targets, features, models or Phase 4 records.
- Within-site comparison with Goff (the grid is about 1 degree).
- Refitting or tuning anything on Goff values.
- Manuscript text.

## 7. Not pursued

- Goff's other statistical parameters (characteristic width, length, azimuth).
- Comparison with the 2010 (v1) grid.
- Spreading-rate or crustal-age dependence of our fine relief.
