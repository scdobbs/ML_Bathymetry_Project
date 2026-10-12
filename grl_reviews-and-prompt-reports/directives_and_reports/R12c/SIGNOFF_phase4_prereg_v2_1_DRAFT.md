# SIGNOFF_phase4_prereg_v2_1 (DRAFT — unsigned; to be saved as `R12c/SIGNOFF_phase4_prereg_v2_1.md` once signed)

Both repositories' Phase 4 locks read the signed file. The CNN repo's `phase4_common.read_signoff` (revision v2.1 constant
`SIGNOFF = R12c/SIGNOFF_phase4_prereg_v2_1.md`) takes the first sha256 it holds and requires it to equal the pre-registration's.
The acquisition repo's `sealed_step1.py --signoff <file> --prereg-sha256 <sha256 below>` requires the same digest to appear in it.
This draft is kept under a `_DRAFT` name so that the lock cannot pass on an unsigned file (R11 used the same template pattern).

| field | value |
|---|---|
| pre-registration | `directives_and_reports/R12c/PREREG_phase4_v2_1.md` (revision v2.1 of `R11/PREREG_phase4_v2.md`, sha256 `9795db3b…`, tag `phase4-prereg-v2`, which stays untouched) |
| pre-registration sha256 | `d6af268bc8211fe9b4ae17439a47e6fdcbe7e5949d8ab387a7be080b2c748334` |
| `phase4_common.py` sha256 (the only script changed; B6 row of v2.1) | `99407ec6ce43f2f4005130d67c9025f229df4de20c9560ee79b4bf713e805111` |
| the other four Phase 4 scripts | unchanged from v2 B6 (`phase4_build.py` `98370164…`, `phase4_figures.py` `44e87590…`, `phase4_predict.py` `7267a9c8…`, `phase4_score.py` `9aefb6ff…`) |
| CNN tag | `phase4-prereg-v2.1` (to be set on the commit holding v2.1 and this sign-off, R12c Part E1.2) |
| acquisition freeze tag | `phase4-acq-freeze` (unchanged) |
| acquisition `freeze_record.json` sha256 | `331c94f01d7d1a2e52fb969c5fb1872aae2cd2ed249a3757b99e9d217b3ac0cb` (unchanged) |
| acquisition `src/phase4/sealed_step1.py` sha256 | `bd773cb77eaff1786efe48c8d92210d69a90f885545cd00a07af9b35cd8753c1` (unchanged) |
| acquisition supplement `manifest/pairs_phase4_sealed.parquet` sha256 | `6968e04554ddeef3c335587d602cb5fab2185db32b46cbd152475a919eb59206` (unchanged; ledger row 35) |
| acquisition citation record `step1_run_phase4_citation.json` sha256 | `6e51efd61f36f92066ff91de2749cdd7ce85192c6f8d20611c1a3de45ffa431e` (unchanged; ledger row 37; deviation note 02) |
| signed by | |
| date | |

## Exposure statement (pre-registration section "Prior exposure", carried forward)

Before v2 was frozen, Steve had seen the sealed pairs' bathymetry three times, and no score:

- **The original co-registration sign-off,** which covered the lockbox pairs.
- **The ACQ-R03 co-registration review** of nu01a and nu03: hillshades before and after the shift, contours and the
  search surface (ruling R6, 2026-10-01).
- **A re-view on 2026-10-10** of the harmonized AUV and ship bathymetry of the sealed pairs (depth and hillshade). He
  noted that nu03 shows very low relief, with survey track lines as the only visible structure. No fine-band,
  difference or roughness layer was viewed, and no rule was changed as a result.

Since v2 was signed, Steve has seen only step-1 metadata (spacings, k, product usability) and lock and stop reports; no HR
statistic, amplitude, tile count, target, feature, prediction or image of a sealed pair.

The blind artifact call of B3 step 3 is made against the R08 §3.1 criteria with this exposure stated.

[ ] confirmed as written      [ ] corrected as follows:

## Errata (wording only; carried forward unchanged from the v2 sign-off; no rule, number or lock affected)

The pre-registration is signed at the sha256 above. Where these wording points and the hash-pinned code differ, the code
governs.

1. **B2.1 describes s_lr loosely.** It says "SD of the valid LR cells of the 256-px window". The pinned code
   (`extract_clean_patches.py`: `tile_ship_side` + `build_inputs`) first area-averages the LR raster onto
   `lr_px = max(2, round(256 × hr_res / lr_native))` cells across the window, then takes the SD of the valid cells
   (at least 2, floored at 0.5 m) (REPORT_R11 §2, note 5).
2. **B2.2 still calls the configuration question a HOLD.** It was ruled on 2026-10-09: cfg 18 is kept.
3. **B3 step 1: "Its sha256 is recorded in `CHANGE_LEDGER.md`"** refers to the supplement
   `manifest/pairs_phase4_sealed.parquet`.

## Procedure note

The amplitude statistic and the figure key exist after B3 step 2. Steve does not see either until his blind calls are
committed. The step-2 report carries no amplitude and no key.

## Declaration

I have read `PREREG_phase4_v2_1.md` at the sha256 above and sign it as the pre-registration for Phase 4. Phase 4 runs in
the order of its B3 and reports every result, whatever it is.

Signature: ________________________________________  Date: ____________
