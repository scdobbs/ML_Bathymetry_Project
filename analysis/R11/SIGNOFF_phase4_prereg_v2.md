# SIGNOFF_phase4_prereg_v2

Both repositories' Phase 4 locks read this file. The CNN repo's `phase4_common.read_signoff` takes the first sha256 it
holds and requires it to equal the pre-registration's. The acquisition repo's
`sealed_step1.py --mode phase4 --signoff <this file> --prereg-sha256 <sha256 below>` requires the same digest to appear
in it.

| field | value |
|---|---|
| pre-registration | `directives_and_reports/R11/PREREG_phase4_v2.md` |
| pre-registration sha256 | `9795db3b22d6c54f7a85618dbaccafa054e88487b33236a7b2083ecb8c605979` |
| CNN tag | `phase4-prereg-v2` |
| acquisition freeze tag | `phase4-acq-freeze` |
| acquisition `freeze_record.json` sha256 | `331c94f01d7d1a2e52fb969c5fb1872aae2cd2ed249a3757b99e9d217b3ac0cb` |
| acquisition `src/phase4/sealed_step1.py` sha256 | `bd773cb77eaff1786efe48c8d92210d69a90f885545cd00a07af9b35cd8753c1` |
| signed by | Steve |
| date | 2026-10-11 |

## Exposure statement (pre-registration section "Prior exposure")

Before this version was frozen, Steve had seen the sealed pairs' bathymetry three times, and no score:

- **The original co-registration sign-off,** which covered the lockbox pairs.
- **The ACQ-R03 co-registration review** of nu01a and nu03: hillshades before and after the shift, contours and the
  search surface (ruling R6, 2026-10-01).
- **A re-view on 2026-10-10** of the harmonized AUV and ship bathymetry of the sealed pairs (depth and hillshade). He
  noted that nu03 shows very low relief, with survey track lines as the only visible structure. No fine-band,
  difference or roughness layer was viewed, and no rule was changed as a result.

The blind artifact call of B3 step 3 is made against the R08 §3.1 criteria with this exposure stated.

[x] confirmed as written      [ ] corrected as follows:

## Errata (wording only; recorded at sign-off, no rule, number or lock affected)

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

I have read `PREREG_phase4_v2.md` at the sha256 above and sign it as the pre-registration for Phase 4. Phase 4 runs in
the order of its B3 and reports every result, whatever it is.

Signature: ______Stephen Cole Dobbs (10/11/2026)________________
