# DEVIATIONS_phase5: departures from pre-registration v3.1 as written (Phase 5)

Written 2026-10-14 by directive R19 from the existing reports and ledgers; no new analysis. "Acquisition ledger" is `co-located-auv-ship-based-multibeam-acquisition/CHANGE_LEDGER.md` (dated rows); "ledger" is this repo's `grl_reviews-and-prompt-reports/CHANGE_LEDGER.md`.

1. **Governing document.** Pre-registration v3.1 (`R16c/PREREG_phase5_v3_1.md`, sha256 `75b72eb6fae3804c7dea61aa4171744a2fcd4fbdb7c3916f3fb98f2c06f2c7b6`, signed, tag `phase5-prereg-v3.1`, ledger row 49) governed every Phase 5 step. By Steve's ruling of 2026-10-14 (stated in ACQ-R10c, ACQ-R13, R17 and R18) no later amendment, sign-off or dress rehearsal was made. ACQ-R12 had proposed a v3.2 to cite freeze v3; it was never written. The departures below were made under Steve's rulings in place of amendments (REPORT_R17, REPORT_R18; ledger rows 50–52).

2. **Acquisition freeze.** v3.1 cites freeze v2 (`freeze_record_phase5_v2.json`, `f5a30b87…9438`, entry script `83af2875…c47`).
   - **v3** (ACQ-R12; `42cb8ebb…a85b`, entry `f1b6ada0…f296`, tag `phase5-acq-freeze-v3`) changed write paths so that sealed-token files stay under admitted roots, wrote a `stopped` record on every exception, and added the in-run leak check. Rehearsal outputs were bit-identical to ACQ-R11b. No computation changed.
   - **Leak-check scope ruling** (Steve, 2026-10-14, in-session, ACQ-R12): **"records only"**. The value search covers every text file, parquet text and log the run writes. The forbidden-key check covers the records the step writes. The frozen contract-v2.1 product files are exempt from the key check because they hold offset and sigma fields by design; they are still value-searched and are on the do-not-open list.
   - Step 1 ran once under v3 (ACQ-R10c, job 47786827).
   - **v4 → v4.1 → v4.2** (ACQ-R13) implemented item 3 below and two fixes of it. The final freeze is v4.2: `freeze_record_phase5_v4_2.json` `168df0c68cf0e4493b62c78be6da2755020ce3bd53c9bd1f2ed02ad9270eadc3`, entry script `2c9ea848…73a7`.
   - Sources: REPORT_ACQ-R12, REPORT_ACQ-R10c, REPORT_ACQ-R13; acquisition ledger rows ACQ-R12, ACQ-R10c, ACQ-R13 (2026-10-14).

3. **The 30 m co-registration bound replaced (ACQ-R13).** Step 1 under v3 (ACQ-R10c) gave **2 of 8 eligible** (Bermuda, Dorado). Four sites (TN188, Escanaba, KN161L04, MV1003) failed only on the 30 m rigid-offset bound; MV1209 failed the vertical-datum check and Monterey had no k. The pooled criterion needs at least 3, so it was not testable. Steve then ruled ("poor processing must not exclude good data") that the bound is replaced by a direct test of the co-registration.
   - **Timing:** the ruling came after the step-1 eligibility counts were known, and before any CNN build, page, prediction or score existed for any site.
   - **Rule:** after the frozen global rigid shift, split the HR/ship overlap 2 × 2 into quadrants (merging a quadrant that fails the frozen QA with its neighbour), and estimate each quadrant's shift with the frozen estimator.
     - **Rigid case:** every quadrant agrees with the global shift within one ship grid cell. The global shift is applied unchanged.
     - **Drift case:** otherwise each group gets its own shift (residual rounded to whole HR pixels), must pass the frozen QA except the 30 m bound, and lattice tiles crossing a group boundary get `joint_valid = 0`.
     - In both cases the frozen vertical-datum check (T_G = 0.200712) must pass. No other threshold changed.
   - **Development check:** NA080__MGDS_31289, KM0923__MGDS_31429 and FK151121__MGDS_22383 were all rigid (`quickcheck_r13.json`); the tolerance stood.
   - **Runs:**
     - Run **47792880** (v4) is complete but invalid for the four drift sites: an inherited `_FillValue` made the drift write fail, and the error was recorded as "not estimated". Its record (`…T165859Z`) and `pairs_phase5_sealed_v4.parquet` are kept as superseded.
     - Run **47814252** (v4.1) stopped at site 1 on the forbidden-key check (`residual_px`, renamed in v4.2).
     - Run 47819657 (v4.2) is the one used.
   - **Eligibility after: 5 of 8.**
     - Escanaba and MV1003 pass in the drift case; KN161L04 passes in the rigid case.
     - TN188 fails (drift case, one quadrant group fails QA).
     - MV1209 still fails the datum check; Monterey still has no k.
   - The rigid sites' rasters are bit-identical to ACQ-R10c.
   - Sources: REPORT_ACQ-R13, supplement `pairs_phase5_sealed_v4_1.parquet` `6fe0d590…49f0`, run record `step1_run_phase5_20261014T204117Z.json` `947ede38…763e`; ledger row 50.

4. **22 Escanaba boundary tiles excluded (R17 H1).** The frozen CNN tiler reads `joint_valid.tif` only in its window test (≥ 50 % of the 256-px window); its target mask is `hr_valid ∧ lr_valid`.
   - Every window lying wholly inside a zeroed lattice tile was excluded (Escanaba 37, MV1003 9).
   - **22 kept Escanaba tiles** straddled zeroed tiles, with 25–75 % of their target pixels inside them. **None was in Escanaba's primary interior view** (248 tiles); all 22 were in its fallback / all-tile sets. MV1003 had none.
   - Steve ruled option (b) as the implementation of ACQ-R13's rule that tiles crossing a quadrant boundary are dropped, with no amendment and no rebuild.
   - The fixed list `R17/exclusions_phase5_r17.json` (sha256 `e0a620a03b3bba951ea63fa8aa06da8897fbbcdcb586070fb0e5e25b395bd095`, commit `45804b9`) was applied in the figures (tiles dropped, footprints masked in copies of the field mosaics) and in every scored row (22 dropped at Escanaba, 0 elsewhere).
   - Sources: REPORT_R17 (H1 and its continuation), REPORT_R18 bullet 1; ledger rows 50, 52.

5. **Script hash locks relaxed for three scripts (R17, R18).** v3.1 B6 pins all five Phase 5 scripts by sha256.
   - `phase5_common.py`, `phase5_figures.py` and `phase5_score.py` are instead accepted when the file at tag `phase5-prereg-v3.1` hashes to the B6 value. The line diff against it is written into every provenance record (`pinned_diff`, `PINNED_DIFF_OK`). `phase5_build.py`, `phase5_predict.py` and the five Phase 4 scripts stayed strictly hash-locked.
   - The l.134 freeze-sha text check now requires the freeze sha that v3.1 states (`f5a30b87…`) and records v4.2 as the one used. This relaxation was made in R17; R18 did not change it.
   - Purpose of each diff:
     - **`phase5_common.py`** (final `e71324be…`): point the supplement, freeze and entry-script constants at ACQ-R13 / v4.2. Relax the l.134 check. Add the tagged-file-plus-diff check (`git show` run with `cwd=`, because the nodes' git 1.8.3 has no `-C`). Add `excluded_tiles()` for the R17 list. (R17 plumbing, R18 `PINNED_DIFF_OK` entry.)
     - **`phase5_figures.py`** (`60cb5b3d…`): apply the R17 exclusion list to the pages (R17 H1 ruling).
     - **`phase5_score.py`** (`e0e48e18…`): apply the exclusion list to every row, compute the declared S1 / S2 rows (item 6), and label MV1003's GBT_ship-derived rows unavailable (item 7). No value is changed. (R18.)
   - Sources: REPORT_R17 bullet 4 and continuation, REPORT_R18 bullet 8; ledger rows 50, 52; Steve's in-session approval of the first lock change (R17).

6. **Declared supplementary rows (R18).** The R18 directive declared two descriptive rows before any prediction or score existed. Neither is the primary outcome.
   - **S1, original eligibility** (the 30 m-eligible sites Bermuda and Dorado; label "not testable under the original rule" if fewer than 3).
   - **S2, rigid co-registration only** (without the drift sites Escanaba and MV1003).
   - Both were computed inside the single scoring run with the pinned `pooled_row` on each subset of the C1 set. Results: S1 +0.738 [+0.424, +0.872] on 2 sites, not testable under the original rule; S2 +0.393 [+0.180, +0.522] on 3 sites.
   - Sources: REPORT_R18 bullet 7; ledger row 52.

7. **MV1003 GBT_ship unavailable.** MV1003's ship products are unusable (`ship_products_usable = False` in the supplement), and the build recorded its product features as masked (REPORT_R17 bullet 3). The primary uses s_lr from the LR grid only and is unaffected.
   - The scorer still computed GBT_ship with masked features. By the R18 directive, MV1003's GBT_ship row and every row derived from it (GBT_ship − s_lr, P-V1 − GBT_ship, all-tile GBT_ship) are reported as **"unavailable (ship products unusable)"**.
   - MV1003 is in the C1 set, so the pooled GBT_ship rows are unavailable too. The values remain in `results_D1.json` with the label (`C2.unavailable`).
   - Sources: REPORT_R18 bullet 6; ledger row 52.
