<!-- generated from refusal_tests/refusal_tests.json (refusal_tests_r12c.py, derived from R11's; pre-registration under test: v2.1 d6af268bc821…) -->
| test | result | exception / note |
|---|---|---|
| 1 phase4_build.py: no sign-off file | passed | Refusal: LOCK sign-off: /scratch/users/scdobbs/grl_review/R12c/refusal_tests/r12c_refusal_zafxe6h6/absent.md does not exist |
| 2a phase4_build.py: wrong pre-registration sha256 (real v2, sign-off holding another sha256) | passed | Refusal: LOCK pre-registration: /home/users/scdobbs/GAN_Bathymetry_Project/grl_reviews-and-prompt-reports/directives_and_reports/R12c/PREREG_phase4_v2 |
| 2b phase4_build.py: wrong artifact hash (copy of v2 with the GBT hash altered) | passed | Refusal: LOCK artifact / code: GBT_ship sha256 ef93467262cf59c63bee8d5ebf898b6d9911204c72219f9bb0e2455258ff3f1f != 22222222222222222222222222222222222 |
| 1 phase4_figures.py: no sign-off file | passed | Refusal: LOCK sign-off: /scratch/users/scdobbs/grl_review/R12c/refusal_tests/r12c_refusal_zafxe6h6/absent.md does not exist |
| 2a phase4_figures.py: wrong pre-registration sha256 (real v2, sign-off holding another sha256) | passed | Refusal: LOCK pre-registration: /home/users/scdobbs/GAN_Bathymetry_Project/grl_reviews-and-prompt-reports/directives_and_reports/R12c/PREREG_phase4_v2 |
| 2b phase4_figures.py: wrong artifact hash (copy of v2 with the GBT hash altered) | passed | Refusal: LOCK artifact / code: GBT_ship sha256 ef93467262cf59c63bee8d5ebf898b6d9911204c72219f9bb0e2455258ff3f1f != 22222222222222222222222222222222222 |
| 1 phase4_predict.py: no sign-off file | passed | Refusal: LOCK sign-off: /scratch/users/scdobbs/grl_review/R12c/refusal_tests/r12c_refusal_zafxe6h6/absent.md does not exist |
| 2a phase4_predict.py: wrong pre-registration sha256 (real v2, sign-off holding another sha256) | passed | Refusal: LOCK pre-registration: /home/users/scdobbs/GAN_Bathymetry_Project/grl_reviews-and-prompt-reports/directives_and_reports/R12c/PREREG_phase4_v2 |
| 2b phase4_predict.py: wrong artifact hash (copy of v2 with the GBT hash altered) | passed | Refusal: LOCK artifact / code: GBT_ship sha256 ef93467262cf59c63bee8d5ebf898b6d9911204c72219f9bb0e2455258ff3f1f != 22222222222222222222222222222222222 |
| 1 phase4_score.py: no sign-off file | passed | Refusal: LOCK sign-off: /scratch/users/scdobbs/grl_review/R12c/refusal_tests/r12c_refusal_zafxe6h6/absent.md does not exist |
| 2a phase4_score.py: wrong pre-registration sha256 (real v2, sign-off holding another sha256) | passed | Refusal: LOCK pre-registration: /home/users/scdobbs/GAN_Bathymetry_Project/grl_reviews-and-prompt-reports/directives_and_reports/R12c/PREREG_phase4_v2 |
| 2b phase4_score.py: wrong artifact hash (copy of v2 with the GBT hash altered) | passed | Refusal: LOCK artifact / code: GBT_ship sha256 ef93467262cf59c63bee8d5ebf898b6d9911204c72219f9bb0e2455258ff3f1f != 22222222222222222222222222222222222 |
| 3 phase4_predict.py: no calls file | passed | Refusal: LOCK calls file: /scratch/users/scdobbs/grl_review/R12c/refusal_tests/r12c_refusal_zafxe6h6/absent_calls.csv does not exist |
| 4 phase4_score.py: calls sha256 not in the ledger (ledger holds the supplement sha256 only) | passed | Refusal: LOCK calls file: sha256 ad2059df5db74fc37a97f748ca3810104d40684c225fa4babf0f98e4e3f4deb6 is not recorded in /scratch/users/scdobbs/grl_review |
| 5 phase4_score.py: results file already exists | passed | Refusal: LOCK single run: results file already exists: /scratch/users/scdobbs/grl_review/R12c/refusal_tests/r12c_refusal_zafxe6h6/results_D1.json |
| 6 any script: pair not admitted (development pair in phase4 mode) | passed | Refusal: LOCK pair list: ['AR26__MGDS_31838'] not admitted in mode phase4 |
| 7 positive control: real v2 + dummy sign-off holding its sha256 + dummy supplement / ledger / acquisition record -> every lock passes; stopped on the  | passed | locks passed=True; safety net=True |
| 9 phase4_build.py: no supplement file | passed | Refusal: LOCK supplement: /scratch/users/scdobbs/grl_review/R12c/refusal_tests/r12c_refusal_zafxe6h6/absent_sup.parquet does not exist |
| 10 supplement sha256 not in the ledger | passed | Refusal: LOCK supplement: sha256 12105542c135ebe790e4f9424e0c0e5171354a9fad5ae2fc9640929e5319ebc5 is not recorded in /scratch/users/scdobbs/grl_review |
| 11a supplement with a wrong pair id | passed | Refusal: LOCK supplement: pair ids ['AR26__MGDS_31838', 'KN182L03__MGDS_30193', 'RC2511__MGDS_32239', 'SUM1004__MGDS_33090', 'TN159__MGDS_21981'] != t |
| 11b supplement with a missing pair id | passed | Refusal: LOCK supplement: pair ids ['AT42-03__MGDS_32007', 'KN182L03__MGDS_30193', 'RC2511__MGDS_32239', 'TN159__MGDS_21981'] != the five pre-register |
| 12 phase4_predict.py: EW0207 reference predictions altered by 1e-3 in log10 | passed | Refusal: LOCK environment control: EW0207 not reproduced: {"s_lr_identical": true, "gbt_max_abs_diff_m": 2.220446049250313e-16, "pv1_seedavg_max_abs_d |
| 13 build: KM0923 row with k emptied (rehearsal, scratch root) | passed | status=not scorable (no k); files built=['pair_stats.json']; fields=False; inputs=False |
| 13 figures: no page for the no-k pair (pair_stats status read by phase4_figures) | passed | skip=True |
| 13 score: no-k pair reported with status 'not scorable (no k)', outside C1 | passed | C1 reading on a pool holding only the no-k pair: not testable (C1 set empty) |
| 14 guard after the R10b tests: every sealed probe still blocked | passed | 702/702 |
| 27a phase4_build.py: no acquisition run record (empty records directory) | passed | Refusal: LOCK acquisition run record: no step1_run_phase4_*.json in /scratch/users/scdobbs/grl_review/R12c/refusal_tests/r12c_refusal_zafxe6h6/acq_emp |
| 27b phase4_build.py: acquisition run record citing a different freeze record sha256 | passed | Refusal: LOCK acquisition run record: no record in /scratch/users/scdobbs/grl_review/R12c/refusal_tests/r12c_refusal_zafxe6h6/acq_badfz is complete in |
| 27c phase4_build.py: acquisition run record citing a different supplement sha256 | passed | Refusal: LOCK acquisition run record: no record in /scratch/users/scdobbs/grl_review/R12c/refusal_tests/r12c_refusal_zafxe6h6/acq_badsup is complete i |
| 27d phase4_build.py: the acquisition repo's ACQ-R06b directory as it stands (only the stopped refusal-test record) | passed | Refusal: LOCK acquisition run record: no record in /home/users/scdobbs/co-located-auv-ship-based-multibeam-acquisition/reports_post_grl_review/ACQ-R06 |
| 27f phase4_build.py: the REAL supplement, ledger and ACQ-R06b directory (ACQ-R07c citation record) -> every lock passes, allow-list not applied | passed | accepted step1_run_phase4_citation.json 6e51efd61f36 |
| 27e phase4_build.py: freeze record on disk with another sha256 | passed | Refusal: LOCK acquisition run record: freeze record /scratch/users/scdobbs/grl_review/R12c/refusal_tests/r12c_refusal_zafxe6h6/PREREG_badart.md sha256 |
| 28 supplement row with ship_products_usable = false -> product features masked (TN399__MGDS_33068 harness row, rehearsal build) | passed | contract columns all NaN=True (2539 rows); pair_stats product_features=masked, ship_products_usable=False; frozen table (usable=true) holds finite pro |
| 30a B2: the five sealed HR paths sealed BEFORE the allow-list | passed | {'KN182L03__MGDS_30193': True, 'RC2511__MGDS_32239': True, 'SUM1004__MGDS_33090': True, 'AT42-03__MGDS_32007': True, 'TN159__MGDS_21981': True} |
| 30b B2: the five sealed HR paths admitted AFTER apply_allowlist(SEALED5) | passed | {'KN182L03__MGDS_30193': False, 'RC2511__MGDS_32239': False, 'SUM1004__MGDS_33090': False, 'AT42-03__MGDS_32007': False, 'TN159__MGDS_21981': False} |
| 30c B2: harmonized_confirmatory/nu99/FAKE__MGDS_00000/hr.tif refused after the allow-list | passed | is_sealed=True; tokens remaining ['harmonized_confirmatory'] |
| 30d B2: harmonized_confirmatory/hr.tif (no pair id) refused after the allow-list | passed | is_sealed=True |
| 30e B2: the other token families (real tokens of non-admitted pairs) still refused when only RC2511 is admitted; RC2511 admitted | passed | {'qc_plots AT42-03': True, 'raw_lr TN159': True, 'staging_phase1 MGDS_32007': True, 'SUM1004 hr (not admitted)': True} |
| 29 guard after every test: every sealed probe still blocked | passed | 702/702; allow-list () |
| **total** | 39 / 39 passed | node sh03-09n21.int; dummy sign-off deleted True |
