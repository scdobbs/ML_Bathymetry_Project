# SIGNOFF_phase5_prereg_v3_1 (DRAFT, unsigned)

The CNN repo's Phase 5 locks read the signed file. `phase5_common.read_signoff` (the imported `phase4_common.read_signoff`; constant `SIGNOFF = R16c/SIGNOFF_phase5_prereg_v3_1.md`) takes the first sha256 it holds and requires it to equal the pre-registration's. The acquisition repo's `sealed_step1_p5.py` lock 1 reads the sign-off and the pre-registration sha256 too (its `sbatch/acq_r10/step1.sbatch` must name v3.1 and this sign-off when ACQ-R10 is re-run). This draft is kept under the `_DRAFT` name, so no lock can pass before it is signed. To sign: fill the signature and date, then save it as `R16c/SIGNOFF_phase5_prereg_v3_1.md` and commit it. After that, tag `phase5-prereg-v3.1`.

| field | value |
|---|---|
| pre-registration sha256 | `75b72eb6fae3804c7dea61aa4171744a2fcd4fbdb7c3916f3fb98f2c06f2c7b6` |
| pre-registration | `directives_and_reports/R16c/PREREG_phase5_v3_1.md` (version 3.1, amended from the signed `R16/PREREG_phase5_v3.md`, sha256 `3df236e0…9d58`, tag `phase5-prereg-v3`, which stays untouched; v3.1 supersedes it) |
| design spec | `directives_and_reports/phase_5/PHASE5_design_spec_2026-10-13.md` (sha256 `7a1194efbd6a787f13273a3ca4cd71688f09d22e7e55ec541fd59cf9c5bf3aa7`; ruled 2026-10-13) |
| Phase 5 scripts (v3.1 B6; only `phase5_common.py` changed from v3) | `phase5_common.py` `cdca2c9b…`, `phase5_build.py` `973b775d…`, `phase5_figures.py` `47921a51…`, `phase5_predict.py` `96797b85…`, `phase5_score.py` `7eb7c1cc…` |
| Phase 4 scripts imported unchanged (v2.1 B6) | `phase4_common.py` `99407ec6…`, `phase4_build.py` `98370164…`, `phase4_figures.py` `44e87590…`, `phase4_predict.py` `7267a9c8…`, `phase4_score.py` `9aefb6ff…` |
| CNN tag | `phase5-prereg-v3.1` (to be set on the commit holding v3.1 and the signed sign-off) |
| acquisition freeze tag | `phase5-acq-freeze-v2` (supersedes `phase5-acq-freeze`) |
| acquisition `PHASE5/freeze_record_phase5_v2.json` sha256 | `f5a30b87c05a90bb9155f37bbdf61de6d20fa55423d97662e59ff42b02809438` |
| acquisition `src/phase5/sealed_step1_p5.py` sha256 | `83af2875e53a28ef9b01c95a7b533ef7befd7a00625dfa6b0fe87178a2accc47` (code commit `f88e21c`) |
| vertical-datum calibration | `ACQ-R11b/vcheck_calibration_v2.csv` sha256 `41973318d9ac55cffc7baa3f57a359da2538f2fd0a7a3a952d39770a14b7d8b2`; T_G = 0.200712 |
| acquisition supplement | `manifest/pairs_phase5_sealed.parquet`, not yet written (ACQ-R10); its sha256 goes into `CHANGE_LEDGER.md` before the Phase 5 build |
| signed by | signed for Steve (Stephen Dobbs) by the executor on his written instruction of 2026-10-13 |
| date | 2026-10-13 |

## Exposure statement (pre-registration section "Prior exposure")

Steve has seen no bathymetry of the eight sites. That includes the June QC figure for Bermuda and the La Jolla Zephyr survey (stated 2026-10-13). The only candidate numbers seen are the ACQ-R08 and ACQ-R08b metadata, listed in REPORT_ACQ-R08b §5:

- per grid: file names, sizes and sha256, grid size, cell size, CRS, extent, nodata and dtype;
- per candidate: the valid area, the footprint polygon and its area, one median depth rounded to 100 m, and the ship cruise, sonar, year and swath file count / size;
- coverage: navigation-track counts, the half-swath buffer, the coverage area and fraction;
- the estimates `lr_est_m`, `cells_across_est`, `ratio_est` and `tiles_est`;
- `same_site_as`, the nearest corpus footprint and status.

Per site:

| site | HR id | HR cell (m) | valid km² | depth, nearest 100 m | ship | lr_est (m) | cells across (est.) | tiles (est.) |
|---|---|---|---|---|---|---|---|---|
| Galápagos Spreading Center | MGDS:18210 | 1.0025 | 75.786 | 1700 | TN188 | 29.67 | 8.65 | 3917 |
| Escanaba Trough | MGDS:30368 | 1.1477 | 84.435 | 3300 | Escanaba | 57.6 | 5.1 | 3918 |
| Monterey Canyon | MGDS:24756 | 1.0 | 27.021 | 1400 | Monterey | 24.44 | 10.47 | 1611 |
| Mid-Atlantic Ridge 16.5°N | MGDS:22436 | 2.0033 | 90.606 | 3400 | KN161L04 | 118.69 | 4.32 | 1351 |
| La Jolla Canyon | MGDS:21415 | 1.0006 | 24.373 | 600 | MV1209 | 10.47 | 24.47 | 1211 |
| Bermuda | MGDS:31675 | 1.0002 | 9.159 | 1600 | KN173L02 | 55.86 | 4.58 | 527 |
| Chile margin | MGDS:31860 | 0.9988 | 7.149 | 2900 | MV1003 | 50.62 | 5.05 | 450 |
| Dorado Outcrop | MGDS:31954 | 1.0032 | 5.584 | 3100 | AT26-24 | 54.11 | 4.75 | 331 |

The ACQ-R08b metadata also cover MGDS:31811, the Galápagos 2 m product that is not used (depth 2700, nearest 100 m). The executor that ran ACQ-R08b saw only those numbers. The R16 executor opened no candidate file. It read REPORT_ACQ-R08b §5 to write this list.

Steve has also seen the development and known-bad G values of the vertical-datum check (ACQ-R11, ACQ-R11b: 45 development-designated rows, maximum 0.0669; NR07-1__MGDS_31813 / _31814 at 0.378 / 0.394; AT37-05 at 0.0273), and no value from any Phase 5 site. The R16c executor opened no candidate file and saw no Phase 5 value.

The blind artifact call of B3 step 3 is made against the R08 §3.1 criteria with this exposure stated.

[ ] confirmed as written      [ ] corrected as follows:

## Procedure note (carried from v2.1)

The amplitude statistic and the figure key exist after B3 step 2. Steve does not see either until his blind calls are committed. The step-2 report carries no amplitude and no key.

## Declaration

I have read `PREREG_phase5_v3_1.md` at the sha256 above and sign it as the pre-registration for Phase 5; it supersedes v3. Phase 5 runs in the order of its B3 and reports every result, whatever it is.

Signature: signed for Steve (Stephen Dobbs) by the executor on his written instruction of 2026-10-13. Date: 2026-10-13
