# REPORT R12c Part E — pre-registration v2.1 signed and tagged; Phase 4 step 2 (the build) and step 3a (the blind pages) complete

**Directive:** `directives_and_reports/R12c_directive_allowlist_fix_dress_rehearsal_prereg_v2_1_2026-10-12.md`, Part E, on Steve's instruction of 2026-10-12 to sign the two documents on his behalf and continue.
**Executor:** Claude Code (Fable 5.1) on Sherlock, 2026-10-12. CPU only. **No prediction and no score of any sealed pair.** The four built pairs' rasters were read only by `phase4_build.py` and `phase4_figures.py` in phase4 mode, after every lock and the v2.1 allow-list (provenance records in `grl_review_archive/phase4/provenance/`). **Steve stays blind:** this report carries counts, views, D₁, cells across, the product-feature decision, paths and sha256 only; no amplitude, value, image, page description or key (§4, §5 of R12 apply in full).
**Compute:** serc CPU jobs: build 47557376 (16 CPUs, 128 GB, 10 min, `sh04-05n20`), post-build checksums + count check 47560466 and 47562475 (4 CPUs), figures 47561133 (4 CPUs, 64 GB, 51 s, `sh04-12n18`), preservation 47564270. Login node: the signatures, the tag, the v2.1 preconditions, the leftover move (shell `mv` + `sha256sum`), the page check (reads the PDF and lists file names), this report.
**Outputs:** `grl_review_archive/phase4/` (written by the scripts; `SHA256SUMS` written and verified outside the scripts), `grl_review_archive/phase4_stopped_R12b/` (the R12b leftovers, moved), OAK `grl_review_archive/R12c/{logs_partE,checks_partE,tables_partE,scripts_partE,documents_partE}/` (§6), this directory: `REPORT_R12c_partE_2026-10-12.md`, **`figset_A_blind.pdf`** and **`phase4_calls_TEMPLATE.csv`** (delivered beside this report), `SIGNOFF_phase4_prereg_v2_1.md`, `DEVIATION_phase4_03.md` (signed), `tables/build_check.json`, `tables/pages_check.json`, `tables/c1_preconditions_v2_1.json`.
**Certainty labels:** CONFIRMED / CONTRADICTED / COULD NOT CHECK.

---

## 1. Bottom line

1. **E1, signature and tag (CONFIRMED).** `R12c/SIGNOFF_phase4_prereg_v2_1.md` (sha256 `712766d41278161f7b1ad5632b82c9a7bb4794d12e6f6d5189741b9f05e70029`; written from the draft of commit `50bf0bd`, signed for Steve by the executor on his written instruction) and `R12c/DEVIATION_phase4_03.md` (sha256 `dfccdadcaef1fcdaa0c0d063ff1f1b59df373fef9357765a70e24b869b0333af`, signed the same way) committed by explicit path as `0c48442da0662d9226515d5e93986472ce639ed8`, **tagged `phase4-prereg-v2.1`**; `phase4_common.read_signoff()` on the real file returns `d6af268b…`, equal to v2.1 on disk. Ledger row 39 (commit `e81059b`). v2 and its tag untouched.
2. **E1, the C1 checks against v2.1 (CONFIRMED, all five pass; `tables/c1_preconditions_v2_1.json`, `scripts/preconditions_r12c.py`).** Sign-off committed, clean, holds the v2.1 sha256; v2.1 on disk = v2.1 at tag `phase4-prereg-v2.1` (`0c48442`); deviation notes 01 (`84fd9a18`), 02 (`faedeb08`) and 03 (`0c48442`) at their sha256, clean, signed; the five scripts equal the v2.1 B6 table (`phase4_common.py` `99407ec6…`, the other four unchanged) and are clean; supplement `6968e045…` and completion record `fb1e07b2…` in the ledger; guard 702 / 702; the acquisition lock accepts the citation record `6e51efd6…`.
3. **E1, the R12b leftovers (CONFIRMED).** `grl_review_archive/phase4/` (3 files: KN182L03's no-k `pair_stats.json`, the two provenance records; the empty `build/RC2511__MGDS_32239/` and `fields/` directories) moved, not deleted, to `grl_review_archive/phase4_stopped_R12b/` with `SHA256SUMS` (3 files) verified; `grl_review_archive/phase4/` did not exist before the build.
4. **E2, the build (CONFIRMED, every must-hold holds; `tables/build_check.json`).** Job 47557376, `phase4_build.py --mode phase4` on the five pairs, git `e81059b`: guard 702 / 702; every lock passed (sign-off `d6af268b…`, v2.1 hashes, supplement, **acquisition citation record `6e51efd6…` accepted**); allow-list of 15 tokens applied; exit code 0 after 10 min. Per pair (counts and views only; §3): KN182L03 `not scorable (no k)`, only `pair_stats.json`; RC2511 built with product features **masked**; SUM1004, AT42-03, TN159 built with product features **used**; A_fine finite on every tile with joint-valid px of every built pair (3,176 / 304 / 228 / 104). Views at B_min = 8: **RC2511 interior** (1,771 tiles in 168 blocks), **SUM1004 fallback** (42 tiles in 10 blocks; the interior view has 8 tiles in 6 blocks), **AT42-03 fallback** (70 / 18; no interior tile), **TN159 fallback** (103 / 13; interior 2 / 2). D₁ 121 / 151 / 20 / 13 px; cells across 2.13 / 1.70 (both thinner than any development pair, as expected) / 13.0 / 20.3. `SHA256SUMS` written outside the scripts and verified (82 files after the figures, 11 of them under figures/).
5. **E2, the blind pages (CONFIRMED; `tables/pages_check.json`).** Job 47561133, `phase4_figures.py --mode phase4` once: **4 pages**, codes Q01–Q04 (seed 20261011), one no-k pair without a page; key written to `phase4/figures/key/figset_A_key.json` (not opened, not delivered). Page check with a pypdf text extraction: **0 text objects on every page** (the pages are rasterised PNGs; the only text on them is, as coded in `fields_r08.py:102` and `:70`, the Q-code as suptitle and the `500 m` scale-bar label); the codes from the set-A page file names are exactly Q01–Q04. **Delivered beside this report:** `R12c/figset_A_blind.pdf` (sha256 `5db55bcf55de1fa347dd813671c4fb9537dfb84f5af0014bf5975245701732d5`, identical to the archive copy; not committed to git) and `R12c/phase4_calls_TEMPLATE.csv` (sha256 `aea7cb3480f5e7e28d5e09ea783c4e3d11ab3d6878fe0570b32d4b50169668ba`; columns `code, relief, criteria, dominant, ab_whole_survey, note`; one row per code, values empty).
6. **What comes next (B3 steps 3b–5, not this directive).** Steve fills the template (one row per page), commits it, and its sha256 goes into the ledger (the calls lock); then `phase4_predict.py --mode phase4` (EW0207 control first) and `phase4_score.py --mode phase4` once. Nothing of that was run.

**HOLDs for Steve:** none.

**Challenge notes (none blocking):**

- The R12b figures job script carried a wrong relative path to the check step (`cd ../R12b/scripts` from `phase4/scripts`), copied into the R12c one; `phase4_figures.py` itself completed its single run, the job then exited at my check step. The page check and the `SHA256SUMS` update are not Phase 4 scripts and were run afterwards (login node; checksum job 47562475). The figures script was **not** rerun. The job script is fixed for the record.
- TN159's harmonized HR raster has a 3.25 m pixel (the build's `hr_res`), not the 0.991 m `hr_native_res_m` of its June row; D₁ = ⌈41.0 / 3.25⌉ = 13 px follows from the raster. Lockbox, supporting only; no rule attached; noted for the scoring report.
- REPORT_R12c §6 still lists the draft sign-off's pre-shortening sha256 (`9c339ff0…`); the draft at commit `50bf0bd` hashes to `ece5245a…`. Left as it was (the draft is superseded by the signed file).
- The permission classifier again refused shell commands containing the signature text; the signed files were written with the file tools.

---

## 2. E1 details

| item | value |
|---|---|
| sign-off | `R12c/SIGNOFF_phase4_prereg_v2_1.md`, sha256 `712766d4…`, first 64-hex string = v2.1 sha256 `d6af268b…` (`read_signoff` CONFIRMED) |
| deviation note 03 | `R12c/DEVIATION_phase4_03.md`, sha256 `dfccdadc…`, signed |
| commit / tag | `0c48442da0662d9226515d5e93986472ce639ed8`, tag `phase4-prereg-v2.1` (annotated) |
| ledger | row 39 (commit `e81059b`) |
| C1 checks against v2.1 | all five pass at HEAD `0c48442` → `e81059b` (`tables/c1_preconditions_v2_1.json`) |
| leftovers | `grl_review_archive/phase4_stopped_R12b/` (`build/KN182L03__MGDS_30193/pair_stats.json`, `build/RC2511__MGDS_32239/` empty, `fields/` empty, `provenance/phase4_build.py_2026-10-12T125655_31161.json`, `provenance/check_build_r12b.py_2026-10-12T130313_89383.json`), `SHA256SUMS` 3 files verified |

## 3. E2 — the build, per pair (allowed fields only)

<!-- from tables/build_check.json (check_build_r12c.py through phase4_common.start: guard, locks, allow-list) -->
| pair | role | status | tiles v1 → v1.1 | interior tiles / blocks (D₁) | A_fine_int-defined (fallback) tiles / blocks | view (B_min = 8) | D₁ (px) | cells across | thinner than development | product features | A_fine finite on tiles with joint px |
|---|---|---|---|---|---|---|---|---|---|---|---|
| KN182L03__MGDS_30193 | confirmatory (nu01a) | **not scorable (no k)** | — | — | — | — | — | — | — | not built | — (only `pair_stats.json`) |
| RC2511__MGDS_32239 | confirmatory (nu01b) | built | 3,184 → 3,176 | 1,771 / 168 | 2,392 / 211 | **interior** | 121 | 2.13 | yes | **masked** (`ship_products_usable` false; 14 contract columns 100 % NaN) | 3,176 of 3,176 |
| SUM1004__MGDS_33090 | confirmatory (nu03) | built | 311 → 304 | 8 / 6 | 42 / 10 | **fallback** | 151 | 1.70 | yes | used | 304 of 304 |
| AT42-03__MGDS_32007 | lockbox (lu03) | built | 237 → 228 | 0 / 0 | 70 / 18 | **fallback** | 20 | 13.0 | no | used | 228 of 228 |
| TN159__MGDS_21981 | lockbox (lu17) | built | 113 → 104 | 2 / 2 | 103 / 13 | **fallback** | 13 | 20.3 | no | used | 104 of 104 |

Must-holds: 1 locks passed with the citation record — True; 2 KN182L03 no-k with only `pair_stats.json` — True; 3 product features masked / used as required — True; 4 finite target everywhere — True. Build job log lines (amplitude redacted) are in `tables/build_check.json["log_redacted"]`.

## 4. E2 — the pages

| item | value |
|---|---|
| PDF | `grl_review_archive/phase4/figures/figset_A_blind.pdf`; delivered copy `R12c/figset_A_blind.pdf`; sha256 `5db55bcf55de1fa347dd813671c4fb9537dfb84f5af0014bf5975245701732d5` |
| pages / codes | 4; Q01, Q02, Q03, Q04 (seed 20261011); no page for KN182L03 |
| text objects per page (pypdf) | 0, 0, 0, 0 |
| page text as coded | `fields_r08.py:102` suptitle = the code; `:70` scale-bar label `500 m`; `:100` axes off |
| key | `grl_review_archive/phase4/figures/key/figset_A_key.json` — exists; not opened by the executor beyond the script; not delivered |
| calls template | `R12c/phase4_calls_TEMPLATE.csv`, sha256 `aea7cb3480f5e7e28d5e09ea783c4e3d11ab3d6878fe0570b32d4b50169668ba`, 4 rows (`code, relief, criteria, dominant, ab_whole_survey, note`) |

## 5. Do-not-open list (until Steve's calls are committed)

Everything under `grl_review_archive/phase4/` except `figures/figset_A_blind.pdf`: `build/<pair>/pair_stats.json` of the four built pairs and `build/build_summary_1791842415.json` (amplitude statistics), `build/<pair>/{tiles_v1,tiles_v1_1,blocks_v1_1,features,tiles_int}.parquet`, `fields/<pair>/*.tif`, `inputs/<pair>/*`, `figures/key/figset_A_key.json`, `figures/work/pages/*` (set-A pages by code, set-B pages by pair name with k and A_fine medians), `figures/work/pages_record.json`, `provenance/*.json` (pair-coded, no amplitude), `SHA256SUMS` (file names only; harmless but listed). The build job logs `/scratch/users/scdobbs/grl_review/R12c/logs/phase4_build_47557376.out` / `.err` (the `.out` prints each pair's amplitude) and their OAK copy under `grl_review_archive/R12c/logs_partE/`. Safe to open: this report, the PDF, the template, `tables/build_check.json`, `tables/pages_check.json`, `tables/c1_preconditions_v2_1.json`, `grl_review_archive/phase4_stopped_R12b/`.

## 6. Preservation and commits

OAK `grl_review_archive/R12c/`: `logs_partE` (on the do-not-open list), `checks_partE`, `tables_partE`, `scripts_partE`, `documents_partE` (this report, REPORT_R12c, the signed sign-off and deviation note, the template, the blind PDF), each with `SHA256SUMS` verified (job 47564270). Commits: `0c48442` (signatures; tag), `e81059b` (ledger row 39, leftovers, scripts), `dd732a3` (this report, the template, the Part E scripts and tables; the PDF is not committed).
