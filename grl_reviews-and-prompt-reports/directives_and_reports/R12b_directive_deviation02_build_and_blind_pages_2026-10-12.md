# DIRECTIVE R12b (CNN repo, CPU only): record deviation note 02, then rerun the Phase 4 build and the blind pages

**Date:** 2026-10-12
**Repo:** CNN repo, `grl_reviews-and-prompt-reports/directives_and_reports/`
**Report to:** `R12b/REPORT_R12b_<date>.md`
**Certainty labels:** CONFIRMED / CONTRADICTED / COULD NOT CHECK.
**Report length:** at most 6 bullets, plus at most 3 HOLDs.

## 0. Purpose and authority

REPORT_R12 stopped at HOLD H1. The acquisition run-record lock refused the ACQ-R07b completion record because it
carries the freeze-record citation under a different key. Steve ruled option (a). ACQ-R07c then wrote
`reports_post_grl_review/ACQ-R06b/step1_run_phase4_citation.json` (sha256
`6e51efd61f36f92066ff91de2749cdd7ce85192c6f8d20611c1a3de45ffa431e`) in the acquisition repo. A dry check in a
guarded `python -B` process showed that `load_acq_run_record` accepts it, and no CNN file changed.

This directive:

1. commits deviation note 02 and the companion ledger row (Part A);
2. reruns only R12 Part C, the Phase 4 build (Part B);
3. reruns only R12 Part D, the blind pages and the calls template (Part C).

The R12 Part A gate is **not** rerun. Its result stands.

**Authority to rerun.** The standing rule is "no rerun after a stop without Steve's ruling". Steve's ruling on R12
H1 (option a) and this directive are that ruling, for exactly one rerun of the build and one run of the figures
script. Any further stop is a HOLD and is not rerun.

**Challenge clause.** If any instruction here is wrong or underspecified against the code or the files on disk,
stop at that point, give the evidence, and report it as a HOLD. Do not work around it.

## 1. Preconditions (read only; any failure stops the directive)

1. **Commits since R12.** R12's last reported commit is `da20703`; CNN HEAD is now `5359d4c4`. List every commit
   from `da20703` to HEAD with its message and changed paths. Confirm that none of them touches:
   - the five Phase 4 scripts;
   - `R11/PREREG_phase4_v2.md`, `R11/SIGNOFF_phase4_prereg_v2.md` or `R11/DEVIATION_phase4_01.md`;
   - ledger rows 1 to 36 of `grl_reviews-and-prompt-reports/CHANGE_LEDGER.md`;
   - any file named in the B6 table or the B2 artifact list.

   If any commit touches one of these, stop and report a HOLD.
2. **R12 C1 preconditions.** Rerun `R12/scripts/preconditions_r12.py`, or an equivalent read-only check, and report
   all five rows:
   - sign-off;
   - pre-registration v2 at its tag;
   - deviation note 01;
   - B6 script hashes and git cleanliness;
   - supplement and completion record hashes in the ledger, and the guard self-test (702 / 702).
3. **The citation record.** Hash it on disk:
   `<acq repo>/reports_post_grl_review/ACQ-R06b/step1_run_phase4_citation.json` =
   `6e51efd61f36f92066ff91de2749cdd7ce85192c6f8d20611c1a3de45ffa431e`. Confirm that it is committed in the
   acquisition repo, and report that commit. Read it as data only. Write nothing to the acquisition repo.
4. **The lock.** In a fresh `python -B` process with the R09 guard installed and self-tested first, call
   `phase4_common.load_acq_run_record("6968e04554ddeef3c335587d602cb5fab2185db32b46cbd152475a919eb59206")` and
   nothing else. It must return the citation record. This was the R12 challenge row that failed.
5. **Single-run locks.** Read how the single-run locks in `phase4_build.py` and `phase4_figures.py` decide that a
   run has already happened, and report file and line. Confirm:
   - that the refused job 47445500 left nothing that these locks count as a prior run;
   - that `grl_review_archive/phase4/` does not exist.

   If either lock would refuse this run, stop and report a HOLD. Do not clear any marker.

## 2. Part A: deviation note 02 and ledger row 37

### 2.1 Steve's written consent to sign on his behalf

> I, Steve, have read the draft of DEVIATION_phase4_02 in REPORT_ACQ-R07c section 2 and approve it with the edits
> below. I instruct the executor of directive R12b to write, sign and commit it on my behalf, with the signature
> line worded as given in section 2.2. Steve, 2026-10-12.

### 2.2 Write the note

Write `R11/DEVIATION_phase4_02.md`, beside deviation note 01, with exactly this content:

```
# DEVIATION_phase4_02: the acquisition run-record lock reads a citation record
1. REPORT_R12 H1: the pinned lock phase4_common.load_acq_run_record (B6 ef70653f...) requires the freeze-record citation at locks.lock2_freeze_record; the ACQ-R07b completion record (fb1e07b2...) carries it under locks.lock2_frozen_code and at top level, so the lock refused it.
2. Ruling (a): the acquisition repo wrote one new record, step1_run_phase4_citation.json (sha256 6e51efd61f36f92066ff91de2749cdd7ce85192c6f8d20611c1a3de45ffa431e), by a new script (src/phase4/write_citation_record.py, 9545896b...). It is a verbatim copy of the completion record plus the citation under the expected key and four labelling keys, and it cites the completion record and the stopped ACQ-R07 record (ACQ-R07c).
3. The lock therefore now accepts a citation record written by a new script, not a record written by the frozen step-1 or completion code, as pre-registration section A2.2 assumed.
4. No rule, criterion, candidate, threshold, value, pinned script or pinned hash changed; nothing was recomputed; no sealed file was opened; phase4_common.py, the pre-registration and every B6 hash stand.
Signed for Steve by the executor of directive R12b, on his written instruction in R12b section 2.1. Date: 2026-10-12
```

### 2.3 Commit and record

1. Commit the note alone, by explicit path. Report its sha256 and commit.
2. Append ledger row 37 to `grl_reviews-and-prompt-reports/CHANGE_LEDGER.md`. It must carry:
   1. the companion line from REPORT_ACQ-R07c, verbatim:
      `Acquisition citation record for the §A2.2 lock (ACQ-R07c, ruling R12-H1 (a)): reports_post_grl_review/ACQ-R06b/step1_run_phase4_citation.json, sha256 6e51efd61f36f92066ff91de2749cdd7ce85192c6f8d20611c1a3de45ffa431e (verbatim copy of the completion record fb1e07b24681e8b7b15dec3b232ea78266c6c7cd82ae64518e82b76ae99dcc07 plus the citation under locks.lock2_freeze_record)`;
   2. the deviation note's path, its full sha256 and its commit.
3. Commit the ledger and this directive by explicit path. Never use `git add -A`.
4. Do not touch the stray untracked `directives_and_reports/CHANGE_LEDGER.md`.

## 3. Part B: the Phase 4 build (rerun of R12 Part C2 only)

1. Submit one CPU job with the same script and arguments as job 47445500 (`R12/scripts/phase4_build.sbatch`, or a
   copy in `R12b/scripts/` that differs only in log paths):

   ```
   python phase4_build.py --mode phase4 --pairs "KN182L03__MGDS_30193 RC2511__MGDS_32239 SUM1004__MGDS_33090 AT42-03__MGDS_32007 TN159__MGDS_21981"
   ```

2. **Must hold.**
   1. Every lock passes, including `load_acq_run_record`, which returns the citation record.
   2. KN182L03 ends `not scorable (no k)` and writes only its `pair_stats.json`.
   3. RC2511 is built with its product features masked; SUM1004, AT42-03 and TN159 are built with product features
      used.
   4. Every built pair has a finite target on every tile with joint-valid pixels.
3. **On any stop:** report it as a HOLD. Do not rerun and do not change anything.
4. **After success:** write `SHA256SUMS` in `grl_review_archive/phase4/`, outside the scripts, and verify it with
   `sha256sum -c`.

## 4. Part C: the blind pages (rerun of R12 Part D only)

1. Run `phase4_figures.py --mode phase4` once. It produces:
   - `figures/figset_A_blind.pdf`;
   - the key under `figures/key/`;
   - the work pages under `figures/work/`;
   - `phase4_calls_TEMPLATE.csv`, one row per page code.
2. **Must hold.**
   1. The page-text check from R10 and R12 section 6. Each set-A page carries only its Q-code as the suptitle and the
      `500 m` scale-bar label. The PDF is assembled from rasterised pages and holds no text objects. Check this with
      a text extraction on the PDF; you may open it for that check.
   2. Count the pages and page codes, and confirm the template has one row per code.
3. Update `SHA256SUMS` in `grl_review_archive/phase4/` and verify it.

## 5. What the report may and may not contain

Steve stays blind until his calls are committed. The report gives counts, flags, views, hashes and paths only.

1. **Allowed per pair:** status; tile count; interior tile count; A_fine_int fallback tile count; block count; the
   view (interior or fallback, B_min = 8); D1; cells across; and the product-feature decision.
2. **Allowed for the pages:** page count, the list of Q-codes, the PDF's path and sha256, and the template's path
   and sha256.
3. **Forbidden:**
   1. any amplitude, A_fine, A_fine_int, F_hr, s_lr, feature or prediction value, including summaries such as
      medians or ranges;
   2. any image, or any description of what a page shows;
   3. any mapping between Q-codes and pairs, and any content of the key;
   4. any excerpt of the build `.out` or `.err` log that is not redacted. `phase4_build.py:77` prints amplitudes.
4. Deliver to Steve only the report, `figures/figset_A_blind.pdf` and `phase4_calls_TEMPLATE.csv`.
5. Restate the do-not-open list from R12 section 7, with the actual paths created by this run, including the build
   job's log paths.

## 6. Preservation

Copy the R12b scripts, tables, logs and report to OAK `grl_review_archive/R12b/` with `SHA256SUMS` verified at the
destination. Before writing the report, check that the logs copied to OAK are on the do-not-open list.

## 7. Out of scope

- Rerunning the R12 Part A gate.
- Running `phase4_predict.py` or `phase4_score.py`, or any prediction or score of a sealed pair.
- Editing any hash-pinned file, the pre-registration, the sign-off or deviation note 01.
- Any write to the acquisition repo.
- Opening the key, the work pages or any file on the do-not-open list, except where a script itself reads it.
- Clearing or editing any lock marker.
- Making blind calls, or assigning page codes to any judgement.

## 8. Not pursued

- Closing the ResourceWarning file handles in frozen code.
- Interpreting the harness predictions from R12 Part A.
- Why the ACQ-R07b completion script wrote the citation under `lock2_frozen_code`.
