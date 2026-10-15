# DIRECTIVE R18 (CNN repo): commit Steve's calls, predict, score once

**Date:** 2026-10-14
**Report to:** `grl_reviews-and-prompt-reports/directives_and_reports/R18/REPORT_R18_<date>.md`
**Report length:** at most 8 bullets, plus the results tables.

## 0. Purpose and rulings

R17 built the five eligible Phase 5 sites and made the blind pages. Steve has made his artifact calls. Commit them, then predict and score once under pre-registration v3.1.

**Steve's rulings (2026-10-14), still in force:**
1. Pre-registration v3.1 governs. No amendments, sign-offs or dress rehearsals.
2. Local co-registration (ACQ-R13) replaces the 30 m bound. The 22 Escanaba boundary tiles in `R17/exclusions_phase5_r17.json` (sha256 `e0a620a03b3bba951ea63fa8aa06da8897fbbcdcb586070fb0e5e25b395bd095`) are excluded from every row.
3. **Fix plumbing yourself** (a path, a version constant, a lock naming the wrong file): fix it, commit by explicit path, log it in one line, continue. No rehearsal.
4. **Stop only if** a fix would change a computation, threshold, rule, the site set or Steve's calls.

**Steve's calls are never edited.** If the score script cannot parse them as written, stop and report the exact error.

## 1. Commit the calls (before any prediction)

1. Write `R18/phase5_calls.csv` with exactly this content (UTF-8, LF line endings, final newline):

```
code,relief,criteria,dominant,ab_whole_survey,note
Q01,yes,a,relief,yes,"Survey-line stripes run NW across the survey, with occasional data gaps along survey lines. Relief dominates."
Q02,yes,a,relief,no,"Stripes in strips along no-data areas and smaller stripes in the eastern part of the page. The highest-amplitude changes are relief."
Q03,yes,a,relief,yes,"Subtle track lines running NE across the survey. Subtle square, stitching-like patterns, which look like figure tile stitching rather than gridding ribbing. Relief still dominates."
Q04,yes,a,relief,yes,"Small lines that appear to be survey-line artifacts. Relief dominates."
Q05,yes,none,relief,na,"Clean survey."
```

   Its sha256 must be `9e7bcccbcab286fd3a6b6eb91f4b280647b60aa6252de92409ba60a537c54ced` (676 bytes). If not, stop and report the difference.
2. Confirm the calls parser in `phase5_score.py` accepts every value (`none`, `na`) without opening any sealed file. If it does not, stop.
3. If the calls lock expects another path, copy the file there byte for byte and confirm the sha256.
4. Commit the calls file alone, by explicit path: `R18: Steve's blind artifact calls, committed before any prediction`. Then one `CHANGE_LEDGER.md` row: path, sha256, commit, the blind PDF it was made from (`figset_A_blind.pdf`, `7f811f3337d82e1944e2f61b61b22a1bb60064a8d31bbf124b341dbe6470b71b`), status "committed before prediction".

## 2. Predict

`phase5_predict.py --mode phase5` on CPU, EW0207 control first. A failed control stops the run.

## 3. Score once

`phase5_score.py --mode phase5`, once, with the exclusion list applied. Report every result, whatever it is. No rescoring with changed rules.

**Declared now, before any result is seen** (descriptive only; never the primary outcome):
- **S1, original eligibility:** the pooled row on the sites that were eligible under the 30 m bound (Bermuda, Dorado), with the label "not testable under the original rule" if fewer than 3 survive.
- **S2, rigid co-registration only:** the pooled row without the drift-case sites (Escanaba, Chile margin).

## 4. Report (Steve is unblinded once the calls are committed)

1. The calls commit, sha256 and ledger row; the EW0207 control result.
2. **The site-to-Q key.**
3. **The artifact rule as applied:** for each site, `dominant`, `ab_whole_survey`, its amplitude statistic, and whether it is in the C1 set.
4. **Primary (v3.1):** the pooled s_lr row, ρ with 95 % CI, the number of sites, and the verdict (confirmed / not confirmed / not testable).
5. **Per site:** view used, tiles and blocks, s_lr ρ and CI, the v2.1 label, and the secondary rows (all-tile ρ, A_fine_int ρ).
6. **C2:** GBT_ship and P-V1 per site and pooled, with CIs and differences from s_lr. MV1003's GBT_ship and every row derived from it are "unavailable (ship products unusable)".
7. S1 and S2.
8. Every plumbing fix, one line each. Commit by explicit path; one `CHANGE_LEDGER.md` row; OAK archive with `SHA256SUMS` verified.

## 5. Out of scope

- Any change to the calls, models, thresholds, rules or the site set.
- Any second scoring run.
- Any write to the acquisition repo.
