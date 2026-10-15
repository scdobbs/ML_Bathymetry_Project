# R19 (CNN repo): close the Phase 4 and Phase 5 record

Steve's rulings stand: no amendments, no sign-offs, no rehearsals, no new analysis. Write notes only, from the existing reports and ledgers. Commit by explicit path; never `git add -A`.

1. **R18 preservation.** Confirm that the preservation job finished and that both `SHA256SUMS` (OAK `grl_review_archive/R18/` and `phase5/`) verify. Report the job id and file counts.
2. **Deviation note 04 (Phase 4).** Record that the scorer needed Steve's calls keyed by pair, so a joined copy with values unchanged was ledgered as row 41 and scored. Give both files' sha256 and show the values are identical. Write it in the same format as notes 01 to 03.
3. **Phase 5 deviations note** (`phase5/DEVIATIONS_phase5.md`). One short paragraph each, citing the report and ledger row:
   1. Pre-registration v3.1 governed; by Steve's ruling of 2026-10-14 no later amendment or sign-off was made.
   2. The acquisition freeze moved from v2 (cited in v3.1) to v3 (ACQ-R12: write paths and in-run leak check, no computation change) and then to v4.2 (ACQ-R13). Include the leak-check scope ruling ("records only").
   3. **After step 1 and before any outcome was seen**, the 30 m co-registration bound was replaced by the quadrant rigidity test on Steve's ruling (ACQ-R13). State the rule, the development check (NA080, KM0923, FK151121 rigid), and the eligibility before (2 of 8) and after (5 of 8). Name the superseded run 47792880 and the stopped run 47814252.
   4. The 22 Escanaba tiles straddling drift-group boundaries were excluded on Steve's ruling (R17 H1), none in the primary view, with the exclusion list sha256.
   5. `phase5_common.py`, `phase5_figures.py` and `phase5_score.py` were checked as the tagged file plus a recorded diff instead of a strict hash. The l.134 freeze-sha check was relaxed (R17, R18). List each diff's purpose in one line.
   6. The S1 and S2 rows were declared in the R18 directive before scoring.
   7. MV1003's GBT_ship rows were unavailable because its ship products are unusable.
4. One `CHANGE_LEDGER.md` row; OAK archive with `SHA256SUMS` verified.

**Report:** at most 5 bullets. The two notes' paths and sha256, the preservation check, the commit, and anything in the reports that contradicts this list. If something contradicts it, write what the reports say, not what this prompt says.
