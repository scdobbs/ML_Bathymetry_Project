"""R16 Part E1 — render R16/PREREG_phase5_v3.md from R12c/PREREG_phase4_v2_1.md (sha256 d6af268b… verified) with exactly the changes of
phase_5/PHASE5_design_spec_2026-10-13.md (ruled 2026-10-13) and the directive's notes. Section by section:
  header            replaced (v3 header; the v2.1 rulings paragraph and fallback status carried verbatim)
  B1                replaced (sites §3.1; product rules §3.2 with S1-S3; eligibility §3.3; notes)
  Prior exposure    replaced (§4)
  B2                carried verbatim + one sentence (spec §2: unchanged)
  B3                replaced (order of spec §2 with the Phase 5 steps; calls join inside the score, Phase 4 deviation note 04)
  B4                carried verbatim except tile rule 0 (+ not testable) and the cells-per-window bullet (now eligibility item 3)
  B5                replaced (C1 primary pooled + per-site secondary + manuscript reason §3.4; expectation §3.6; C2 with CIs §3.5; no C3)
  B6                replaced (Phase 5 locks; B6 table of the five Phase 5 scripts + the five imported Phase 4 scripts; acquisition freeze sha256)
  Changes from v1, Revision v2.1   carried verbatim
  Changes in v3, Notes (v3)        appended
Writes R16/PREREG_phase5_v3.md, R16/tables/prereg_v3_render.json and R16/tables/prereg_v3_diff.md (section-by-section diff)."""
import difflib, hashlib, json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__)); DR = os.path.dirname(os.path.dirname(HERE)); V21 = f"{DR}/R12c/PREREG_phase4_v2_1.md"; OUT = f"{DR}/R16/PREREG_phase5_v3.md"
V21_SHA = "d6af268bc8211fe9b4ae17439a47e6fdcbe7e5949d8ab387a7be080b2c748334"; P5 = f"{DR}/phase5/scripts"; ACQ_FREEZE_SHA = "6769583198da63d874472ea5a2ca62cb26c2f9f55003eb6e97838bfebab1d9d1"
SCRIPTS5 = ("phase5_common.py", "phase5_build.py", "phase5_figures.py", "phase5_predict.py", "phase5_score.py")
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
def sections(text):
    """[(heading or '(header)', body)] split at '## ' headings."""
    parts = re.split(r"(?m)^(## .*)$", text); out = [("(header)", parts[0])]
    for i in range(1, len(parts), 2): out.append((parts[i], parts[i + 1]))
    return out

HEADER = """# PREREG_phase5_v3 — Phase 5 pre-registration, version 3 (a second pre-registered blind test)

> **Version 3, rendered 2026-10-13 by directive R16 from `R12c/PREREG_phase4_v2_1.md` (sha256 `d6af268b…8334`, tag `phase4-prereg-v2.1`) with exactly the changes of `phase_5/PHASE5_design_spec_2026-10-13.md` (ruled by Steve on 2026-10-13) and the directive's notes. Phase 4 stays closed and is reported as it is: C1 not confirmed (0 of 2), pooled −0.20 [−0.34, −0.02] (REPORT_R13). "Phase 5" keeps its paths, locks, records and outputs separate from Phase 4. When v3 was rendered no Phase 5 candidate file had been opened by the CNN repo and the acquisition step had not run in phase5 mode; the only candidate information that exists is the ACQ-R08 / ACQ-R08b metadata (Prior exposure). Changes from v2.1 are listed in "Changes in v3".**

**Why (spec §1):** Phase 4 did not confirm C1. Both scorable confirmatory sites had about 2 ship cells per tile, outside the development range, and R14 showed that coarse sampling alone does not explain the failure. Phase 5 tests the same claim, with the same frozen models, on new sites inside the development sampling range.

**Carried from v2.1 (rulings of 2026-10-09, in force):** {rulings}

{fallback}

"""

B1 = """## B1. Sites (spec §3.1–§3.3)

| unit | site | high resolution | ship partner | pair id |
|---|---|---|---|---|
| p5u01 | Galápagos Spreading Center | MGDS:18210, 1 m product only (MGDS:31811 not used) | TN188, Simrad EM300, 2005 | `TN188__MGDS_18210` |
| p5u02 | Escanaba Trough | MGDS:30368 | Escanaba, Simrad EM300, 1998 | `Escanaba__MGDS_30368` |
| p5u03 | Monterey Canyon | MGDS:24756 | Monterey, Simrad EM300, 1998 | `Monterey__MGDS_24756` |
| p5u04 | Mid-Atlantic Ridge 16.5°N | MGDS:22436 | KN161L04, SeaBeam 2112, 2000 | `KN161L04__MGDS_22436` |
| p5u05 | La Jolla Canyon | MGDS:21415 | MV1209, Kongsberg EM122, 2012 | `MV1209__MGDS_21415` |
| p5u06 | Bermuda | MGDS:31675 | KN173L02, SeaBeam 2112, 2003 | `KN173L02__MGDS_31675` |
| p5u07 | Chile margin | MGDS:31860 | MV1003, Simrad EM122, 2010 | `MV1003__MGDS_31860` |
| p5u08 | Dorado Outcrop | MGDS:31954 | AT26-24, Kongsberg EM122, 2014 | `AT26-24__MGDS_31954` |

Each site is one unit and one pair: one high-resolution mosaic, one ship cruise. The sites come from ACQ-R08b; no other site may be added and no site is replaced. Pair ids follow `RECORD_CONTRACT_phase5.md` (`<ship cruise>__<hr id with ':' → '_'>`); units follow the acquisition site list (`inputs/phase5_sites.csv`) order. The Phase 4 sealed pairs are not used, and no Phase 4 sealed pair joins the development set.

**Mechanical product rules (applied by script at step 1, without viewing):**

1. **Resolution group:** the finest group of grids with cell size 0.985 to 4.99 m whose union covers at least 80 % of the largest such group (the ACQ-R08b rule).
2. **Repeat surveys** (overlapping grids of the same seafloor, for example Monterey): for each overlap, keep the grid whose acquisition date is closest to the ship cruise's year; ties go to the larger valid area. Dates come from file names or metadata. Non-overlapping grids are mosaicked. Ruling S2: two grids overlap when their validity masks share ≥ 50 % of the smaller grid's valid area; below that they are mosaicked with the rule order as priority; an undated grid counts as a tie.
3. **Dispersed grids (ruling S1):** after rules 1 and 2 the kept grids go to the unchanged development mosaic logic (`grid_lr.grid_region`, `harmonize_new.hr_mosaic`).
4. **Ship side:** the named cruise's raw swath, gridded by the frozen development chain with contract v2.1 products; harmonized, co-registered and k-swept exactly as the development pairs were. Ruling S3: the current `grid_lr` region rule is used.

**Eligibility (fixed now; decided at step 1 from metadata only).** A site enters the C1 set only if all hold:

1. co-registration passes the development QA thresholds (ACQ-R03: offset ≤ 30 m, overlap ≥ 1 km², ≥ 1 256-px tile with ≥ 50 % joint-valid cells);
2. the frozen sweep recovers a k;
3. the gridded ship spacing gives 3.88 to 51.2 ship cells across a 256-px window (256 × unrounded HR pixel / lr_native), and the HR spacing is 0.985 to 4.99 m;
4. at scoring, the B4 view rules give a scorable view (10 tiles in at least 8 blocks).

A site failing 1 to 3 is **not testable** at step 1, with its reason (supplement column `eligibility_reason`); it has no build, figure page, prediction or score. A site failing 4 is **not scorable** at step 5. Then the artifact rule applies at the blind call. No site is replaced or added.

**Notes (v3):**

- **The 3.88 lower bound and FK151121.** The bound is FK151121's development value, 256 × 1.000274 / 65.99 = 3.8804 cells. The frozen Phase 5 chain, with today's region rule, gives FK151121 lr_native 66.09 m and 3.8746 cells, which would be "not testable". The bound is kept as written (ruling ACQ-R09 H-1, Steve, 2026-10-13). Nothing changes unless a site lands within about 0.3 % of 3.88; the lowest ACQ-R08b estimate is 4.32 cells (Mid-Atlantic Ridge).
- **The gridding region rule.** Step 1 grids the ship swath with the current `grid_lr` rule (`grid_region`: the ship grid on the HR patch cluster; ruling S3). FK151121's development grid predates that rule (whole footprint, commit `ce0ebb3`). ACQ-R09 reproduced NA080 and KM0923 bit-identically under the current rule, and FK151121 bit-identically only under its recorded region in a separate harness.
- **`vertical_offset` fallback.** The contract-v2.1 `vertical_offset` flag uses its 0.5 m fallback threshold for every Phase 5 site, because no CNN s_lr exists for them. This is the frozen builder's behaviour, as it was for the Phase 4 sealed pairs.
- **No person views candidate data at step 1.** Co-registration passes or fails by the automated development thresholds only; no QC figure is produced, and the step-1 report carries only status, eligibility fields, spacings, cells across, k, product status and usability, file counts and record sha256 values (ACQ-R09 §2.10). The only viewing in Phase 5 is Steve's blind call on the Q-coded pages of step 2 (B3 step 3).

"""

EXPOSURE = """## Prior exposure

- Steve has seen none of the eight sites' bathymetry. That includes the June QC figure for Bermuda and the La Jolla Zephyr survey (stated 2026-10-13).
- Everyone has seen the ACQ-R08 and ACQ-R08b metadata (listed in REPORT_ACQ-R08b §5). That includes one median depth per site, rounded to 100 m.
- The executor that ran ACQ-R08b saw only those numbers.

The blind artifact call of B3 step 3 is made against the R08 §3.1 criteria with this exposure stated.

"""

B2_ADD = "B2 applies unchanged at Phase 5 (spec §2): no retraining, refitting or tuning; the development set is unchanged.\n\n"

B3 = """## B3. Order of operations at Phase 5 (each step a separate directive or job; nothing out of order)

1. **Acquisition repo (ACQ-R10):** the frozen `src/phase5/sealed_step1_p5.py --mode phase5` (tag `phase5-acq-freeze`; code commit `4d93c8a`; freeze record `reports_post_grl_review/PHASE5/freeze_record_phase5.json` sha256 `{acq}`; script sha256 `9f143e0c3f000163d9da48001cbd7578805c8c1042141f9994a972df73b1b582`), rehearsed in ACQ-R09.
   - It runs locks 1 to 4: the sign-off holding this document's sha256, the freeze record, exactly the eight listed sites, and no second completed run.
   - It fetches the named cruise's raw swath and grids it with the frozen development chain. It selects and mosaics the HR grids by the B1 product rules, then harmonizes and co-registers. It runs the frozen C2c sweep (k = the largest k with verdict `recoverable_signal`) and builds the contract-v2.1 products.
   - `ship_products_usable` is true if and only if the products were built and carry no contract-v2.1 flag (frozen Phase 4 rule C1).
   - It decides eligibility items 1 to 3 from metadata only.
   - It writes `manifest/pairs_phase5_sealed.parquet` once: the 64 columns of the Phase 4 supplement, then `coreg_pass, k_recoverable, cells_across, hr_res_in_range, eligible, eligibility_reason`. It also writes the run record `reports_post_grl_review/PHASE5/step1_run_phase5_<UTC>.json` with the keys of `RECORD_CONTRACT_phase5.md`. No record key holds a depth statistic, amplitude or residual.

   The supplement's sha256 is recorded in `CHANGE_LEDGER.md`.
2. **CNN repo, build and blind pages:**
   - `phase5_build.py --mode phase5` verifies the acquisition run record and builds each eligible site. It writes each site's amplitude statistic (B4). `max_recoverable_k` comes from the supplement; the acquisition repo owns the rule. A site's product features are used only if its supplement row has `ship_products_usable = true`; otherwise they are masked.
   - A not-testable site is not opened; only its status is written.
   - `phase5_figures.py --mode phase5` then produces figure set A for the eligible sites. Pages are shuffled with seed 20261011 and coded Q01 onward; the key is in a separate file.
   - **No prediction or score yet.**
3. **Blind artifact judgement:** Steve judges each page against the R08 §3.1 criteria and fills one row per page:

   | field | values |
   |---|---|
   | `code` | Q01 onward, one per page |
   | `relief` | yes / weak / no |
   | `criteria` | any of a, b, c, or none |
   | `dominant` | relief / artifact / mixed |
   | `ab_whole_survey` | yes / no / na: do stripes (a) or ribbing (b) extend across the whole survey? |
   | `note` | free text |

   The calls file stays keyed by Q-code. It is committed and its sha256 recorded in `CHANGE_LEDGER.md` **before** any prediction is computed. The Q-code to pair join is part of this procedure: `phase5_score.py` joins the committed file to the pairs through the figure key, asserting every value unchanged, and writes no separate joined file (Phase 4 deviation note 04).
4. **Predict:** `phase5_predict.py --mode phase5`, on CPU. The script first reproduces the EW0207 reference predictions (s_lr identical; GBT_ship within 1e-6 m; P-V1 within 1e-4 in log₁₀) and refuses to continue otherwise. Not-testable sites are not predicted.
5. **Score once:** `phase5_score.py --mode phase5`. Report all results, whatever they are. No retuning, re-selection or re-scoring with changed rules afterwards.

"""

B4_RULE0_OLD = "  0. A pair with no recoverable k has no fine band. It is **not scorable (no k)**, has no figure page, and is outside C1."
B4_RULE0_NEW = "  0. A site that failed eligibility at step 1 (B1 items 1–3) is **not testable**, with its reason; it has no build, figure page, prediction or score, and is outside C1. A pair with no recoverable k has no fine band. It is **not scorable (no k)**, has no figure page, and is outside C1 (at Phase 5 this cannot occur for an eligible site: k is eligibility item 2)."
B4_CELLS_OLD_RE = r"- \*\*Cells per window:\*\* `256 × hr_res / lr_native` is reported for each sealed pair beside the development minimum of 3\.88\..*?all below the development minimum\."
B4_CELLS_NEW = "- **Cells per window:** `256 × hr_res / lr_native` is eligibility item 3 at step 1 (3.88 to 51.2; B1). It is reported for each site, and the score output keeps v2.1's flag \"window thinner than any development pair\" (below 3.88), with no rule attached. From the ACQ-R08b metadata the eight sites are expected at about 4.3 to 24.5 cells."

B5 = """## B5. Criteria (fixed now)

- **C1 set:** the eligible sites (B1 items 1–3) that are scorable (item 4), not artifact-dominated and not low-amplitude artifact-affected. Its size is stated.
- **C1, primary: pooled ranking transfers (spec §3.4).** The mean within-site s_lr ρ over the C1-set sites, with a 95 % CI from a block bootstrap stratified by unit (non-empty blocks resampled within each site, 2,000 resamples, one `numpy.random.default_rng(20261013)`, sites in ascending pair-id order; a resample with any undefined site ρ is dropped and counted, > 5 % → not estimable).
  - **Confirmed** if the CI lower bound is above zero.
  - **Not confirmed** otherwise.
  - **Not testable** if fewer than 3 sites are in the C1 set.

  The pooled row is also reported with the low-amplitude artifact-affected sites included. That row does not change the label.
- **C1-per-site, secondary.** Each site's s_lr CI as in v2.1 (B4), with the count of sites whose lower bound is above zero and the v2.1 all-sites label, reported beside the primary:
  - **confirmed** if the lower bound is above zero on every site of the C1 set;
  - **partially confirmed** if on some;
  - **not confirmed** if on none;
  - **not testable** if the set is empty.

  It is also stated with the low-amplitude artifact-affected sites included.
- **Why the change, for the manuscript (Steve's condition):** with n sites and a per-site pass rate of 9 in 14 at development, "every site passes" occurs with probability about 0.64^n even if the method works as in development (about 3 % at n = 8). So the v2.1 per-site label cannot confirm transfer with more than two or three sites. The pooled mean estimates average transfer directly and has adequate power. The change is made after Phase 4 was known. Under it, Phase 4 would still read "not confirmed" (pooled −0.20 [−0.34, −0.02]), so the change does not alter that result. Both readings are reported for Phase 5.
- **Expectation (spec §3.6):** at development, pooled s_lr is +0.392 [+0.317, +0.430] over 14 pairs, and +0.35 at 1.7 cells per tile (R14). Phase 4 was −0.20 on two sites. Phase 5 tests which of these the new sites resemble. For the secondary reading: development s_lr met the per-site condition on 9 of 14 pairs at D₁ (REPORT_R10 §6.3), pair ρ −0.09 to +0.82, median +0.41.
- **C2, descriptive: model complexity (spec §3.5).**
  - GBT_ship and P-V1 against s_lr, per site and pooled over the C1 set. Each candidate's ρ and each difference from s_lr (GBT_ship − s_lr, P-V1 − s_lr) gets a block-bootstrap 95 % CI.
  - The CIs use the same resamples as s_lr: per site, the block draws of `default_rng(20261009)`; pooled, the draws of `default_rng(20261013)`, sites in ascending pair-id order. The same undefined-resample rule applies.
  - `phase5_score.py` (pinned in B6) computes them. The frozen Phase 4 score gives no CI for GBT_ship or P-V1.
  - P-V1 − GBT_ship is reported as a point value.
  - One sentence per comparison states its direction across sites. No superiority claim is made.
- C3 (the lockbox role) has no counterpart at Phase 5.

"""

B6 = """## B6. Provenance requirements at Phase 5

Every Phase 5 script runs in `phase5` mode only if `R16/SIGNOFF_phase5_prereg_v3.md` exists and holds this document's sha256. Before running, it verifies the following against this document:

- that sha256;
- the candidate artifacts (B2);
- the frozen code (R05 §6; `targets_v1_int.py`; `bar_v1_1.py`; the R07 model code hash; `fields_r08.py`);
- the five Phase 5 scripts;
- the five Phase 4 scripts, which the Phase 5 scripts import unchanged for every frozen function (targets, tiles, interior masks, features, model inputs, the three models, the EW0207 control, the artifact rule, the single-run lock and the bootstrap seeds).

It also requires the acquisition freeze record sha256 `{acq}` to be stated here.

| script | sha256 |
|---|---|
{rows5}

Imported unchanged (v2.1 B6, tag `phase4-prereg-v2.1`):

| script | sha256 |
|---|---|
{rows4}

**Supplement lock.** In `phase5` mode the eight site rows are read only from `manifest/pairs_phase5_sealed.parquet`, beside manifest v2.1. The file must meet all of these:

- it holds exactly the eight pre-registered pair ids;
- its first 64 columns are the Phase 4 supplement's names, order and arrow types (`manifest/pairs_phase4_sealed.parquet`, sha256 `6968e045…9206`);
- they are followed by exactly the six eligibility columns;
- `eligible` agrees with B1 items 1–3 on every row;
- its sha256 is recorded in `CHANGE_LEDGER.md`.

Development rows come from the frozen v2.1 file. The eligible rows are the admitted sites; a row with `eligible = False` is "not testable (<eligibility_reason>)".

**Acquisition run-record lock.** After the supplement lock, every script verifies the acquisition run record. The acquisition repo's `reports_post_grl_review/PHASE5/step1_run_phase5_*.json` must:

- have status complete in `phase5` mode;
- cite the freeze record sha256 `{acq}` in `locks.lock2_freeze_record`;
- cite the supplement's sha256.

The `freeze_record_phase5.json` on disk must hash to that value and list `src/phase5/sealed_step1_p5.py` at `9f143e0c3f000163d9da48001cbd7578805c8c1042141f9994a972df73b1b582`. Otherwise the script refuses.

**Guard.**
- The R09 guard of Phase 4 is installed unchanged and self-tests 702 of 702 probes.
- The Phase 5 extension seals the roots `sealed_candidates_v3` and `auv_ship_colocated_bathy/phase5`, every site's pair id, ship cruise and HR id, and MGDS:31811. It self-tests on the real Phase 5 names before any lock.
- After every lock, a path is admitted only if it contains an eligible site's full pair id. Every other path holding a Phase 5 token stays refused, and the Phase 4 tokens are never admitted.

**Calls and single run.** `phase5_predict.py` and `phase5_score.py` require the committed code-keyed calls file, with its sha256 in `CHANGE_LEDGER.md`. `phase5_predict.py` first reproduces the EW0207 reference predictions from the R10 inputs on OAK and writes the comparison to its provenance record. Outputs go to `grl_review_archive/phase5/` with `SHA256SUMS`. `phase5_score.py` refuses to run a second time.

"""

CHANGES_V3 = """## Changes in v3 (Phase 5; directive R16; spec of 2026-10-13)

- **Name and separation:** Phase 5 has its own scripts (`phase5/scripts/`), locks, records and outputs (`grl_review_archive/phase5/`). Phase 4, its files, records, tags and outputs are unchanged.
- **Sites and product rules** (B1; spec §3.1, §3.2; rulings S1–S3), replacing the five Phase 4 sealed pairs, the lockbox role and its history.
- **Eligibility** (B1; spec §3.3): items 1–3 at step 1 from metadata (`eligible` / `eligibility_reason` in the supplement), item 4 at scoring. A failing site is "not testable" or "not scorable" and is never replaced. Cells per window becomes eligibility item 3 (B4).
- **C1:** the primary criterion is the pooled row; the v2.1 per-site label is secondary (B5; spec §3.4, with the manuscript reason).
- **C2** gets block-bootstrap CIs for GBT_ship, P-V1 and their differences from s_lr, per site and pooled, from the same resamples (B5; spec §3.5).
- **Expectation** (B5; spec §3.6) and **prior exposure** (spec §4) restated for Phase 5.
- **Order** (B3; spec §2): step 2 is build and blind pages; the calls stay keyed by Q-code and the join to pairs is done inside `phase5_score.py` (Phase 4 deviation note 04).
- **Locks** (B6): the sign-off `R16/SIGNOFF_phase5_prereg_v3.md`; the five Phase 5 scripts and the five imported Phase 4 scripts; the supplement `manifest/pairs_phase5_sealed.parquet` with 64 + 6 columns; the run record of `RECORD_CONTRACT_phase5.md` in `reports_post_grl_review/PHASE5/`; the acquisition freeze record sha256 `{acq}` (tag `phase5-acq-freeze`); the Phase 5 guard extension and allow-list.
- **Unchanged:** the models and artifacts (B2), targets, tiles, interior masks, features, model inputs, views, B_min = 8, the 10-tile minimum, the bootstrap (2,000 resamples, seeds 20261009 and 20261013), the amplitude statistic, the artifact rule (0.04 m floor), the EW0207 control, the figure seed 20261011 and codes, the calls fields, CPU inference, the single-run lock.
- **Rehearsal (R16 Part C):** before signature, the four Phase 5 scripts ran end to end in rehearsal mode on the ACQ-R09 rehearsal outputs copied into a layout named for the real Phase 5 roots, through the same `start`, data locks, allow-list and guard as phase5 mode; the R10 reproduction and the refusal-test suite passed (REPORT_R16).
"""

if __name__ == "__main__":
    assert sha(V21) == V21_SHA, sha(V21); text = open(V21).read(); S = sections(text); names = [h for h, _ in S]
    assert names == ["(header)", "## B1. Sealed units", "## Prior exposure", "## B2. Frozen candidates (artifact paths and sha256)", "## B3. Order of operations at Phase 4 (each step a separate directive or job; nothing out of order)",
                     "## B4. Scoring rules", "## B5. Criteria (fixed now)", "## B6. Provenance requirements at Phase 4", "## Changes from v1", "## Revision v2.1 (2026-10-12; directive R12c; deviation note 03)"], names
    sec = dict(S); head = sec["(header)"].splitlines()
    rulings = [l for l in head if l.startswith("**Written:**")]; fallback = [l for l in head if l.startswith("**Fallback status")]; assert len(rulings) == 1 and len(fallback) == 1
    b4 = sec["## B4. Scoring rules"]; assert b4.count(B4_RULE0_OLD) == 1; b4n = b4.replace(B4_RULE0_OLD, B4_RULE0_NEW); assert len(re.findall(B4_CELLS_OLD_RE, b4n, flags=re.S)) == 1; b4n = re.sub(B4_CELLS_OLD_RE, B4_CELLS_NEW, b4n, flags=re.S)
    rows4 = re.findall(r"(?m)^\| `phase4_\w+\.py` \| `[0-9a-f]{64}` \|$", sec["## B6. Provenance requirements at Phase 4"]); assert len(rows4) == 5
    rows5 = [f"| `{s}` | `{sha(os.path.join(P5, s))}` |" for s in SCRIPTS5]
    out = (HEADER.format(rulings=rulings[0], fallback=fallback[0]) + B1 + EXPOSURE + "## B2. Frozen candidates (artifact paths and sha256)" + sec["## B2. Frozen candidates (artifact paths and sha256)"].rstrip("\n") + "\n\n" + B2_ADD
           + B3.format(acq=ACQ_FREEZE_SHA) + "## B4. Scoring rules" + b4n + B5 + B6.format(acq=ACQ_FREEZE_SHA, rows5="\n".join(rows5), rows4="\n".join(rows4))
           + "## Changes from v1" + sec["## Changes from v1"] + "## Revision v2.1 (2026-10-12; directive R12c; deviation note 03)" + sec["## Revision v2.1 (2026-10-12; directive R12c; deviation note 03)"].rstrip("\n") + "\n\n" + CHANGES_V3.format(acq=ACQ_FREEZE_SHA))
    open(OUT, "w").write(out); S3 = sections(out)
    # section-by-section diff
    old = {h: b for h, b in S}; new = {h: b for h, b in S3}; lines = ["# PREREG v2.1 -> v3: section-by-section diff (R16 Part E1)", "", f"v2.1 `R12c/PREREG_phase4_v2_1.md` sha256 `{V21_SHA}`; v3 `R16/PREREG_phase5_v3.md` sha256 `{sha(OUT)}`.", "", "| v2.1 section | v3 section | change | lines −/+ |", "|---|---|---|---|"]
    pairing = [("(header)", "(header)", "replaced (v3 header; v2.1 rulings paragraph and fallback status carried verbatim)"), ("## B1. Sealed units", "## B1. Sites (spec §3.1–§3.3)", "replaced (spec §3.1–§3.3, rulings S1–S3, four notes)"),
               ("## Prior exposure", "## Prior exposure", "replaced (spec §4)"), ("## B2. Frozen candidates (artifact paths and sha256)", "## B2. Frozen candidates (artifact paths and sha256)", "verbatim + 1 sentence (spec §2)"),
               ("## B3. Order of operations at Phase 4 (each step a separate directive or job; nothing out of order)", "## B3. Order of operations at Phase 5 (each step a separate directive or job; nothing out of order)", "replaced (spec §2 order; deviation note 04 join)"),
               ("## B4. Scoring rules", "## B4. Scoring rules", "verbatim except rule 0 (+ not testable) and the cells-per-window bullet"), ("## B5. Criteria (fixed now)", "## B5. Criteria (fixed now)", "replaced (spec §3.4–§3.6; C3 dropped)"),
               ("## B6. Provenance requirements at Phase 4", "## B6. Provenance requirements at Phase 5", "replaced (Phase 5 locks; B6 tables; freeze sha256)"), ("## Changes from v1", "## Changes from v1", "verbatim"),
               ("## Revision v2.1 (2026-10-12; directive R12c; deviation note 03)", "## Revision v2.1 (2026-10-12; directive R12c; deviation note 03)", "verbatim"), (None, "## Changes in v3 (Phase 5; directive R16; spec of 2026-10-13)", "added")]
    full = []
    for a, b, what in pairing:
        oa = old.get(a, "") if a else ""; nb = new[b]; d = list(difflib.unified_diff(oa.splitlines(), nb.splitlines(), lineterm="", n=0)); minus = sum(1 for l in d if l.startswith("-") and not l.startswith("---")); plus = sum(1 for l in d if l.startswith("+") and not l.startswith("+++"))
        lines.append(f"| {a or '—'} | {b} | {what} | −{minus} / +{plus} |"); full += [f"### {b}", "```diff"] + d + ["```", ""]
    assert set(new) == {b for _, b, _ in pairing}, sorted(set(new) ^ {b for _, b, _ in pairing})
    open(f"{DR}/R16/tables/prereg_v3_diff.md", "w").write("\n".join(lines + ["", "## Full diff per section", ""] + full) + "\n")
    rec = dict(v2_1=dict(path=V21, sha256=V21_SHA), v3=dict(path=OUT, sha256=sha(OUT), bytes=os.path.getsize(OUT)), phase5_scripts={s: sha(os.path.join(P5, s)) for s in SCRIPTS5}, acq_freeze_sha256=ACQ_FREEZE_SHA, sections=[h for h, _ in S3])
    json.dump(rec, open(f"{DR}/R16/tables/prereg_v3_render.json", "w"), indent=1); print("\n".join(lines)); print("v3 sha256", rec["v3"]["sha256"])
