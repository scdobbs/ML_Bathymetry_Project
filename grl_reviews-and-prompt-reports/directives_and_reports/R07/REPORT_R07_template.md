# REPORT R07 — Phase 3 (CNN repo): encoder pretraining on the ship-only corpus and the pre-registered ablation against round 1

**Directive:** `directives_and_reports/R07_directive_phase3_pretraining_ablation_2026-10-07.md`
**Executor:** Claude Code (Fable 5.1) on Sherlock, 2026-10-07 → 2026-10-09 (one instance throughout; Steve ruled on the array throttle mid-run, §7).
**Compute:** 123.8 GPU-h of the 300 GPU-h cap: 120.1 GPU-h logged by the 136 fine-tunings, 3 pretraining runs and the §3.3 test (`scores/compute_summary.json`), plus 3.7 GPU-h of the first P-V1 pretraining attempt, which failed at step 23,000 and wrote no log (`sacct` elapsed 03:39:46). Jobs: staging 46903869, §1.4 test + cache 46905122, §3.3 test 46904378, pilot 46906534, pretraining P-V2 46906576, P-V1 46906575 (failed) → 46999622, scratch-V2 array 46906577, pretrained-arm array 46999627, scorer self-test 46906532, scoring/freeze 47119698, preservation 47119699.
**Outputs:** `/scratch/users/scdobbs/grl_review/R07/` (staged corpus, window cache, pretraining runs, fine-tuning models, scores, checks, provenance, logs), copied to OAK `grl_review_archive/R07/` with SHA256SUMS verified at the destination; this directory holds `scripts/` as run, `tables/` (every table below is generated: `scripts/report_tables_r07.py`, `scripts/score_r07.py`, spliced verbatim by `scripts/assemble_report_r07.py`), `freeze_record.*`.
**Certainty labels:** CONFIRMED / CONTRADICTED / COULD NOT CHECK. Meters unless marked. Every ρ is the eval_v1.1 statistic (within-pair Spearman ρ, median over eligible pairs, median over the 16 units, bootstrap over units, exact one-sided sign test).

---

## 1. Bottom line

1. **Challenge findings, guard, equivalence tests.** Eleven notes in §2, none blocking; one HOLD (H1, a code-revision question). Guard self-test 72 / 72 probes blocked (6 lockbox / confirmatory paths × 12 readers, now including geopandas, pyogrio and pyarrow parquet) at the start of every script; every frozen input (R05 §6, the R06 freeze record, the corpus manifest and `SHA256SUMS`) matched in every run (`provenance/*.json`). **§1.4: 200 / 200 development windows byte-identical** to the stored R06 input arrays and 200 / 200 to the recomputed R06 builder, V2 channels and masks included (`checks/equiv_windows.json`). **§3.3: identical** — the R07 code with init = none reproduces R06's V1 / lu00 / 20261002 run bit for bit on an H100 (max |Δ| 0 for predictions and weights, final 100-step loss 0.075181 in both; `checks/scratch_equiv/result.json`). CONFIRMED.
2. **Compute.** 123.8 GPU-h against the 300 GPU-h cap; the post-pilot estimate was 125.1 GPU-h (`compute_estimate.json`). One failure: the first P-V1 pretraining attempt died at step 23,000 (DataLoader worker killed on a node whose 1 TB of memory was fully allocated); it was rerun from scratch with the same seed and data order under a memory-bounded loader (§4), and both fine-tuning arrays completed with no requeue (`task_ledger.json`).
3. **Pretraining data** (`cache/cache_manifest.json`, `cache/per_bin.csv`): 49,225 of 51,322 corpus blocks for the reconstruction task (30 below 10 m and 2,067 above 256 m excluded), 45,705 for the sounding-spread task; 13 held-out cruises (2,822 blocks) chosen by family; 3,840,000 training windows, exactly 12.5 % per cell bin (479,750–481,316), none dropped; 16,384 held-out windows.
4. **Pretext results on the held-out cruises** (`pretrain/<body>/eval.json`): P-V1 reconstruction MSE 0.092 against the mean-fill baseline 0.626 (ratio 0.146), log₁₀ `ship_rsd` block-level Spearman ρ 0.633; P-V2 0.090 (ratio 0.143) and ρ 0.736. The pilot passed its rule (0.847 → 0.114; 0.865 → 0.558). Within-window ρ stays low (0.13 / 0.24): most of the spread skill is between windows.
5. **Primary comparison (§4.1), P-V1 − V1, seed-averaged, A_fine / tile: Δ median ρ +0.043 [−0.179, +0.105], 9 / 16 units better, sign-test p = 0.402 → under the fixed rule, "no detectable effect" (the CI includes 0).** P-V1 scores 0.560 [0.299, 0.586] against V1's 0.517 [0.378, 0.641] (`scores/R07_scores.json:primary`). The gain concentrates on lu05, lu07, lu09 and lu10; the losses on lu02 (0.669 → 0.342), lu11 (0.485 → 0.299) and nu00 (0.389 → 0.292) widen the lower bound.
6. **P-V2 − V2 (seed-averaged): −0.058 [−0.208, +0.060], 4 / 16 units better, p = 0.989** ("no detectable effect" under the same rule; lu02 0.734 → 0.062 and lu11 0.471 → 0.097 drive it). **Against the bar** (GBT_ship 0.510 [0.362, 0.604]): P-V1 +0.050 [−0.176, +0.088] (9 / 7), V1 +0.007 [−0.074, +0.121] (6 / 10), P-V2 −0.019 [−0.175, +0.082] (8 / 8), V2 +0.039 [−0.073, +0.154] (10 / 6). **No arm beats the bar**; none approaches the 12 of 16 units a one-sided sign test needs.
7. **Seed spread** (max − min of the three medians): P-V1 0.027, V1 0.023 (R06), P-V2 0.019, V2 0.016; the P-V1 − V1 difference (+0.043) is below twice either spread.
8. **EW0207 and native 7.** EW0207 tile ρ (all-unit models, seed-averaged where three exist): P-V1 0.564, V1 0.539 (seed 20261002 only), P-V2 0.542, V2 0.522 (its seed-20261004 model scores 0.313); references ref_s_lr 0.552, GBT_noship 0.556, GBT_ship 0.467. Native 7 against R02 σ_m (tile): P-V1 +0.070 [−0.224, +0.085] (3 / 7), V1 +0.024, P-V2 +0.035, V2 +0.061; every CI includes 0.

**HOLD for Steve (1):**

- **H1 — one code revision between the two pretraining runs.** After the first P-V1 attempt failed, `pretrain_r07.py` was patched (bounded shard mmaps per loader worker, prefetch 2, resume checkpoints) and P-V1 was retrained; P-V2 and the 34 scratch-V2 fine-tunings had already run under the earlier revision. The training mathematics, seed, data order and schedule are unchanged and `train_r07.py` was not edited, but the combined code hash recorded in the logs differs (`3d6dcae7…` for P-V1 and the 102 pretrained-arm fine-tunings, `4526a59e…` for P-V2, the pilot and the 34 V2 fine-tunings; `freeze_record.md`), and the earlier `pretrain_r07.py` text is not preserved. Options: (a) accept as documented; (b) rerun P-V2 pretraining under the final revision (about 12 GPU-h) and re-fine-tune P-V2 (41 GPU-h) for a single hash; (c) rerun only P-V2 pretraining and check that its body is bit-identical to the archived one (12 GPU-h), which would close the question without touching the arms.

**Not pursued (one line each):**

- Why both pretrained arms lose on the low-amplitude units lu02 and lu11 (and why P-V1 gains on lu05 / lu07 / lu09 / lu10).
- Why the V2 seed-20261004 all-unit model scores 0.313 on EW0207 while the other two V2 seeds score 0.50–0.53.
- Whether the low within-window spread ρ (0.13 / 0.24 against 0.63 / 0.74 pooled) means the pretext learns window-level level rather than sub-window structure.
- A target normalised by the masked window's statistics (§2 item 2), other masking ratios, schedules or linear probes (§8).
- The hardware mix (A100-40 / A100-80 / H100) across folds and arms (recorded per training).
- Per-pair tables for the multi-pair units.

---

## 2. Challenge clause (§0) — wrong or underspecified instructions, with evidence, and the executor's calls

1. **§1.4 "256 × 256 px at that hr_res" gives very few LR cells on the coarse bins.** The development representation sets the LR tile to `lr_px = max(2, round(256 · hr_res / cell))` cells per side (extract_clean_patches.pair_setup). With hr_res ≈ 1 m (11 of the 16 development units) a block with a 94 m cell (the corpus median) yields a 3 × 3 LR tile, and bins 6–7 (114–256 m) yield 2 × 2; the §1.4 50 % validity rule then admits windows with 2–9 valid cells and the §2.2 30 % mask marks 1–3 of them. Implemented as written (the directive fixes both the window and the representation); the §1.2 bin sampling gives these coarse windows 3/8 of the training presentations. Evidence: `cache/per_bin.csv` and the `lr_px` field of the cache shards. Not a HOLD.
2. **§2.2 item 1: which window's statistics normalise the target and the input is not stated.** Implemented literally: the masked window goes through the unchanged builder, so its mu_lr, s_lr and FiLM vector come from the remaining cells (exactly as missing LR does in development), while the target is the builder's normalised bicubic of the *unmasked* window (its own mu_lr / s_lr). The mean-fill baseline is the masked input's bicubic (missing cells ← mu_lr of the remaining cells) expressed in the target's units. A target normalised by the masked window's statistics would be a variant not listed (§8) and was not run.
3. **§2.2 item 2: "area-weighted median of the cells covering each 32-px block; non-covered blocks masked" leaves the coverage rule and log₁₀ 0 open.** Implemented: a block counts as covered when native cells with finite `ship_rsd` overlap ≥ 50 % of its area (overlap area = weight); `log₁₀ max(rsd, 1 mm)`, the R06 product-normalisation floor (`ship_rsd` is 0 in a cell whose ≥ 3 soundings agree to the millimetre). Standardisation uses the mean 0.326 and SD 0.581 of the 33.0 M covered training-cache blocks (`cache/cache_manifest.json:rsd_log10_standardisation`).
4. **§1.4 hr_res draw: "the R06 sampler (unit uniform, then pair uniform within unit)" is defined per fold.** Pretraining has no fold; the draw is over all 16 development units (units-v1) and their 38 pairs, never EW0207. The 16 units' HR resolutions are 1.0–5.0 m (11 units at ≈ 1 m).
5. **§1.4 "cell = the block's cell size" and the HR grid of a block.** `cell` is the manifest's nominal `cell_m` (the development FiLM also uses the manifest's nominal `lr_native_res_m`); the actual grid pixel differs by < 0.1 % (e.g. 43.808 vs 43.847 m). The synthetic HR grid has pixel hr_res, the block's CRS, and is anchored at the grid's top-left corner; window positions are integer px on it. LR validity of a block is "finite grid cell" (blocks carry no lr_valid.tif).
6. **§1.4 "rejected and redrawn if less than 50 % of its area has valid LR"** is applied to the window's lr_px² LR cells, with the additional requirement of ≥ 3 valid cells so that ≥ 2 remain after masking (`build_inputs` needs 2). Positions are redrawn up to 30 times; no window needed more (0 dropped, `cache/ledger.csv`).
7. **§1.5 cache size.** One unique window per training presentation (60,000 × 64 = 3,840,000) plus 16,384 held-out windows, so no window is seen twice and the mask of a window is fixed by its id. The cache stores the LR-level tiles (area-averaged LR, validity, the 4 raw products on the window's LR grid, the spread target); bicubic, normalisation, FiLM and the V2 channels are built at load time by the same R06 code (the §1.4 test covers the whole path).
8. **§2.3 pilot schedule.** The pilot's cosine schedule runs over its own 2,000 steps (the pilot is a timing and sanity check, not a truncated full run).
9. **§4.2.4 "EW0207 ... all four arms (all-unit models)".** V1 has one all-unit model (R06, seed 20261002); V2 (R06 seed 20261002 + the two R07 seeds), P-V1 and P-V2 have three. The table gives per-seed rows and seed-averaged rows where three exist.
10. **§3.3 hardware.** The test ran on the GPU type and software stack R06 used for that fold (H100 80GB HBM3, torch 2.4.0a0); as in R06, the fine-tuning arrays landed on A100-40GB, A100-80GB and H100 nodes as the scheduler allocated them (63 / 61 / 12 trainings), so the arm comparison also contains any hardware effect (recorded per training in `train_log.json:gpu`).
11. **Working tree.** The repository's working tree carries uncommitted edits from June 2026 to `models/`, `training/`, `preprocessing/data_utils.py`, `CLAUDE.md` and `METHOD.md` (pre-dating R04–R06 and present during those runs too); the functions the R06/R07 builders import (`bicubic_upsample`, `reproject_from_array`, `tile_ship_side`, `build_inputs`) are unchanged by the diff (`git diff`). COULD NOT CHECK that every imported module equals its committed version; the byte-identity tests (§1.4, §3.3) check what matters for this directive.

---

## 3. Design as implemented — CONFIRMED against code

| item | as run |
|---|---|
| fixed inputs (§0.2) | `r07_common.start()` in every script: R05 §6 hashes (via `r06_common.check_frozen`, 15 files), the 14 R06 scripts against `R06/freeze_record.json` and the R06 training-code hash, the corpus manifest and `SHA256SUMS` against the directive's two hashes, the pairs manifest v2.1 hash; every R06 prediction / checkpoint / log read is verified against the OAK `R06/models/SHA256SUMS` first. All matched in every run (`provenance/*.json`). |
| guard (§0.2) | R04 guard (builtins.open, rasterio.open, numpy.load, pandas.read_parquet) extended to geopandas.read_file / read_parquet, pyogrio.read_dataframe / read_arrow / raw.read, pyarrow.parquet.read_table / ParquetFile / ParquetDataset; self-test 72 / 72 probes blocked at every script start. |
| corpus staging (§1.5, §6) | `stage_corpus.sbatch`: OAK `pretrain_v1/` → `$SCRATCH/grl_review/R07/corpus/` (271 files, 19 GB), every file verified against `SHA256SUMS` at the destination; all later jobs read the scratch copy (`/vsitar/` on the staged tars) or the cache, never OAK. |
| blocks (§1.1) | cell 10–256 m inclusive; spread task = those with `products_usable` (verified equal to "all five products available and no QA flags"). |
| bins and sampler (§1.2) | 8 equal-width log₂ bins from log₂ 10 to log₂ 256; bin ~ uniform, block ~ uniform within bin, position ~ uniform over the integer px positions of the synthetic HR grid; hr_res ~ R06 development sampler (unit uniform, pair uniform; the pair's tiles_v1.1 hr_res). Seeds `default_rng([20261007, tag])`, positions `default_rng([20261007, split, window id])`. |
| held-out cruises (§1.3) | 5 % of corpus cruises per instrument family (≥ 1), `default_rng(20261007)`: 13 cruises, 2,822 blocks; no window from them enters pretraining; used only for the pretext curves. |
| windows (§1.4) | `r07_windows.extract_lr_tile` (= `extract_clean_patches.tile_ship_side` + the product `reproject_from_array` call of `build_inputs_r06.window_inputs`) at cache time; `r07_windows.assemble` (= `build_inputs` + FiLM + the V2 product normalisation / fill / bicubic / nearest-up mask, verbatim) at load time. Equivalence: 200 development windows (positions and hr_res from the R06 input records, ≥ 2 per pair, 38 pairs, 139 on product-usable pairs) byte-identical to the stored R06 arrays and to the recomputed R06 builder. |
| masking (§2.2.1) | 30 % of the window's valid LR cells (round), uniform; marked missing before `build_inputs`; loss = MSE over the HR px of the masked cells (nearest-upsampled cell map) against the unmasked window's normalised bicubic. |
| spread target (§2.2.2) | area-weighted median of log₁₀ max(ship_rsd, 1 mm) over the native product cells overlapping each of the 16 central-tile blocks; covered if ≥ 50 % of the block area has finite rsd; standardised by the training-cache mean / SD; MSE over covered blocks of windows from usable blocks. For P-V2 the `ship_rsd` channel and its mask are zero on every window. |
| model (§2.1) | `r07_pretext_model.PretextNet`: `r06_model.UNetBody` (module `body`, 1,912,992 parameters) [+ `r06_model.ProductBranch` (module `branch`, 133,408)], a 1 × 1 conv reconstruction head (33) and the R06 block-head structure as the spread head (1,089 / 9,409; last layer zero-initialised). `body_state_dict()` = the `body.*` (+ `branch.*`) tensors; loaded into a fresh `BlockHeadNet` they leave exactly the four `head.*` tensors to the R06 zero-init (asserted in `train_r07.load_init`). |
| schedule (§2.3) | 60,000 steps, batch 64, AdamW 2e-4 / wd 1e-4, cosine to 1e-6, grad-clip 1.0, bf16 autocast, the 8 square symmetries per window (applied identically to input, target, mask map, products and the 4 × 4 maps), seed 20261007; final checkpoint. Loader (final revision): 12 workers, ≤ 6 shards mapped per worker, prefetch 2, resume checkpoint every 5,000 steps (never used: the rerun completed in one job). |
| fine-tuning (§3.1) | `train_r07.py` = `train_r06.py` with the `--init` option (`load_init`) and arm names; same inputs (R06 input arrays), sampler, augmentation, within-pair centred loss, 8,000 steps, optimiser, schedule, product dropout 0.3 (V2), final checkpoint, LOUO over the 16 units + the all-unit model for EW0207 (never trained on, asserted). Seeds 20261002 / 20261003 / 20261004 for P-V1 and P-V2; V2 completed with 20261003 / 20261004 (init none). |
| scoring (§4) | `score_r07.py`: predictions as columns of the R05 bar tables, scored only through `eval_v1_1` (`score_one`, `compare`, `median_ci`) with the R05 `pair_rules.json`; seed-averaged = mean of the three seeds' log₁₀ block predictions, tile = RMS of the 16 blocks (as R06). Self-tested on synthetic predictions (`test_score_synthetic_r07.py`, CPU job 46906532; reproduces the R06 V1 rows exactly). |

---

## 4. Pretraining data (§1)

{{table:pretraining_blocks}}

Blocks per cell bin, held-out split, product usability and the effective share of training windows per bin (§1.2):

{{table:pretraining_bins}}

Held-out cruises (§1.3):

{{table:heldout_cruises}}

Cache (§1.5):

{{table:cache_summary}}

Equivalence checks (§1.4 and §3.3):

{{table:equivalence}}

---

## 5. Pretraining (§2)

**Pilot (§2.3), P-V1 body, 2,000 steps:**

{{table:pilot}}

**P-V1 body** (held-out curve every 2,000 steps on 4,096 windows; the final row is the whole held-out set):

{{table:pretext_P-V1}}

**P-V2 body:**

{{table:pretext_P-V2}}

QC figures (6 held-out windows per body: masked input, masked cells, reconstruction, target, predicted vs actual log₁₀ `ship_rsd`): `pretrain/P-V1/qc_P-V1.png`, `pretrain/P-V2/qc_P-V2.png` (scratch and OAK). Pretext heads are kept in `ckpt_final.pt`; the bodies alone in `body_P-V1.pt` / `body_P-V2.pt` (sha256 in `freeze_record.md`).

**The failed first P-V1 attempt** (job 46906575, node sh03-14n03 shared with five V2 array tasks, 1 TB RAM with 1,007 GB allocated): stopped at step 23,000 with `RuntimeError: DataLoader worker (pid 26050) is killed by signal: Killed`; its logs and curves are kept at `pretrain/P-V1_failed_46906575/` (its held-out curve to step 22,000 matches the rerun's at the same steps, as expected with the same seed and data order). The rerun (46999622) ran on an A100-40GB, 9.18 h, no resume needed.

---

## 6. Fine-tuning and scores (§3–§4)

All numbers below are eval_v1.1 statistics from `scores/R07_scores.{md,json}`; prediction tables `scores/pred_tile_r07.parquet`, `scores/pred_block_r07.parquet`, `scores/pred_*_EW0207_r07.parquet`. V1 (three seeds) and V2 seed 20261002 are the R06 outputs, verified against the OAK `R06/models/SHA256SUMS` before use.

{{scores:§4.1 Primary}}

{{scores:§4.2.1 P-V2}}

{{scores:§4.2.2 Each arm}}

{{scores:§4.2.3 Seed spread}}

{{scores:§4.2.4 EW0207 (lu16), tile}}

{{scores:§4.2.4 EW0207 (lu16), block}}

{{scores:§4.2.5 Native}}

{{scores:§4.2.6 Block scale}}

{{scores:§4.2.7 Per-unit}}

Secondary endpoints (T_fine, E_1m) and the full comparison tables with capture20 and log-log R² are in `scores/R07_scores.md`.

**§4.3.** This directive adopts no model and sets no adoption criteria. The R06 note applies: selecting among variants on the same 16 units biases the winner's development score upward.

---

## 7. Compute and Sherlock compliance (§6)

Estimate after the pilot and the §3.3 test (gate: `compute_estimate.json:within_budget`):

{{table:compute_estimate}}

Used (logged; add 3.7 GPU-h for the failed P-V1 attempt, which wrote no log → 123.8 GPU-h):

{{table:compute_used}}

Ledger of skipped, failed and requeued items (`task_ledger_r07.py` from `sacct`; the PENDING / RUNNING rows are the scoring and preservation jobs at the time the ledger was generated):

{{table:ledger}}

**Sherlock / SERC rules, as followed (Steve asked for this check during the run):**
- Every job has an explicit time limit; GPU jobs request `-G 1` and select hardware with `-C` constraints (H100 for the §3.3 test; H100 | A100 for pretraining and the arrays after the H100 pool proved fully allocated), never `--exclude`.
- At most 8 concurrent GPUs on serc by construction: the two pretraining jobs ran alongside the scratch-V2 array throttled to `%6`; the pretrained-arm array was chained with `--dependency=afterok` on the P-V1 pretraining job and throttled to `%8`, lowered to `%7` on Steve's request (2026-10-08 12:00) while other jobs of his were running. Concurrency never exceeded 7 after that point and 8 before it.
- Login nodes were used for editing, inspection, job submission and sub-minute checks only. One lapse: the synthetic scorer self-test was started on a login node, recognised as sustained computation after about a minute, stopped, and re-run as CPU job 46906532. Staging, the §1.4 test, cache building, scoring, the freeze record and preservation all ran as Slurm jobs.
- Jobs read only `$SCRATCH` (staged corpus tars, the window cache, the R06 input arrays) and a node-local copy of the cache (`$L_SCRATCH_JOB`, 21 GB, staged in 74–101 s per pretraining job); OAK is read at job start only for the small `SHA256SUMS` files of the fixed-input checks. No job reads data from `$HOME` or `$GROUP_HOME` (the Python environment itself lives in `$HOME`, as in R04–R06).
- Scheduler etiquette: stages chained with dependencies; status checks at ≥ 10-minute intervals on the first day and a few times a day afterwards; no `watch` or shell polling loops; no duplicate submissions; the one failed job was resubmitted after its cause was identified (§5).
- Group scratch before launch: 1.8 TB of 100 TB and 977.5 K of 20 M files (`sh_quota`, 2026-10-07 08:50); user scratch 126 GB. The R07 footprint on scratch is about 41 GB (corpus copy 19 GB, cache 21 GB, models and pretraining < 1 GB), all regenerable or archived.

---

## 8. Preservation and freeze (§5)

1. **Code.** The window builder, cache builder, pretext model, pretraining, fine-tuning, scoring, ledger, tables and driver scripts under `R07/scripts/`; per-file sha256 and the combined code hash in `freeze_record.md` / `freeze_record.json`. Two combined hashes appear in the logs (HOLD H1): `4526a59e…` (pilot, P-V2 pretraining, the 34 V2 fine-tunings) and `3d6dcae7…` (P-V1 pretraining and the 102 pretrained-arm fine-tunings); the only file edited between them is `pretrain_r07.py` (loader memory bound, prefetch 2, resume checkpoints; §5).
2. **Artifacts on OAK `grl_review_archive/R07/`** with SHA256SUMS verified at the destination (job 47119699): `pretrain/` (both bodies, both `ckpt_final.pt` with the pretext heads, the pilot, the failed attempt's logs, curves, QC figures), `models/` (every fine-tuning fold's checkpoint, curve, log and prediction table: 136 folds), `scores/`, `checks/` (§1.4 and §3.3 results, scorer self-test), `provenance/`, `logs/`, `root_files/` (compute estimate, task ledger, job ids), `cache_record/` (cache manifest, shards table, per-bin table, held-out cruise list, ledger, build log; not the regenerable shards), `scripts/`. Counts verified at the destination: pretrain 24, models 544, scores 8, checks 150, provenance 148, logs 102, root_files 2, cache_record 7, scripts 24 — 1,009 files (`preserve_47119699.out`); the scripts directory as committed (incl. `assemble_report_r07.py`, added after the first copy) is archived additively as `scripts_final/`.
3. **Tag and ledger.** The R07 commit is tagged `phase3-r1`; `CHANGE_LEDGER.md` row 21 describes Phase 3 round 1.

---

## 9. Paths and reproduction

Scratch `/scratch/users/scdobbs/grl_review/R07/` — `corpus/` (staged tars, regenerable from OAK), `cache/` (regenerable: `build_cache_r07.py`), `pretrain/<body>/`, `models/<arm>/seed<seed>/fold_<fold>/`, `checks/`, `scores/`, `provenance/`, `logs/`. OAK `grl_review_archive/R07/` (everything but the corpus copy and the cache shards). Run order: `sbatch stage_corpus.sbatch` → `sbatch build_cache.sbatch` (§1.4 test, then the cache) → `sbatch scratch_equiv.sbatch` (§3.3) → `sbatch --export=ALL,BODY=P-V1,PILOT=1 pretrain.sbatch` → `sbatch --export=ALL,BODY=P-V1 pretrain.sbatch` and `BODY=P-V2` (each runs `estimate_r07.py` and the §2.3 / §6 gates first) → `sbatch --array=1-12%6 --export=ALL,TASKS=tasks_scratch_v2.txt train_array.sbatch` → `sbatch --array=1-30%8 --export=ALL,TASKS=tasks_pretrained.txt train_array.sbatch` → `score_r07.py && task_ledger_r07.py && report_tables_r07.py && freeze_r07.py` (CPU job via `cpu_step.sbatch`) → `sbatch preserve_r07.sbatch` → `assemble_report_r07.py`.

R07 commit, tagged `phase3-r1`: `21596cc7e46fc074b6546e9a9b1919c7fe573425` (parent 206963c, the R06 report hash note; phase2-r1 = 683a99f; phase0-freeze = 866320b). Job logs under `R07/logs/` are gitignored; they are on scratch and OAK `grl_review_archive/R07/logs/`.
