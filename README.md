# Ship-multibeam roughness and unresolved seafloor relief: analysis, predictors and blind-test scoring

This repository holds the analysis side of the paper below: it turns co-located ship and AUV multibeam grids into
per-tile targets and ship-only features, evaluates how well ship-grid roughness ranks the AUV-measured fine-scale relief,
trains the gradient-boosted and CNN predictors, and runs the two pre-registered blind tests. The grids, masks and manifests it
reads are produced by the companion acquisition repository
([10.5281/zenodo.21300925](https://doi.org/10.5281/zenodo.21300925)) and are published as the data record
[10.5281/zenodo.23515276](https://doi.org/10.5281/zenodo.23515276). The repository also keeps, marked as superseded, the
rectified-flow super-resolution model of the submitted manuscript.

> Dobbs, S. C. "Ship Multibeam Roughness Ranks Unresolved Seafloor Relief at Unseen Sites: A Pre-Registered Blind Test."
> *Geophysical Research Letters*, 2026GL125574 (in revision).

## The question and the headline result

Does the roughness of an ordinary ship multibeam grid tell you where the seafloor has fine-scale relief that the ship grid
cannot resolve? The AUV grid gives the truth for each 128-pixel tile: `A_fine`, the RMS of the band-passed relief between
the ship cell and the AUV pixel. The ship-only feature `s_lr` is the standard deviation of the ship grid in the 256-pixel
window around the tile. Within the development sites, `s_lr` ranks `A_fine` across tiles with a Spearman correlation of about
0.4. Pre-registered blind test 2, on eight sites chosen and sealed before any ship data was examined, confirmed this: pooled
rho 0.43, 95 % block-bootstrap interval 0.28 to 0.53 over the five testable sites, with the frozen `s_lr` as the primary
candidate and the predictors GBT_ship and P-V1 as secondary comparisons. Blind test 1, which deliberately reached outside
the development range of ship cell sizes, was not confirmed. Every number traces to a tagged freeze, a frozen tile table and a
cited acquisition record.

## Data

The data record [10.5281/zenodo.23515276](https://doi.org/10.5281/zenodo.23515276) (CC BY-NC-SA 4.0) holds the 52 pairs
with rasters (39 development, 5 blind test 1, 8 blind test 2), one row per pair, one row per scored tile with `s_lr`,
`A_fine`, `A_fine_int`, the interior and exclusion flags and the held-out GBT_ship and P-V1 predictions, both pre-registrations,
the committed blind calls, the deviation notes and the frozen scoring outputs. Raw swath is not redistributed; it comes from
NOAA NCEI, MGDS / IEDA, PANGAEA and USGS, each to be cited as listed in the record's `SOURCE_LICENSES.csv`. The pretraining
corpus for P-V1 (ship-only 10 km blocks from 400 NCEI cruises) is not redistributed.

## Quick start

### Environment

Results were produced on Stanford's Sherlock cluster: Python 3.12.1, PyTorch 2.4 (module `py-pytorch/2.4.1_py312`, CUDA
12.6, cuDNN 9.4), torchvision 0.19.1, and the packages pinned in `requirements.txt`. CNN training and pretraining ran on
NVIDIA A100, H100 and L40S GPUs; everything else, including both blind tests, ran on CPU.

```bash
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt            # PyTorch per https://pytorch.org for your CUDA
# On Sherlock: module load python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312
```

### Reproduce one pair's score from the data record

Download `discol_so242_1.zip`, `tiles.parquet` and `load_pair.py` from the data record into one folder, then:

```bash
python load_pair.py <folder> discol_so242_1
```

It opens the pair's grids, applies the frozen scoring rule (interior view if at least 10 tiles in at least 8 blocks, else
the fallback view on `A_fine_int`) and prints the Spearman rho of `s_lr` against the target next to the frozen value.

### Rescore the development table and both blind tests from the packaged tile tables

```bash
python grl_reviews-and-prompt-reports/scripts/rescore_from_package.py <folder>
```

With `tiles.parquet` and `pairs.parquet` from the data record in `<folder>`, this recomputes the per-pair rho of `s_lr`,
GBT_ship and P-V1 for every scored pair of the development set, blind test 1 and blind test 2 with the frozen rule, applies
the R17 exclusions, and checks each against the frozen value to three decimals; it runs in seconds. The bootstrap intervals and
pooled statistics come from the frozen scorers, `grl_reviews-and-prompt-reports/directives_and_reports/phase4/scripts/phase4_score.py`
and `phase5/scripts/phase5_score.py` (rehearsal mode; the single phase-mode runs are locked and were run once).

## Pipeline map

All paths below are under `grl_reviews-and-prompt-reports/directives_and_reports/` (written `DR/`) unless stated. Each
round `Rnn/` holds its scripts, tables, records and freeze notes; the directive and report texts of each round are not part
of the public repository (the ledger summarizes every change).

| Stage | Entry script | Inputs | Outputs | Round |
|---|---|---|---|---|
| Clean windows (ship side only reaches the inputs) | `preprocessing/extract_clean_patches.py` | the acquisition manifest and the harmonized `hr.tif`, `lr.tif` and masks | 256-px windows at stride 128 with `s_lr` | R02 |
| Targets and tiles (`A_fine`, `T_fine`, band-passed fields; tiles_v1) | `DR/R04/scripts/build_tiles_targets.py` with the frozen definitions `DR/R04/scripts/targets_v1.py` | windows, `hr.tif`, the pair's `k` and `lr_native` | `tiles_v1.parquet`, `blocks_v1.parquet`, per-pair field mosaics (archive `R04/`) | R04 |
| Tile selection tiles_v1.1 and the frozen feature bank | `DR/R05/scripts/build_tiles_v1_1.py`, `DR/R05/scripts/features_v1_1.py` | tiles_v1, blocks_v1 | `tiles_v1_1.parquet` (26,154 tiles, 39 pairs) | R05 |
| Interior targets (`A_fine_int`, interior flag) | `DR/R09/scripts/build_int_r09.py` with `DR/R09/scripts/targets_v1_int.py` | the frozen `F_hr` fields and `joint_v1` masks | `targets_int/tiles_int.parquet` | R09 |
| Evaluation eval_v1.1 (within-pair Spearman, unit block bootstrap, seed 20261001) | `DR/R05/scripts/eval_v1_1.py` | tile tables and predictions | per-pair and pooled rho with 95 % intervals | R05 |
| GBT_ship (gradient-boosted bar on the ship-only features; nested leave-one-unit-out) | `DR/R05/scripts/run_bar_v1_1.py` with `bar_v1_1.py`; the all-16-unit refit `DR/R09/scripts/refit_gbt16_r09.py` | `tiles_v1_1`, `features_v1_1` | `pred_tile.parquet`; artifact `R09/artifacts/gbt_ship_all16_A_fine_tile.joblib` (sha256 `ef934672…`) | R05, R09 |
| CNN inputs and variants V0 to V3 (V1 three seeds) | `DR/R06/scripts/build_inputs_r06.py`, `DR/R06/scripts/train_r06.py`, model `DR/R06/scripts/r06_model.py` | tiles_v1.1 windows, ship side only | `R06/models/<variant>/seed<seed>/fold_<unit>/` with `pred.parquet` | R06 |
| Pretraining on the ship-only corpus and fine-tuning (P-V1, P-V2) | `DR/R07/scripts/build_cache_r07.py`, `pretrain_r07.py` (60,000 steps, seed 20261007), `train_r07.py --init` | the pretraining corpus `pretrain_v1`, R06 inputs | `R07/pretrain/P-V1/body_P-V1.pt`; `R07/models/P-V1/seed{20261002,20261003,20261004}/fold_ALL/ckpt.pt` | R07 |
| Development scoring and the unit set (ten real-relief, six artifact-affected) | `DR/R08b/scripts/score_r08b.py`, `DR/R09/scripts/score_r09.py` | predictions, tiles_int, Steve's R08 figure-set calls | `R09/scores/master_tile_r09.parquet`, `R10/tables/artifact_rule.md` | R08, R08b, R09, R10 |
| Blind test 1 (Phase 4): build, blind pages, predict, score once | `DR/phase4/scripts/phase4_build.py`, `phase4_figures.py`, `phase4_predict.py`, `phase4_score.py` (common module `phase4_common.py` with the guard and locks) | the acquisition supplement `pairs_phase4_sealed.parquet` and its freeze record; the frozen GBT_ship and P-V1 artifacts | archive `phase4/{build,figures,predictions,score_inputs,scores}`; `R13/tables/per_pair.md` | R10 to R13 |
| Blind test 2 (Phase 5): parameterized copies of the Phase 4 scripts, R17 exclusions | `DR/phase5/scripts/phase5_build.py`, `phase5_figures.py`, `phase5_predict.py`, `phase5_score.py`; `DR/R17/scripts/make_exclusions_r17.py` | `pairs_phase5_sealed_v4_1.parquet`, acquisition freeze v4.2 | archive `phase5/scores/{results_D1.json,per_pair_D1.csv}` (also `R18/tables/`); `R17/exclusions_phase5_r17.json` | R16, R16c, R17, R18, R19 |
| Exploratory analyses (not in the claims) | `DR/R14/scripts/` (ship coarsening), `DR/R15/` (Goff abyssal-hill comparison) | | reports in `R14/`, `R15/` | R14, R15 |

The main ledger `grl_reviews-and-prompt-reports/CHANGE_LEDGER.md` has one row per departure from the submitted-paper
conventions. `DR/INTERFACE_CONTRACT_v2.1.md` is the contract for the per-sounding ship product rasters the acquisition side
delivers.

## Frozen versions

| Tag | Date | What it froze |
|---|---|---|
| `r04-as-run` | 2026-10-01 | R04 as run: targets_v1, eval_v1, features_v1, bar_v1 |
| `phase0-freeze` | 2026-10-01 | targets_v1, eval_v1.1, features_v1.1, bar_v1.1 and `tiles_v1_1.parquet`; the development table every later result reads |
| `phase2-r1` | 2026-10-02 | the CNN variants V0 to V3 retrained and scored (R06) |
| `phase3-r1` | 2026-10-09 | P-V1 and P-V2 pretraining and the pretraining ablation (R07) |
| `phase4-prereg` | 2026-10-09 | pre-registration v1 of blind test 1 and the interior targets (R09) |
| `phase4-prereg-v2` | 2026-10-10 | `R11/PREREG_phase4_v2.md`, signed |
| `phase4-prereg-v2.1` | 2026-10-12 | `R12c/PREREG_phase4_v2_1.md`, signed; `phase4_common.py` as run; the governing document of blind test 1 |
| `phase5-prereg-v3` | 2026-10-13 | `R16/PREREG_phase5_v3.md`, signed |
| `phase5-prereg-v3.1` | 2026-10-13 | `R16c/PREREG_phase5_v3_1.md` and its sign-off; the governing document of blind test 2 |
| acquisition `phase4-acq-freeze`, `phase5-acq-freeze-v4.2` (companion repo) | 2026-10-10, 2026-10-14 | the sealed acquisition chains whose freeze records `phase4_common.py` and `phase5_common.py` pin by sha256 |
| `grl-revision-v2.0` | 2026-10-20 | this repository as archived for the GRL revision |

## Pre-registrations and deviations

- Blind test 1: `DR/R12c/PREREG_phase4_v2_1.md` (governing; v2 in `DR/R11/`, v1 in `DR/R09/`), sign-off
  `DR/R12c/SIGNOFF_phase4_prereg_v2_1.md`, Steve's committed calls `DR/R13/phase4_calls.csv`.
- Blind test 2: `DR/R16c/PREREG_phase5_v3_1.md` (governing; v3 in `DR/R16/`, diffs in `R16/tables/` and `R16c/tables/`),
  sign-off `DR/R16c/SIGNOFF_phase5_prereg_v3_1.md`, calls `DR/R18/phase5_calls.csv`, design spec
  `DR/phase_5/PHASE5_design_spec_2026-10-13.md`.
- Deviation notes, one line each:
  - `DR/R11/DEVIATION_phase4_01.md`: Phase 4 acquisition step 1 stopped on a directory-mode check and was completed once by a frozen completion script.
  - `DR/R11/DEVIATION_phase4_02.md`: the acquisition run-record lock reads a citation record (key-name mismatch in the completion record).
  - `DR/R12c/DEVIATION_phase4_03.md`: the confirmatory allow-list fix and pre-registration revision v2.1.
  - `DR/R13/DEVIATION_phase4_04.md`: the calls file joined to pairs before scoring.
  - `DR/phase5/DEVIATIONS_phase5.md`: seven departures from v3.1 as written (governing document; acquisition freeze v2 to v3 to v4.2; the 30 m bound replaced by the local quadrant test; the 22 Escanaba tiles excluded; three scripts checked as tagged file plus recorded diff; the declared S1 and S2 rows; MV1003 GBT_ship unavailable).
- Rulings: `DR/R08b/ruling_unit_set_2026-10-09.md` (the ten real-relief and six artifact-affected units); other rulings are recorded in the ledger.

## What is superseded

The submitted manuscript used a heteroscedastic U-Net (`models/het_production_v1/`, config of record
`models/het_production_v1/step0_bakeoff/het_louo.py`) whose per-pixel sigma product was meant to flag unresolved relief, and
the project before it trained a rectified-flow super-resolution model (`models/diffusion.py`, `models/unet.py`,
`models/losses.py`, `training/`, `preprocessing/synthetic_lr.py`) on a synthetic-LR regime in which ship-like grids were
simulated by degrading AUV grids. These are kept for the record and do not support the revised paper's claims: the review
found the sigma product's apparent skill came from information leaking from the AUV grid into the inputs (R01, R02), and the
revised analysis uses only ship-side inputs. `CLAUDE.md` and `METHOD.md` describe the old project and carry a superseded
banner; `models/het_clean_r02/` is the leak-free retrain of R02.

## Hardware and runtime

| Stage | Hardware | Approximate time |
|---|---|---|
| Clean windows, targets, tiles and features (R02, R04, R05) | 16 to 32 CPUs | 25 to 40 min per stage |
| GBT_ship nested leave-one-unit-out refit (R05) | 32 CPUs | 1 h 13 min; the all-16-unit refit 1 min |
| CNN variants V0 to V3, 100 trainings of 8,000 steps (R06) | A100 40/80 GB, L40S, H100 | about 59 GPU-hours in total (0.08 to 0.36 s per step) |
| P-V1 and P-V2 pretraining, 60,000 steps each (R07) | A100 40 GB | 9.2 h and 12.4 h |
| Fine-tuning P-V1, P-V2 and V2 (R07) | A100, H100 | about 124 GPU-hours in total |
| Blind test 1 build, blind pages, predict (CPU inference), score (R12c, R13) | 16 CPUs, 64 GB | 17 min, 28 min, 34 s |
| Blind test 2 build, predict, score (R17, R18) | 16 CPUs | 10 min, 23 min, 57 s |
| Rescoring from the packaged tile tables (quick start) | laptop | seconds |

## Citation and license

Code: MIT (`LICENSE`). Cite the paper, the data record and this code record:

- Dobbs, S. C. (in revision). Ship Multibeam Roughness Ranks Unresolved Seafloor Relief at Unseen Sites: A Pre-Registered
  Blind Test. *Geophysical Research Letters*, 2026GL125574.
- Dobbs, S. C. (2026). Co-located ship and AUV multibeam bathymetry corpus: processed grids, masks, co-registration and
  tile tables (version 1.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.23515276 (CC BY-NC-SA 4.0).
- Dobbs, S. C. (2026). Ship-multibeam roughness analysis, predictors and blind-test scoring code (version 2.0) [Software].
  Zenodo. https://doi.org/10.5281/zenodo.21300922 (this repository; see `CITATION.cff`). The acquisition code is
  https://doi.org/10.5281/zenodo.21300925.

## Paths to change

The scripts name the Sherlock layout they ran on: the data root and archive (`DR/R03/scripts/r03_common.py` `DATA`,
`DR/R04/scripts/r04_common.py` `ARCHIVE` and `MANIFEST`), the pretraining corpus (`DR/R07/scripts/r07_common.py`
`OAK_CORPUS`), the acquisition repository (`r04_common.py` `MANIFEST`, `DR/R08/scripts/r08_common.py` `ACQ_REPO`,
`DR/phase4/scripts/phase4_common.py` and `DR/phase5/scripts/phase5_common.py`), the scratch roots in each `r0n_common.py`,
and the virtual environment and module lines in the `.sbatch` files.
