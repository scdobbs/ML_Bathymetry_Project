# het_production_v1 — the paper's deployable heteroscedastic model

This is the durable, version-controlled home of **het_production_v1** (within-pair total-error
R² = **0.555**), the model retained as the paper's deployable after two pre-registered comparisons
(Directive 54 rank-hybrid probe → RETAIN; Directive 55 expansion → see REPORT_55).

## What this model is

A compact heteroscedastic CNN (`HeteroscedasticUNetNative`, ~1.9 M params) that predicts, from a
surface-ship MBES grid, a per-pixel (mean, log-variance) for the AUV-scale seafloor it fails to
resolve. Trained with masked Gaussian NLL, leave-one-unit-out across 8 native co-located AUV/ship
survey units. σ localizes *within a survey footprint* where the ship reconstruction is unreliable
(within-pair R² 0.555; top-20%-area error capture 0.502).

## Important: no original fold weights exist

The original LOUO run (`step0_bakeoff/het_louo.py`, un-seeded) **discarded its fold checkpoints** after
evaluation — only the per-patch σ parquet was persisted. This directory therefore preserves the model
as **code + config + seed + validated LOUO outputs**, which is what defines and reproduces it:

- `step0_bakeoff/het_louo.py` — architecture, hyperparameters, folds, caps (the config of record).
- `step0_bakeoff/step0_bakeoff.py` — band/coverage/feature helpers it imports.
- `step0_bakeoff/run_het_louo.sbatch` — launch config; `het_louo_32571349.out` — the training log.
- `step0_bakeoff/het_louo/het_louo_perpatch.parquet` — the LOUO per-patch σ (the literal 0.555 evidence).
- `step0_bakeoff/het_louo/*.json` — the downstream validations keyed to it (decomp/drape/guard5/heuristic/validation).
- `production_manifest.json` — sha256 inventory (11 files). This copy is checksum-identical to the
  scratch backup and the OAK copy (`/oak/.../auv_ship_colocated_bathy/models/het_production_v1/`).

## Loadable weights

Because no original weights survive, **`het_production_v1_repro`** (Directive 55 §C) provides loadable
per-fold + all-data checkpoints from a byte-faithful re-run of the same recipe, validated against this
parquet (V1–V4 gate). Use `het_production_v1_repro` when you need weights; use the parquet here as the
frozen ground-truth for any reproduction check.

Three independent homes: scratch backup, OAK, and this repo — checksums agree across all three.
