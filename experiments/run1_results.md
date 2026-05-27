# Run 1 Results — Rectified Flow Bathymetric Super-Resolution

**Date:** 2026-05-15
**Branch:** `rewrite/flow-matching-2026`
**Authors:** Steve Dobbs (collaborator: George Hilley)

## Executive summary

The first full training run of the post-rewrite rectified-flow model converged cleanly on Sherlock H100 and A100 GPUs (500 epochs, ~347K optimizer steps). Quantitative metrics improved substantially over the 200-step smoke test (SSIM 0.14 → 0.89, Slope KS 0.98 → 0.40, Coverage@90 0.75 → 0.82) and the model passed the mode-collapse diagnostic with a healthy median inter-realization std of 2.73 m. **However, qualitative inspection shows the model produces geomorphologically implausible "jigsaw" textures across the board** — diverse but uniformly wrong realizations even on smooth shelf terrain where the LR carries the correct signal. Diagnosis: distribution-coverage failure (85M params trained on ~1–2K effectively independent geomorphic regions). Next attempt: shrink the model to ~22M params and densify the training set to ~45K patches via reduced stride.

---

## 1. Setup

### Data

| Item | Value |
|---|---|
| HR resolution | 2 m (resampled from 1 m AUV/MBES) |
| LR resolution | 32 m |
| SR ratio | 16× |
| HR patch | 512 × 512 (1024 m × 1024 m) |
| LR patch (raw) | 32 × 32 (1024 m × 1024 m) |
| Stride | 256 HR pixels (50 % overlap) |
| Train / Val / Test patches | 11,125 / 2,864 / 1,256 |
| Test surveys (held out) | BankTopEastofCanyon3, HeadlessCanyon, LuciaChica_2007, Bathymetry_DrakesBay, Bathymetry_OffshoreRefugioBeach |
| LR grids used | MorroBay (EPSG:32610), NorCal-to-Triple-Junction (EPSG:26910), Santa Barbara (EPSG:26910) |
| HR surveys | 59 tifs across the California coast (Morro Bay, Monterey, NorCal shelf, SB channel) |
| Normalization | Per-patch z-score using HR statistics; raw LR + bicubic LR + HR all stored |
| Augmentation | Random flips + 90° rotations (already on) |
| Patches dir | `/scratch/groups/hilley/bathy_diffusion_2026/patches_512/` |

Patches were extracted by running `extract_paired_patches.py` three times sequentially (once per LR grid) into the same `out_dir` with first-run-wins behavior (`--clobber` flag, default skip-if-exists). The Morro Bay LR therefore provides authoritative conditioning for Morro Bay-area surveys (the survey footprint geographically dominates that LR's bounds).

### Model

`models/unet.py::BathyUNet`, 6 levels, `base_ch=64`, `ch_mult=(1,2,4,6,8,12)`. Levels 0–3 are CNN ResBlocks with AdaLN-Zero; levels 4–5 are DiT blocks with self-attention, cross-attention onto the raw LR, and an MLP. **Total: 85,646,913 parameters.**

### Training procedure

Standard rectified-flow with linear interpolant `x_t = (1-t) x_0 + t z`, logit-normal *t* sampling, CFG dropout 0.1, AdamW (lr 2e-4, β=(0.9, 0.999), wd 1e-4), cosine LR over `n_epochs × steps_per_epoch` to `eta_min=1e-6`, EMA decay 0.9999 with 1000-step warmup, gradient-clip 1.0, bf16 autocast, `torch.compile(mode='default')`. Batch size 16. Spectral loss off (`λ=0`).

### Two runs were submitted

| GPU | Job ID | Wall time | Steps | Final val_loss |
|---|---|---|---|---|
| H100 80 GB | 24866196 | 24 h 42 m | 347,500 (500 epochs) | 0.0201 |
| A100 40 GB | 24866198 | ~48 h | ~347 K (500 epochs) | 0.0203 |

Final checkpoints are at `/scratch/users/scdobbs/bathy_diffusion/training_{h100,a100}/checkpoints/ckpt_final.pt`. Throughput: **H100 ≈ 3.9 steps/sec, A100 ≈ 2.05 steps/sec** (~1.9× speedup).

The val_loss plateaued near 0.020 by **epoch ~200** and barely moved over the remaining 300 epochs.

---

## 2. Quantitative evaluation

H100 checkpoint evaluated on 100 stratified test patches (`evaluation/evaluate.py`, 10 realizations per patch, CFG scale 1.5, 50 Heun steps).

| Metric | Smoke (200 steps) | **Run 1 (500 epochs)** | Δ | Notes |
|---|---|---|---|---|
| PSNR (dB) | 12.87 | **15.32** | +2.45 | improvement |
| SSIM | 0.14 | **0.89** | +0.75 | large but dominated by low-freq slope structure |
| MAE (m) | 3.55 | **2.72** | −0.83 | |
| RMSE (m) | 4.32 | **3.37** | −0.95 | |
| CRPS (m) | 2.35 | **1.69** | −0.66 | proper scoring rule |
| Coverage @ 90 % | 0.749 | **0.815** | +0.066 | ideal 0.900 — slightly under-covered |
| Sharpness @ 80 % (m) | 8.68 | 8.08 | small | width of 80 % credible interval |
| Slope KS | 0.978 | **0.404** | −0.574 | major; 0 = perfect distributional match |

Calibration: `calibration.png` shows empirical < nominal across the board — the model's prediction intervals are tighter than they should be (over-confident), but the gap shrank substantially from the smoke test.

### Sample-diversity / mode-collapse diagnostic (Phase 7)

Stratified 4 patches × 5 surveys × 50 realizations each, per-pixel inter-realization std:

| Survey | gt_range (m) | median std (m) | std / range |
|---|---|---|---|
| Bathymetry_DrakesBay (smooth shelf) | 3.3–10.5 | 0.48–1.60 | ~15 % |
| LuciaChica_2007 | 10.3–14.9 | 1.90–2.97 | ~20 % |
| BankTopEastofCanyon3 | 7.6–17.6 | 1.71–4.17 | ~24 % |
| Bathymetry_OffshoreRefugioBeach | 6.4–36.1 | 1.05–7.64 | ~20 % |
| HeadlessCanyon | 108–217 | 16.4–51.0 | ~20–25 % |

**Median-of-medians inter-realization std = 2.73 m** — well above the 0.1 m mode-collapse threshold. The std scales appropriately with local terrain complexity (smooth shelves → low std, canyon walls → high std).

---

## 3. Qualitative findings (the important part)

Visual inspection of generated SR realizations, both as stitched regional outputs and as 8-realization side-by-side comparisons of single 1024 m × 1024 m patches, shows **the model is producing geomorphologically implausible outputs almost everywhere**.

See `realization_comparison_20260515_134907/` for the side-by-side figures.

### What the visuals show

- **Drakes Bay (smooth, 6.8 m range)**: HR is a gentle gradient; the model's 8 realizations are wildly different jigsaw blob patterns spanning the entire 6.8 m range. Per-pixel std max 2.9 m — **43 % of the dynamic range** on terrain that should be nearly featureless.
- **HeadlessCanyon (140 m range)**: HR shows a coherent canyon edge / bench; the model places a different "island" or ridge fragment in a different location in each realization. The realizations are mutually incompatible.
- **BankTopEastofCanyon3 (16 m range)**: HR is a relatively smooth bank top; the model produces streaky, white-peaked, angular features bearing no resemblance to the HR.

The artifacts are **uniform across the patch** (not localized to high-relief or low-relief regions) and **intrinsic to single patches** (not tile-stitching artifacts — these are 1024 m × 1024 m single-tile outputs).

### Diagnosis

This is **not**:

- Mode collapse — realizations are diverse, too diverse in fact.
- Tile-stitching artifacts — these appear in single-patch outputs.
- Under-training — val_loss plateaued at epoch 200 of 500.

This **is** a *distribution-coverage* failure:

- The model has learned **plausible local pixel statistics** (terrain-like texture).
- The model has **not learned the morphological rules** that connect those textures into valid global structure (drainage networks, curvature constraints, regional coherence, "rivers flow downhill").
- The realizations disagree everywhere rather than at known high-uncertainty features, because the model has no consistent prior to anchor any region to.

Quantitatively, the dataset has ~11 K patches but only **~1–2 K effectively independent regions** (the 50 %-overlap sliding window inflates the count). With 85 M parameters, that's roughly 40 K params per independent region — far beyond what generative models normally need.

### Why the metrics looked OK despite the visual failure

- **SSIM = 0.89 is dominated by the low-frequency slope structure**, which is already in the bicubic LR. It does not stringently test whether the high-frequency texture is in the right place.
- **Slope KS = 0.40 measures distribution overlap**, not spatial correspondence. The model produces approximately correct *amounts* of roughness, but not in the right locations.
- **Val_loss is the flow-matching velocity-MSE**, a noisy proxy that says nothing about whether the model's samples look like real seafloor.

This is a textbook case of "good statistics, wrong content."

---

## 4. What worked

1. **End-to-end pipeline**: preprocessing → multi-LR patch extraction → training → tiled inference → uncertainty maps → evaluation → diversity check, all functional and reproducible from `sherlock/` sbatch files.
2. **Hilley's rectified-flow code**: trains stably for 500 epochs with no instabilities; bf16 + `torch.compile` work cleanly with the venv'd Python 3.12 + Triton.
3. **DDP scaffolding**: same script runs at WORLD_SIZE=1 and 2+ without modification (we only tested WORLD_SIZE=1).
4. **EMA implementation**: dotted-key checkpoint format (`ema.model`, `ema.null_lr`) works once the loader is taught about it.
5. **Multi-LR preprocessing with first-run-wins** preserves regionally-appropriate LR conditioning for each survey.
6. **No mode collapse**: realizations are genuinely diverse, with appropriate scaling by terrain complexity.
7. **A100 → H100 portability**: both GPUs trained the same code with no changes; only throughput differs.

## 5. What didn't work / unresolved

1. **Visual fidelity**: jigsaw / patchwork artifacts everywhere, even on smooth terrain.
2. **Coverage is under-calibrated** (0.82 vs target 0.90) — model is over-confident.
3. **Santa Barbara region produced wild uncertainty** (median std 2.7 m, max 10,798 m). Likely a residual CRS / datum mismatch — SB LR is EPSG:26910 but the smoke bbox is numerically valid in either zone, so reads succeed but at the wrong spatial location.
4. **The flow-matching val_loss plateaued at epoch ~200** but downstream metrics kept improving for hundreds more epochs. Val_loss is not a useful stopping signal.
5. **Setup friction**: `setup.sh` initially used `--user` pip installs that conflicted with rasterio's bundled libgdal; the gdal module conflicts with PyTorch's NCCL; `torchrun` spawns workers under the module's Python which doesn't see the venv's Triton. All resolved (use a project venv + `python3 -m torch.distributed.run`), but worth noting these are durable Sherlock-specific gotchas.
6. **`np.nanpercentile` in `evaluation/metrics.py::coverage()`** was 50–100× slower than `np.quantile`; one 12-hour eval job timed out 2.5 hours past `metrics.json` finishing because of it. Fixed.

## 6. Other observations worth noting

- **Throughput is memory-bound, not compute-bound**: H100 was only ~1.9× faster than A100 despite ~3× more raw TFLOPs. The activation memory at 512 × 512 × 85 M params dominates. Flash-attn would help but didn't install cleanly on the current Sherlock pytorch/2.4 build — we fall back to SDPA's math backend.
- **Patch normalization is suspicious**: per-patch z-score destroys absolute depth and regional setting. The model sees a 1024 m × 1024 m HR target normalized to its own local distribution; the only regional information it has is the 32 × 32 raw LR, which is ~1 km² of context. This is plausibly part of why the model doesn't know whether it's generating "shelf" or "canyon."

---

## 7. Changes for Run 2 (in progress)

Currently submitting:

| Change | Hypothesis | Status |
|---|---|---|
| `base_ch = 32` (~22 M params, ~4× smaller) | 85 M is way over-parameterized for ~1–2 K effective regions | sbatch ready: `sherlock/train_small.sbatch` |
| `stride = 128` (~45 K train patches, ~4× more) | Denser sampling gives more gradient steps per epoch, more position-views per feature | preprocessing submitted: `sherlock/preprocess_s128.sbatch` (job 25101745) |
| `n_epochs = 125` (vs 500) | Keep total optimizer steps ≈ 350 K (same as Run 1) so we compare like-for-like | in sbatch |
| Output dir: `patches_512_s128/` and `training_small_s128/` | Preserve Run 1 artifacts as a baseline | in sbatch |

Everything else (architecture topology, optimizer, loss, augmentation, normalization) is unchanged. **This is a controlled experiment isolating model capacity × data density**. If it improves visual fidelity, we know the problem was an over-parameterized model relative to the effective dataset. If it doesn't, the problem is something else (architecture / loss / data variety) and we can rule out capacity-to-data ratio.

## 8. Future interventions to consider (not yet attempted)

Ordered by expected bang-for-buck:

1. **Synthetic bathymetry pre-training** (highest potential payoff). Use a landscape-evolution model (FastScape, Badlands, LEM-Earth) to generate unlimited geomorphically-valid HR examples; pretrain on synthetic; fine-tune on real surveys. Directly addresses the global-rules problem.
2. **Wider LR context window**. Train with 256 × 256 raw LR (≈ 8 km × 8 km) instead of 32 × 32 (1 km × 1 km). Gives the model real regional setting — knows whether it's on a shelf vs in a canyon vs over a pockmark field.
3. **Geomorphic loss term**. Penalize divergence between predicted and true slope-distribution / curvature-distribution / drainage-connectivity. Forces the model to produce physically realistic outputs at the cost of some MSE.
4. **Multi-scale curriculum training**. Pretrain at 4× SR (32 m → 8 m), fine-tune at 8×, then 16×. Builds up morphological coherence at progressively finer scales.
5. **Resolve the Santa Barbara CRS issue**. Either reproject the SB LR to EPSG:32610 to match the rest of the data, or fix the inference-bbox transform in the per-LR loop.
6. **Per-patch normalization alternatives**. Try global z-score (one mean/std across the entire dataset) so the model retains some sense of absolute depth — at the cost of potentially blowing up the dynamic range for outlier surveys.

## 9. Open scientific questions

- **Is the model actually super-resolving, or doing principled interpolation with hallucinated texture?** A clean test would compare SR FFT power spectra to bicubic baseline and HR ground truth on the same regions. We did something equivalent on the old DDPM branch but haven't redone it here on the flow-matching model.
- **Does the model generalize across geomorphic settings, or only to surveys near its training distribution?** Test-set evaluation is averaged across all 5 held-out surveys; per-survey breakdown would tell us if (say) the canyon survey is much worse than the shelf surveys.
- **Is the per-patch normalization actually load-bearing?** Worth an ablation against a no-normalization run.

---

## 10. File pointers (for reproducibility)

| Thing | Path |
|---|---|
| Training script | `training/train.py` |
| Inference script | `inference/generate_realizations.py` |
| Evaluation script | `evaluation/evaluate.py` |
| Preprocessing | `preprocessing/extract_paired_patches.py` (with `--clobber` flag) |
| Method document | `METHOD.md` |
| Shared training patches | `/scratch/groups/hilley/bathy_diffusion_2026/patches_512/` |
| H100 final checkpoint | `/scratch/users/scdobbs/bathy_diffusion/training_h100/checkpoints/ckpt_final.pt` |
| Run 1 eval outputs | `/scratch/users/scdobbs/bathy_diffusion/infer_eval_h100_20260514_144500/` |
| Realization comparison figures | `/scratch/users/scdobbs/bathy_diffusion/realization_comparison_20260515_134907/` |
| Run 2 sbatch (in progress) | `sherlock/preprocess_s128.sbatch`, `sherlock/train_small.sbatch` |
