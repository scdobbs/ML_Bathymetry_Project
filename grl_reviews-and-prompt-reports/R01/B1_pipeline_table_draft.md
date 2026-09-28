## Phase B1 — Pipeline table (code reading; every row cites file:line)

Files: `P` = `preprocessing/extract_colocated_patches.py` (native branch, `process_pair_native`, the writer of
`patches_native_phase2`; invoked by `sherlock/preprocess_native.sbatch` with `--native --native_hr_px 256
--context_factor 4 --lr_context_px 64 --train_stride_frac 0.25 --eval_stride_frac 0.5 --min_valid 0.50`,
`--flat_threshold_m` default 0.05 at P:371). `H` = `models/het_production_v1/step0_bakeoff/het_louo.py`
(training and `eval_fold`; the reproduction `dir55/repro/het_repro.py` imports it unchanged). `U` =
`preprocessing/data_utils.py`.

| Item | Training | Evaluation (`eval_fold`) | Reference |
|---|---|---|---|
| Input channel 1 | `lr_up`: bicubic upsample (scipy `zoom`, order 3) of the ship grid, resampled by area-average onto the patch's LR cell grid | same array from the same npz | P:293–294, P:314; U:42–49; H:139 (train), H:200 (eval) |
| Input channel 2+ | none (`in_ch=1`); `lr_context` is written to the npz but never read by the model | same | H:87, H:139–147 |
| Conditioning vector | `scale = [log2(lr_native), log2(hr_res)]`, FiLM at every block (LR/HR grid spacings from the manifest; not HR content) | same | P:256, H:147, H:58–69 |
| Normalization of the input | `lr_up_norm = (bicubic(lr_filled) − mean) / std` where `mean = nanmean(HR valid)`, `std = nanstd(HR valid)` (1.0 if < 1e-6) | same, the stored array | **P:305–314** |
| Normalization of the target | `hr_norm = (where(hr_valid, hr, mean) − mean)/std`, same HR statistics | same | P:311 |
| Fill of invalid pixels before the network | HR: patch HR mean (→ 0 normalized). LR: patch **HR** mean (→ (mean − mean)/std = 0 normalized) before bicubic | same | P:311–314 |
| Invalid pixels in the loss | excluded: `valid = hr_valid × nearest-upsampled lr_valid`, masked NLL | metrics use the same mask | H:143–146, H:122–125, H:209 |
| Conv padding | zero padding, `padding=1`, default `padding_mode='zeros'` on every 3×3 conv (in_conv, both convs per block, stride-2 downs, ups) | same | H:74–75, H:92, H:96, H:100 |
| σ parameterization | `log_var` head, clamped to [−10, 6] → σ ∈ [0.0067, 20.09] | same | H:88, H:118 |
| Mean parameterization / output activation | `mean = lr_up + head_mean(h)` (residual on the input); no activation on either head; heads zero-initialised | same | H:117–118, H:104–105 |
| Patch size / stride | 256 × 256 HR px; training stride 0.25 × 256 = 64 px, eval splits 0.5 × 256 = 128 px at preprocessing; the model sees whole patches | whole 256-px patches, no overlap, no cropping | P (sbatch args), H:193–203 |
| Selection | per-pair cap 2,500 files, evenly spaced in sorted order; batch 32, 8,000 steps, AdamW 2e-4, cosine, grad-clip 1.0, bf16 autocast | held-out cap 200 per pair, same spacing | H:151–162, H:165–185, H:171–172, `dir55/repro/het_folds/config.json` |
| Augmentation | none (`DS.__getitem__` loads arrays only) | none | H:130–148 |
| Preprocessing rejections | joint-valid fraction < 0.50; HR-valid fraction < 0.50; HR std < 0.05 m | same files | P:279, P:302, P:305–307 |

**Does any HR-derived quantity reach the network's inputs at evaluation? YES (CONFIRMED).**
The only input channel, `lr_up`, is normalized by the mean and standard deviation of the AUV (HR) patch
over its valid pixels (P:305–314). Every evaluation patch therefore enters the network as
(ship grid − AUV patch mean) / AUV patch std, and the target enters as (AUV − AUV mean)/AUV std, so the
target has mean 0 and std 1 by construction on every patch. Two consequences follow directly from the code:

1. The vertical ship-minus-AUV offset is visible in the input as its spatial mean (the HR mean is zero, so
   mean(`lr_up`) over valid pixels equals the negative offset in normalized units). A predictor that
   subtracts its input's mean removes the offset without any seafloor information. This is AUV information
   at the applied site.
2. The residual variance the network must explain is 1 − var(`lr_up`) plus offset², a function of the
   normalized input alone, so the magnitude of the "total error" per patch is partly encoded in the input
   scale. σ can learn this from the input without reading seafloor content.

H-norm follows from the same lines: on a patch of HR std 0.05 to 0.5 m (the preprocessing floor is 0.05 m,
P:371), a meter-scale offset becomes a normalized input of 2 to 20, and invalid LR cells are filled with the
HR mean (0 normalized) rather than with the LR level, creating a step of the full offset at every coverage
edge before the bicubic upsample (P:312–314). Zero padding at the tile border (H:74–75) adds a second step
from the input level to 0. Whether these steps drive the observed blow-ups is a Phase A/B3 measurement
(A4, A5, B3), not settled by code reading.

The `scale` vector carries only the two grid spacings; it is not HR content.

Per §0.1 and B1, this is an escalation: Phase B stops here. B2 to B5 and Phase C are HELD for Steve.
