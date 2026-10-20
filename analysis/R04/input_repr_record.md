# §7 Input-representation record (R02 clean pipeline; factual, no proposal)

| item | as implemented |
|---|---|
| lr_representation | tile_ship_side: LR area-averaged (Resampling.average) onto lr_px x lr_px cells over the exact 256-px HR window footprint, lr_px = max(2, round(256 * hr_res / lr_native)); validity nearest; invalid cells <- mu_lr; bicubic (scipy zoom order 3) to 256 x 256 HR px (build_inputs) |
| normalization | per window, ship-side statistics only: mu_lr = mean(LR | valid), s_lr = max(std(LR | valid), 0.5 m); x = (bicubic(filled LR) - mu_lr) / s_lr; target y = (HR - delta - mu_lr) / s_lr with delta = median(HR - bicubic) over joint px (label only) |
| channels | in_ch = 2: channel 1 = x (normalized bicubic ship surface), channel 2 = lr_valid_hr (LR validity nearest-upsampled to 256 px, floor(i*L/H)); the model output is (mean, log-variance) per pixel (HetUNetClean, residual mean) |
| augmentation | none (het_clean.py docstring: 'no augmentation'; recipe = het_louo.py) |
| scale_conditioning | scale = [log2(lr_native), log2(hr_res)] from the manifest, fed through FiLM (2-vector) in every block; lr_native and hr_res enter only this way; k does not enter the model (it sets the band edges of the evaluation only); depth enters only through mu_lr normalization (removed from x) - not as an input |
| window | 256 x 256 HR px window; training stride 64 px (train pairs) / 128 px (val-test pairs) in R02; R04 tiles_v1 uses stride 128 everywhere |

Code evidence (regex hits in extract_clean_patches.py + het_clean.py): {'area_average': True, 'bicubic_zoom': True, 's_floor': True, 'in_ch2': True, 'no_aug': True, 'film': True, 'scale_log2': True}

| unit | pairs | LR cell (m) | lr_native (m) | LR cells per 256-px window | HR px per LR cell (min-max) | k |
|---|---|---|---|---|---|---|
| lu00 | 1 | 20.22 | 20.30 | 12.6 | 20.3-20.3 | 8 |
| lu02 | 1 | 38.68 | 38.00 | 14.7 | 17.4-17.4 | 4 |
| lu04 | 1 | 23.23 | 23.20 | 11.1 | 23.2-23.2 | 8 |
| lu05 | 3 | 44.18/44.19/43.63 | 45.80/44.22/44.50 | 5.6/5.8/28.7 | 8.9-45.8 | 16/16/2 |
| lu06 | 1 | 50.00 | 50.00 | 15.4 | 16.7-16.7 | 4 |
| lu07 | 2 | 48.00/86.94 | 50.40/86.94 | 10.5/6.0 | 24.4-42.3 | 2/16 |
| lu08 | 1 | 65.71 | 66.10 | 4.1 | 62.6-62.6 | 16 |
| lu09 | 1 | 28.38 | 30.00 | 16.1 | 15.9-15.9 | 4 |
| lu10 | 1 | 32.03 | 32.10 | 8.4 | 30.5-30.5 | 8 |
| lu11 | 1 | 15.19 | 15.20 | 17.8 | 14.4-14.4 | 4 |
| lu12 | 18 | 10.00/10.00/10.00/10.00/10.00/10.00/10.00/10.00/10.00/10.00/10.00/10.00/10.00/10.00/10.00/10.00/10.00/10.00 | 10.00/10.00/10.00/10.00/10.00/10.00/10.00/10.00/10.00/10.00/10.00/10.00/10.00/10.00/10.00/10.00/10.00/10.00 | 51.2/51.2/51.2/51.2/51.2/51.2/51.2/51.2/51.2/51.2/51.2/51.2/51.2/51.2/51.2/51.2/51.2/51.2 | 5.0-5.0 | 2/2/2/2/2/2/2/2/2/2/2/2/2/2/2/2/2/2 |
| lu13 | 2 | 57.29/57.23 | 57.40/57.87 | 4.5/4.4 | 57.4-57.9 | 16/16 |
| lu14 | 1 | 29.36 | 29.40 | 15.4 | 16.6-16.6 | 4 |
| lu15 | 2 | 51.20/36.84 | 51.30/36.90 | 5.0/6.9 | 36.9-51.3 | 16/8 |
| lu16 | 1 | 79.28 | 79.40 | 4.0 | 64.5-64.5 | 16 |
| nu00 | 1 | 65.91 | 65.99 | 3.9 | 66.0-66.0 | 16 |
| vu02 | 1 | 16.37 | 16.37 | 15.6 | 16.4-16.4 | 4 |
