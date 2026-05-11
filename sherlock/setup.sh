#!/usr/bin/env bash
# One-time Sherlock setup for the bathy-diffusion project.
#
# Run on a Sherlock login node:
#     cd ~                                          # or wherever
#     git clone <repo-url> GAN_Bathymetry_Project   # if not already cloned
#     bash GAN_Bathymetry_Project/sherlock/setup.sh
#
# What it does:
#   1. Checks out the rewrite branch.
#   2. Loads the cluster's py-pytorch module (Python 3.12 + torch 2.4.1 + CUDA 12.1).
#   3. pip --user installs the project's runtime deps (einops, rasterio, etc.).
#   4. pip --user installs flash-attn 2.6.3 with --no-build-isolation
#      (re-uses the module's torch instead of resolving its own).
#   5. pip --user installs the project itself in editable mode.
#   6. Runs an import sanity check and prints a "ready" line.
#
# Idempotent: safe to re-run after a code update.

set -e
set -o pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/GAN_Bathymetry_Project}"
BRANCH="${BRANCH:-rewrite/flow-matching-2026}"

if [ ! -d "$PROJECT_DIR" ]; then
    echo "ERROR: $PROJECT_DIR does not exist."
    echo "Clone the repo first: git clone <repo-url> $PROJECT_DIR"
    exit 1
fi
cd "$PROJECT_DIR"

echo "=== [1/6] Branch & status ==="
git fetch --quiet
git checkout "$BRANCH"
git pull --ff-only
git --no-pager log --oneline -3
echo

echo "=== [2/6] Module load ==="
module load py-pytorch/2.4.1_py312
echo "Python : $(which python) ($(python --version 2>&1))"
echo "Torch  : $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA   : $(python -c 'import torch; print(torch.version.cuda)')"
echo "Driver : $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo '(no GPU on login node — fine)')"
echo

echo "=== [3/6] Runtime deps (--user) ==="
python -m pip install --user --upgrade pip setuptools wheel
python -m pip install --user \
    "einops>=0.8" \
    "rasterio>=1.3" \
    "tensorboard>=2.16" \
    "tqdm>=4.66"
# scipy, numpy, matplotlib, scikit-image typically ship with the module.
# Install only if missing.
python -m pip install --user \
    "scipy" "matplotlib" "scikit-image" || true
echo

echo "=== [4/6] flash-attn 2.6.3 (CUDA 12.x, --no-build-isolation) ==="
# --no-build-isolation lets the wheel build against the module's torch
# rather than pip resolving its own. Skip if already importable.
if python -c "import flash_attn; print(flash_attn.__version__)" 2>/dev/null; then
    echo "flash-attn already installed."
else
    python -m pip install --user --no-build-isolation "flash-attn==2.6.3" || {
        echo "WARN: flash-attn install failed. The model still works — SDPA falls"
        echo "      back to the math backend on H100 (slower but correct)."
    }
fi
echo

echo "=== [5/6] Editable install of the project ==="
python -m pip install --user -e .
echo

echo "=== [6/6] Import sanity check ==="
python - <<'PY'
import torch
print("torch          :", torch.__version__, "| CUDA available:", torch.cuda.is_available())
import einops; print("einops         :", einops.__version__)
import rasterio; print("rasterio       :", rasterio.__version__)
import scipy; print("scipy          :", scipy.__version__)
try:
    import flash_attn; print("flash-attn     :", flash_attn.__version__)
except Exception as e:
    print("flash-attn     : NOT IMPORTABLE —", e)

# Project imports (require pip install -e .)
from models.unet import BathyUNet
from models.diffusion import RectifiedFlow
from models.losses import FlowMatchingLoss
from preprocessing.data_utils import BathymetryPatchDataset
from training.config import Config
from utils.device import get_device
from utils.distributed import setup_distributed
print("project import : OK")

cfg = Config()
print("Config         : patch_size=%d sr_ratio=%d lr_patch_size=%d precision=%s compile=%s"
      % (cfg.patch_size, cfg.sr_ratio, cfg.lr_patch_size, cfg.precision, cfg.compile_model))
PY

echo
echo "===================================================="
echo "Setup complete."
echo
echo "Next steps:"
echo "  1. (one-time) Preprocess patches:"
echo "       sbatch sherlock/preprocess.sbatch"
echo
echo "  2. End-to-end smoke test:"
echo "       sbatch sherlock/e2e.sbatch"
echo
echo "  3. Production training:"
echo "       sbatch sherlock/train.sbatch"
echo "===================================================="
