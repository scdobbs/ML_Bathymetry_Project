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
# Conda's `python` on $PATH can shadow the module-provided Python 3.12,
# and the module only ships `python3` (no `python` symlink). Deactivate any
# active conda environments before loading modules.
type conda >/dev/null 2>&1 && {
    conda deactivate >/dev/null 2>&1 || true
    conda deactivate >/dev/null 2>&1 || true
}
unset PYTHONPATH

module load python/3.12.1 py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312

# Create / activate a project-local venv. The cluster's --user site-packages
# location is shared across all Python environments and breaks rasterio
# (the wheel's bundled libgdal conflicts with anything else loaded). A
# dedicated venv keeps the dependency tree self-contained.
VENV_DIR="${VENV_DIR:-$HOME/bathy_diffusion_env}"
if [ ! -x "${VENV_DIR}/bin/python3" ]; then
    echo "Creating venv at ${VENV_DIR} (one-time)..."
    python3 -m venv --system-site-packages "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"
PY=python3
echo "Python : $(command -v $PY) ($($PY --version 2>&1))"
$PY -c 'import sys; assert sys.version_info[:2] == (3, 12), f"need Python 3.12, got {sys.version}"; print("Python version check OK")'
echo "Torch  : $($PY -c 'import torch; print(torch.__version__)')"
echo "CUDA   : $($PY -c 'import torch; print(torch.version.cuda)')"
echo "Driver : $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo '(no GPU on login node — fine)')"
echo

echo "=== [3/6] Runtime deps (into venv) ==="
$PY -m pip install --upgrade pip setuptools wheel
$PY -m pip install \
    "einops>=0.8" \
    "rasterio>=1.3" \
    "tensorboard>=2.16" \
    "tqdm>=4.66" \
    "triton==3.0.0"
# scipy/matplotlib/scikit-image come from the module's system-site-packages
# (the venv was created with --system-site-packages). Install only if missing.
$PY -m pip install "scipy" "matplotlib" "scikit-image" || true
echo

echo "=== [4/6] flash-attn 2.6.3 (CUDA 12.x, --no-build-isolation) ==="
# --no-build-isolation lets the wheel build against the module's torch
# rather than pip resolving its own. Skip if already importable.
if $PY -c "import flash_attn; print(flash_attn.__version__)" 2>/dev/null; then
    echo "flash-attn already installed."
else
    $PY -m pip install --no-build-isolation "flash-attn==2.6.3" || {
        echo "WARN: flash-attn install failed. The model still works — SDPA falls"
        echo "      back to the math backend on H100 (slower but correct)."
    }
fi
echo

echo "=== [5/6] Editable install of the project ==="
$PY -m pip install -e .
echo

echo "=== [6/6] Import sanity check ==="
$PY - <<'PY'
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
