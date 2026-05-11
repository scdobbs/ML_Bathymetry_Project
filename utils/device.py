"""Device selection helper."""

import torch


def get_device() -> str:
    """Pick the best available compute device: CUDA → MPS → CPU."""
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'
