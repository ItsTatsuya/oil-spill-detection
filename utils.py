"""
Shared utilities for oil spill detection project.

Consolidates helpers (set_reproducibility, colored_print, etc.) into one module.
All TensorFlow/Keras dependencies removed — now pure PyTorch.
"""

import os
import random
import logging

import numpy as np

logger = logging.getLogger('oil_spill')


def setup_logging(level=logging.INFO):
    """Configure logging for the entire project."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    return logging.getLogger('oil_spill')


def set_reproducibility(seed: int = 42):
    """Set all random seeds for reproducibility (Python, NumPy, PyTorch)."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic CUDNN ops (slight perf hit, improves reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Terminal colours
# ---------------------------------------------------------------------------
class TermColors:
    """ANSI colour codes for terminal highlighting."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def colored_print(text, color=TermColors.BLUE, bold=False):
    """Print coloured text to terminal."""
    prefix = TermColors.BOLD if bold else ""
    print(f"{prefix}{color}{text}{TermColors.ENDC}")
