"""
Shared utilities for oil spill detection project.

Consolidates duplicated helpers (silent_tf_import, colored_print, etc.) into one module.
"""

import os
import sys
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


def silent_tf_import():
    """Import TensorFlow while suppressing noisy startup warnings."""
    orig_stderr_fd = sys.stderr.fileno()
    saved_stderr_fd = os.dup(orig_stderr_fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, orig_stderr_fd)
    os.close(devnull_fd)

    import tensorflow as tf

    os.dup2(saved_stderr_fd, orig_stderr_fd)
    os.close(saved_stderr_fd)
    return tf


def set_reproducibility(seed: int = 42):
    """Set all random seeds for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    random.seed(seed)
    np.random.seed(seed)

    import tensorflow as tf
    tf.random.set_seed(seed)


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
