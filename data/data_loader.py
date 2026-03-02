"""
Data loading pipeline for oil spill segmentation — PyTorch implementation.

Key design:
- OilSpillDataset: torch.utils.data.Dataset subclass
- Training: load at original resolution → random patch crop → augmentation in __getitem__
- Val / test: resize to IMG_SIZE, no augmentation
- Class weights computed from actual pixel distribution (inverse-frequency), cached to disk
- DataLoader with pin_memory=True and num_workers for fast GPU transfers
"""

import os
import glob
import random
import logging

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from data.augmentation import augment_single_sample

logger = logging.getLogger('oil_spill')


def _worker_init_fn(worker_id: int) -> None:  # noqa: ARG001
    """Seed each DataLoader worker independently to prevent identical augmentation."""
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)

IMG_SIZE   = 384
NUM_CLASSES = 5


# ---------------------------------------------------------------------------
# Class weight computation (inverse-frequency from actual labels)
# ---------------------------------------------------------------------------
def analyze_class_distribution(label_paths, num_classes=NUM_CLASSES, cache_path=None):
    """Compute inverse-frequency class weights from actual label statistics."""
    if cache_path and os.path.exists(cache_path):
        weights = np.load(cache_path)
        logger.info("Loaded cached class weights from %s", cache_path)
        return {i: float(weights[i]) for i in range(num_classes)}

    logger.info("Computing class weights from %d label files …", len(label_paths))
    counts = np.zeros(num_classes, dtype=np.int64)
    for path in label_paths:
        label = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if label is None:
            continue
        label = np.clip(label, 0, num_classes - 1)
        for c in range(num_classes):
            counts[c] += int(np.sum(label == c))

    freq = counts / (counts.sum() + 1e-12)
    raw_weights = 1.0 / (freq + 1e-6)
    raw_weights /= raw_weights.sum()

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
        np.save(cache_path, raw_weights)
        logger.info("Saved class weights to %s", cache_path)

    print(f"Computed class weights: {dict(enumerate(np.round(raw_weights, 4)))}")
    return {i: float(raw_weights[i]) for i in range(num_classes)}


# ---------------------------------------------------------------------------
# Patch sampling (pure NumPy, called from Dataset.__getitem__)
# ---------------------------------------------------------------------------
def sample_patches(image: np.ndarray, label: np.ndarray, patch_size: int = IMG_SIZE,
                   rng=None):
    """
    Random crop of *patch_size* × *patch_size* from (possibly larger) image.

    With 50% probability the crop is centred on an Oil-Spill (class 1) or
    Ship (class 3) pixel to up-sample rare classes.

    Parameters
    ----------
    image : [H, W, C]  float32  (or [H, W] for grayscale)
    label : [H, W]     uint8
    rng   : np.random.Generator  (pass for reproducibility; uses global state otherwise)

    Returns
    -------
    img_crop   : [H', W', C] (or [H', W'])
    label_crop : [H', W']
    """
    if rng is None:
        rng = np.random.default_rng()

    h, w = label.shape[:2]

    # Pad if smaller than patch
    pad_h = max(patch_size - h, 0)
    pad_w = max(patch_size - w, 0)
    if pad_h > 0 or pad_w > 0:
        if image.ndim == 3:
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)))
        else:
            image = np.pad(image, ((0, pad_h), (0, pad_w)))
        label = np.pad(label, ((0, pad_h), (0, pad_w)))
        h, w = label.shape[:2]

    def random_crop():
        top  = rng.integers(0, h - patch_size + 1)
        left = rng.integers(0, w - patch_size + 1)
        return top, left

    def class_centred_crop(cls):
        ys, xs = np.where(label == cls)
        if len(ys) == 0:
            return random_crop()
        idx  = rng.integers(0, len(ys))
        cy, cx = int(ys[idx]), int(xs[idx])
        half = patch_size // 2
        top  = int(np.clip(cy - half, 0, h - patch_size))
        left = int(np.clip(cx - half, 0, w - patch_size))
        return top, left

    if rng.random() < 0.5:
        has_ship = np.any(label == 3)
        has_oil  = np.any(label == 1)
        if has_ship:
            top, left = class_centred_crop(3)
        elif has_oil:
            top, left = class_centred_crop(1)
        else:
            top, left = random_crop()
    else:
        top, left = random_crop()

    if image.ndim == 3:
        img_crop = image[top:top + patch_size, left:left + patch_size, :]
    else:
        img_crop = image[top:top + patch_size, left:left + patch_size]
    lbl_crop = label[top:top + patch_size, left:left + patch_size]
    return img_crop, lbl_crop


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class OilSpillDataset(Dataset):
    """
    PyTorch Dataset for oil spill SAR segmentation.

    Parameters
    ----------
    image_files  : list of str
    label_files  : list of str
    split        : 'train' | 'val' | 'test'
    patch_size   : int   (training: random crop to this size; val/test: resize)
    augment      : bool  (apply stochastic augmentation in __getitem__)
    """

    def __init__(self, image_files, label_files, split='train',
                 patch_size=IMG_SIZE, augment=True):
        if len(image_files) != len(label_files):
            raise ValueError(
                f"Image/label count mismatch: {len(image_files)} vs {len(label_files)}"
            )
        self.image_files = image_files
        self.label_files = label_files
        self.split       = split
        self.patch_size  = patch_size
        self.augment     = augment and (split == 'train')
        self._rng        = np.random.default_rng()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # ---- load image (grayscale SAR) ---
        img_path = self.image_files[idx]
        lbl_path = self.label_files[idx]

        img_bgr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_bgr is None:
            img_bgr = np.zeros((self.patch_size, self.patch_size), dtype=np.uint8)

        lbl = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)
        if lbl is None:
            lbl = np.zeros((self.patch_size, self.patch_size), dtype=np.uint8)

        img = img_bgr.astype(np.float32) / 255.0           # [H, W]  float32
        lbl = np.clip(lbl, 0, NUM_CLASSES - 1).astype(np.int64)

        if self.split == 'train':
            # ---------- random patch crop at original resolution ----------
            img, lbl = sample_patches(img, lbl, self.patch_size, self._rng)
        else:
            # ---------- resize to fixed size for val / test ---------------
            img = cv2.resize(img, (self.patch_size, self.patch_size),
                             interpolation=cv2.INTER_LINEAR)
            lbl = cv2.resize(lbl.astype(np.uint8), (self.patch_size, self.patch_size),
                             interpolation=cv2.INTER_NEAREST).astype(np.int64)

        # Convert to tensors: image [C=1, H, W], label [H, W]
        img_t = torch.from_numpy(img[None]).float()         # [1, H, W]
        lbl_t = torch.from_numpy(lbl).long()                # [H, W]

        # ---- augmentation (training only) --------------------------------
        if self.augment:
            img_t, lbl_t = augment_single_sample(img_t, lbl_t)

        return img_t, lbl_t


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------
def load_dataset(
    data_dir: str = 'dataset',
    split: str = 'train',
    batch_size: int = 16,
    val_split: float = 0.15,
    class_weights_cache: str = 'dataset/class_weights.npy',
    num_workers: int = 4,
    patch_size: int = IMG_SIZE,
):
    """
    Build a DataLoader for the given split.

    Returns
    -------
    loader         : DataLoader
    class_weights   : dict {class_idx: weight}  or None (non-training splits)
    num_batches     : int
    """
    if split not in ('train', 'val', 'test'):
        raise ValueError(f"Invalid split '{split}'. Choose from 'train', 'val', 'test'.")

    base_dir  = os.path.join(data_dir, 'train' if split in ('train', 'val') else 'test')
    image_dir = os.path.join(base_dir, 'images')
    label_dir = os.path.join(base_dir, 'labels_1D')

    image_files = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
    label_files = sorted(glob.glob(os.path.join(label_dir, '*.png')))

    if len(image_files) != len(label_files):
        raise ValueError(
            f"Image-label mismatch in {base_dir}: "
            f"{len(image_files)} images vs {len(label_files)} labels"
        )

    # ---- deterministic train / val split ---------------------------------
    if split in ('train', 'val'):
        n_total = len(image_files)
        n_val   = max(1, int(val_split * n_total))
        if split == 'val':
            image_files = image_files[-n_val:]
            label_files = label_files[-n_val:]
        else:
            image_files = image_files[:-n_val]
            label_files = label_files[:-n_val]

    # ---- class weights (training only) -----------------------------------
    class_weights = None
    if split == 'train':
        class_weights = analyze_class_distribution(
            label_files, num_classes=NUM_CLASSES,
            cache_path=class_weights_cache,
        )

    # ---- dataset & loader ------------------------------------------------
    augment = (split == 'train')
    dataset = OilSpillDataset(
        image_files, label_files,
        split=split,
        patch_size=patch_size,
        augment=augment,
    )

    # On Windows: num_workers > 0 requires if __name__ == '__main__' guard in script.
    # Use persistent_workers only when num_workers > 0.
    effective_workers = num_workers
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=effective_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=(split == 'train'),
        persistent_workers=(effective_workers > 0),
        worker_init_fn=_worker_init_fn if effective_workers > 0 else None,
    )

    num_batches = len(loader)
    print(
        f"Loaded {split} dataset: {len(dataset)} samples, "
        f"{num_batches} batches (batch_size={batch_size})"
    )
    return loader, class_weights, num_batches
