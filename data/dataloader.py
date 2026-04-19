import logging
import math
import random
import re
import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from constants import CLASS_NAMES, DEFAULT_CLASS_PIXEL_COUNTS

logger = logging.getLogger(__name__)


def _worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class DistributedWeightedSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        dataset: Any,
        weights: torch.Tensor,
        num_replicas: int,
        rank: int,
        num_samples_per_replica: int,
        replacement: bool = True,
        seed: int = 0,
    ) -> None:
        self.dataset = dataset
        self.weights = weights.float()
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples_per_replica = num_samples_per_replica
        self.replacement = replacement
        self.seed = seed
        self.current_epoch = 0

    def __iter__(self):
        seed = self.seed + self.current_epoch
        g = torch.Generator()
        g.manual_seed(seed)
        indices = torch.multinomial(
            self.weights,
            self.num_replicas * self.num_samples_per_replica,
            replacement=self.replacement,
            generator=g,
        )
        rank_indices = indices[self.rank :: self.num_replicas]
        return iter(rank_indices.tolist())

    def __len__(self) -> int:
        return self.num_samples_per_replica

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = epoch


class CurriculumDataLoaderFactory:
    def __init__(
        self,
        dataset: Any,
        config: dict,
        is_distributed: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.dataset = dataset
        self.config = config
        self.is_distributed = is_distributed
        self.rank = rank
        self.world_size = world_size

        train_cfg = config.get("training", {})
        self.batch_size = train_cfg.get("batch_size_per_gpu", 8)
        self.num_workers = train_cfg.get("num_workers", 8)
        self.pin_memory = train_cfg.get("pin_memory", True)
        self.cache_by_phase = bool(train_cfg.get("cache_dataloaders_by_phase", True))
        self.land_rich_min_fraction = self._safe_ratio(
            train_cfg.get("land_rich_min_fraction", 0.08),
            key="training.land_rich_min_fraction",
            default=0.08,
        )
        self.land_rich_sampling_floor_ratio = self._safe_ratio(
            train_cfg.get("land_rich_sampling_floor_ratio", 0.0),
            key="training.land_rich_sampling_floor_ratio",
            default=0.0,
        )
        self.persistent_workers = bool(
            train_cfg.get(
                "persistent_workers",
                False,
            )
        )

        self.curriculum_cfg = config.get("curriculum", {})
        self.sampler_seed = int(config.get("dataset", {}).get("split_seed", 42))
        self._phase_ranges = self._parse_phase_ranges()

        pixel_counts = config.get("dataset", {}).get("pixel_counts")
        if pixel_counts is None:
            pixel_counts = self._compute_dataset_pixel_counts()
            self.config.setdefault("dataset", {})["pixel_counts"] = dict(pixel_counts)
            logger.warning(
                "dataset.pixel_counts missing; computed from active train split: %s",
                pixel_counts,
            )
        self._per_image_weights = self._compute_image_weights(pixel_counts)

        self._ship_image_indices = [
            i for i, meta in enumerate(dataset.metadata) if meta["has_ship"]
        ]
        self._land_rich_indices = self._resolve_land_rich_indices(
            min_fraction=self.land_rich_min_fraction
        )
        logger.info(
            f"Ship images in training set: {len(self._ship_image_indices)}/{len(dataset)}"
        )
        logger.info(
            "Land-rich sampling candidates: %d/%d (min_land_fraction=%.4f)",
            len(self._land_rich_indices),
            len(dataset),
            self.land_rich_min_fraction,
        )

        self._current_phase = None

        self._phase_override: Optional[int] = None

        self._cached_loader: Optional[DataLoader] = None
        self._cached_phase: Optional[int] = None
        self._cached_transform_id: Optional[int] = None

    def _parse_phase_ranges(self) -> list[tuple[int, int, int]]:
        phase_ranges: list[tuple[int, int, int]] = []
        phase_key_re = re.compile(r"^phase_(\d+)$")
        for key, phase_cfg in self.curriculum_cfg.items():
            if not isinstance(key, str):
                continue
            match = phase_key_re.match(key)
            if match is None or not isinstance(phase_cfg, Mapping):
                continue
            phase = int(match.group(1))
            epochs = phase_cfg.get("epochs")
            if not isinstance(epochs, Sequence) or len(epochs) < 2:
                continue
            start_epoch = int(epochs[0])
            end_epoch = int(epochs[1])
            phase_ranges.append((phase, start_epoch, end_epoch))

        if phase_ranges:
            phase_ranges.sort(key=lambda item: item[0])
            return phase_ranges

        training_epochs = int(self.config.get("training", {}).get("num_epochs", 1))
        if training_epochs <= 0:
            training_epochs = 1

        if self.curriculum_cfg:
            raise ValueError(
                "curriculum is configured but no valid phase_N.epochs entries were found. "
                "Define curriculum phases like phase_1: {epochs: [start, end]}"
            )

        # No curriculum provided: use a single config-driven phase over the full run.
        return [(1, 1, training_epochs)]

    def _shutdown_loader_workers(self, loader: Optional[DataLoader]) -> None:
        if loader is None:
            return
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None and hasattr(iterator, "_shutdown_workers"):
            iterator._shutdown_workers()
            loader._iterator = None

    def clear_cache(self) -> None:
        self._shutdown_loader_workers(self._cached_loader)
        self._cached_loader = None
        self._cached_phase = None
        self._cached_transform_id = None

    def _build_loader_for_phase(self, phase: int) -> DataLoader:
        phase_weights = self._get_phase_weights(phase)
        if self.is_distributed:
            weights_tensor = torch.from_numpy(phase_weights)
            num_samples_per_replica = math.ceil(len(self.dataset) / self.world_size)
            sampler = DistributedWeightedSampler(
                dataset=self.dataset,
                weights=weights_tensor,
                num_replicas=self.world_size,
                rank=self.rank,
                num_samples_per_replica=num_samples_per_replica,
                replacement=True,
                seed=self.sampler_seed,
            )
            return DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=True,
                worker_init_fn=_worker_init_fn,
                persistent_workers=self.persistent_workers and self.num_workers > 0,
            )

        sampler = WeightedRandomSampler(
            weights=phase_weights.tolist(),
            num_samples=len(self.dataset),
            replacement=True,
        )

        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            worker_init_fn=_worker_init_fn,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
        )

    def _compute_image_weights(
        self, pixel_counts: Mapping[str, int] | Sequence[int]
    ) -> np.ndarray:
        if isinstance(pixel_counts, Mapping):
            class_pixel_counts = {
                name: float(pixel_counts[name])
                for name in CLASS_NAMES
                if name in pixel_counts
            }
        else:
            if len(pixel_counts) != len(CLASS_NAMES):
                raise ValueError(
                    "pixel_counts sequence must match CLASS_NAMES length "
                    f"({len(CLASS_NAMES)}), got {len(pixel_counts)}"
                )
            class_pixel_counts = {
                name: float(count) for name, count in zip(CLASS_NAMES, pixel_counts)
            }

        total_pixels = sum(class_pixel_counts.values())
        num_classes = len(class_pixel_counts)

        class_weights = {}
        for name, count in class_pixel_counts.items():
            class_weights[name] = total_pixels / (num_classes * max(count, 1))

        weights = np.ones(len(self.dataset), dtype=np.float32)

        for i, meta in enumerate(self.dataset.metadata):
            w = 1.0
            if meta.get("has_ship", False):
                w += class_weights.get("ship", 1.0)
            if meta.get("has_oil_spill", False):
                w += class_weights.get("oil_spill", 1.0)
            if meta.get("has_look_alike", False):
                w += class_weights.get("look_alike", 1.0)
            weights[i] = w

        weights = weights / weights.sum()
        return weights

    def _compute_dataset_pixel_counts(self) -> Mapping[str, int]:
        metadata = getattr(self.dataset, "metadata", None)
        if not isinstance(metadata, list) or not metadata:
            logger.warning(
                "Dataset metadata unavailable for pixel-count aggregation. "
                "Falling back to DEFAULT_CLASS_PIXEL_COUNTS."
            )
            return dict(DEFAULT_CLASS_PIXEL_COUNTS)

        counts = {name: 0 for name in CLASS_NAMES}
        for meta in metadata:
            per_image = meta.get("pixel_counts", {})
            if isinstance(per_image, Mapping):
                for name in CLASS_NAMES:
                    counts[name] += int(per_image.get(name, 0))
            elif isinstance(per_image, Sequence) and len(per_image) == len(CLASS_NAMES):
                for idx, name in enumerate(CLASS_NAMES):
                    counts[name] += int(per_image[idx])

        if sum(counts.values()) <= 0:
            logger.warning(
                "Aggregated pixel counts are empty. Falling back to DEFAULT_CLASS_PIXEL_COUNTS."
            )
            return dict(DEFAULT_CLASS_PIXEL_COUNTS)
        return counts

    def get_phase(self, epoch: int) -> int:
        for phase, start_epoch, end_epoch in self._phase_ranges:
            if start_epoch <= epoch <= end_epoch:
                return phase

        if epoch < self._phase_ranges[0][1]:
            return self._phase_ranges[0][0]
        return self._phase_ranges[-1][0]

    def get_dataloader(self, epoch: int) -> DataLoader:
        phase = (
            self._phase_override
            if self._phase_override is not None
            else self.get_phase(epoch)
        )

        if phase != self._current_phase:
            self._log_phase_transition(phase, epoch)
            self._current_phase = phase

        transform_id = id(getattr(self.dataset, "transform", None))
        if (
            self.cache_by_phase
            and self._cached_loader is not None
            and self._cached_phase == phase
            and self._cached_transform_id == transform_id
        ):
            return self._cached_loader

        build_start = time.perf_counter()
        self.clear_cache()
        loader = self._build_loader_for_phase(phase)
        build_seconds = time.perf_counter() - build_start
        logger.info(
            "Built dataloader for phase %d in %.2fs (workers=%d, cache_by_phase=%s)",
            phase,
            build_seconds,
            self.num_workers,
            self.cache_by_phase,
        )
        if self.cache_by_phase:
            self._cached_loader = loader
            self._cached_phase = phase
            self._cached_transform_id = transform_id
        return loader

    def set_phase_override(self, phase: Optional[int]) -> None:
        old = self._phase_override
        self._phase_override = phase
        if old != phase:
            self.clear_cache()
            logger.info(f"DataLoader sampling phase override: {old!r} -> {phase!r}")

    def _get_phase_weights(self, phase: int) -> np.ndarray:
        phase_cfg = self.curriculum_cfg.get(f"phase_{phase}", {})
        default_oversample = {
            "ship": {1: 2.0, 2: 3.0, 3: 2.0},
            "oil_spill": {1: 1.0, 2: 1.0, 3: 1.0},
            "look_alike": {1: 1.0, 2: 1.0, 3: 1.0},
        }
        ship_boost = float(
            phase_cfg.get(
                "ship_oversample_factor", default_oversample["ship"].get(phase, 1.0)
            )
        )
        oil_boost = float(
            phase_cfg.get(
                "oil_spill_oversample_factor",
                default_oversample["oil_spill"].get(phase, 1.0),
            )
        )
        look_alike_boost = float(
            phase_cfg.get(
                "look_alike_oversample_factor",
                default_oversample["look_alike"].get(phase, 1.0),
            )
        )
        ship_sampling_floor_multiplier = float(
            phase_cfg.get("ship_sampling_floor_multiplier", 1.0)
        )
        land_rich_min_fraction = self._safe_ratio(
            phase_cfg.get("land_rich_min_fraction", self.land_rich_min_fraction),
            key=f"curriculum.phase_{phase}.land_rich_min_fraction",
            default=self.land_rich_min_fraction,
        )
        land_rich_sampling_floor_ratio = self._safe_ratio(
            phase_cfg.get(
                "land_rich_sampling_floor_ratio",
                self.land_rich_sampling_floor_ratio,
            ),
            key=f"curriculum.phase_{phase}.land_rich_sampling_floor_ratio",
            default=self.land_rich_sampling_floor_ratio,
        )
        land_rich_sampling_floor_ratio = float(
            min(0.95, max(0.0, land_rich_sampling_floor_ratio))
        )

        weights = self._per_image_weights.copy()
        for idx, meta in enumerate(self.dataset.metadata):
            sample_boost = 1.0
            if meta.get("has_ship", False):
                sample_boost += max(ship_boost - 1.0, 0.0)
            if meta.get("has_oil_spill", False):
                sample_boost += max(oil_boost - 1.0, 0.0)
            if meta.get("has_look_alike", False):
                sample_boost += max(look_alike_boost - 1.0, 0.0)
            weights[idx] *= sample_boost

        if ship_sampling_floor_multiplier > 1.0:
            for idx in self._ship_image_indices:
                base_weight = self._per_image_weights[idx]
                floor_weight = base_weight * ship_sampling_floor_multiplier
                if weights[idx] < floor_weight:
                    weights[idx] = floor_weight

        land_rich_indices = self._resolve_land_rich_indices(
            min_fraction=land_rich_min_fraction
        )
        if land_rich_sampling_floor_ratio > 0.0 and land_rich_indices:
            land_idx = np.asarray(land_rich_indices, dtype=np.int64)
            land_mask = np.zeros(len(weights), dtype=bool)
            land_mask[land_idx] = True

            total_mass = float(weights.sum())
            land_mass = float(weights[land_mask].sum())
            if total_mass <= 0.0 or land_mass <= 0.0:
                weights[land_mask] = weights[land_mask] + 1e-8
                total_mass = float(weights.sum())
                land_mass = float(weights[land_mask].sum())

            if total_mass > 0.0 and land_mass > 0.0:
                land_ratio = land_mass / total_mass
            else:
                land_ratio = 0.0

            if land_ratio < land_rich_sampling_floor_ratio:
                non_land_mass = float(weights[~land_mask].sum())
                if non_land_mass > 0.0:
                    denom = (1.0 - land_rich_sampling_floor_ratio) * max(
                        land_mass, 1e-12
                    )
                    scale_land = (land_rich_sampling_floor_ratio * non_land_mass) / max(
                        denom, 1e-12
                    )
                    weights[land_mask] *= max(scale_land, 1.0)

        weights = weights / weights.sum()
        return weights

    def _resolve_land_rich_indices(self, min_fraction: float) -> list[int]:
        land_rich_indices: list[int] = []
        for idx, meta in enumerate(self.dataset.metadata):
            pixel_fractions = meta.get("pixel_fractions", {})
            land_fraction = 0.0
            if isinstance(pixel_fractions, Mapping):
                land_fraction = float(pixel_fractions.get("land", 0.0))
            elif isinstance(pixel_fractions, Sequence) and len(pixel_fractions) == len(
                CLASS_NAMES
            ):
                land_fraction = float(pixel_fractions[CLASS_NAMES.index("land")])

            if land_fraction >= min_fraction:
                land_rich_indices.append(idx)

        if land_rich_indices:
            return land_rich_indices

        return [
            idx
            for idx, meta in enumerate(self.dataset.metadata)
            if bool(meta.get("has_land", False))
        ]

    def _safe_ratio(self, raw_value: Any, *, key: str, default: float) -> float:
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            logger.warning("Invalid %s=%r; defaulting to %.4f", key, raw_value, default)
            return float(default)
        if not np.isfinite(value):
            logger.warning(
                "Non-finite %s=%r; defaulting to %.4f", key, raw_value, default
            )
            return float(default)
        return float(value)

    def _log_phase_transition(self, new_phase: int, epoch: int) -> None:
        phase_cfg = self.curriculum_cfg.get(f"phase_{new_phase}", {})
        n_ship = len(self._ship_image_indices)
        land_min_fraction = self._safe_ratio(
            phase_cfg.get("land_rich_min_fraction", self.land_rich_min_fraction),
            key=f"curriculum.phase_{new_phase}.land_rich_min_fraction",
            default=self.land_rich_min_fraction,
        )
        land_floor = self._safe_ratio(
            phase_cfg.get(
                "land_rich_sampling_floor_ratio",
                self.land_rich_sampling_floor_ratio,
            ),
            key=f"curriculum.phase_{new_phase}.land_rich_sampling_floor_ratio",
            default=self.land_rich_sampling_floor_ratio,
        )
        n_land_rich = len(
            self._resolve_land_rich_indices(min_fraction=land_min_fraction)
        )
        desc = (
            f"Configured phase sampling ({n_ship} ship images "
            f"ship={phase_cfg.get('ship_oversample_factor', 1.0)}x, "
            f"oil={phase_cfg.get('oil_spill_oversample_factor', 1.0)}x, "
            f"look_alike={phase_cfg.get('look_alike_oversample_factor', 1.0)}x, "
            f"ship_floor={phase_cfg.get('ship_sampling_floor_multiplier', 1.0)}x, "
            f"land_rich_floor={land_floor:.2f}, "
            f"land_rich_min_frac={land_min_fraction:.3f}, "
            f"land_rich_candidates={n_land_rich})"
        )
        logger.info(f"=== Curriculum Phase {new_phase} (epoch {epoch}): {desc} ===")
