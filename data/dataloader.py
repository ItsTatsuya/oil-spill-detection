import logging
import math
import random
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
        self.persistent_workers = bool(
            train_cfg.get(
                "persistent_workers",
                False,
            )
        )

        self.curriculum_cfg = config.get("curriculum", {})
        self.sampler_seed = int(config.get("dataset", {}).get("split_seed", 42))
        self._phase1_end = self.curriculum_cfg.get("phase_1", {}).get(
            "epochs", [1, 50]
        )[1]
        self._phase2_end = self.curriculum_cfg.get("phase_2", {}).get(
            "epochs", [51, 150]
        )[1]

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
        logger.info(
            f"Ship images in training set: {len(self._ship_image_indices)}/{len(dataset)}"
        )

        self._current_phase = None

        self._phase_override: Optional[int] = None
        self._phase_epoch_offset = 0

        self._cached_loader: Optional[DataLoader] = None
        self._cached_phase: Optional[int] = None
        self._cached_transform_id: Optional[int] = None

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

    def _to_phase_epoch(self, epoch: int) -> int:
        if self._phase_epoch_offset <= 0:
            return int(epoch)
        return max(int(epoch) - self._phase_epoch_offset, 1)

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
            weights=phase_weights,
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
        phase_epoch = self._to_phase_epoch(epoch)
        stagger_1_to_2 = self.curriculum_cfg.get("phase_1_to_2", {})
        stagger_2_to_3 = self.curriculum_cfg.get("phase_2_to_3", {})

        if stagger_1_to_2:
            sampling_start_phase2 = int(
                stagger_1_to_2.get("sampling_epoch", self._phase1_end + 1)
            )
            sampling_start_phase3 = int(
                stagger_2_to_3.get("sampling_epoch", self._phase2_end + 1)
            )
            if phase_epoch < sampling_start_phase2:
                return 1
            elif phase_epoch < sampling_start_phase3:
                return 2
            else:
                return 3

        if phase_epoch <= self._phase1_end:
            return 1
        elif phase_epoch <= self._phase2_end:
            return 2
        else:
            return 3

    def get_dataloader(self, epoch: int) -> DataLoader:
        phase = (
            self._phase_override
            if self._phase_override is not None
            else self.get_phase(epoch)
        )

        if phase != self._current_phase:
            self._log_phase_transition(phase, epoch, self._to_phase_epoch(epoch))
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

    def set_phase_epoch_offset(self, epoch_offset: int) -> None:
        old_offset = self._phase_epoch_offset
        new_offset = max(int(epoch_offset), 0)
        self._phase_epoch_offset = new_offset
        if old_offset != new_offset:
            self.clear_cache()
            self._current_phase = None
            logger.info(
                "Curriculum epoch offset updated: %d -> %d",
                old_offset,
                new_offset,
            )

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

        if phase == 3 and ship_sampling_floor_multiplier > 1.0:
            for idx in self._ship_image_indices:
                base_weight = self._per_image_weights[idx]
                floor_weight = base_weight * ship_sampling_floor_multiplier
                if weights[idx] < floor_weight:
                    weights[idx] = floor_weight

        weights = weights / weights.sum()
        return weights

    def _log_phase_transition(
        self, new_phase: int, epoch: int, phase_epoch: int
    ) -> None:
        if new_phase == 2:
            phase2_cfg = self.curriculum_cfg.get("phase_2", {})
            n_ship = len(self._ship_image_indices)
            desc = (
                f"Ship-enriched sampling ({n_ship} ship images "
                f"ship={phase2_cfg.get('ship_oversample_factor', 3.0)}x, "
                f"oil={phase2_cfg.get('oil_spill_oversample_factor', 1.0)}x, "
                f"look_alike={phase2_cfg.get('look_alike_oversample_factor', 1.0)}x)"
            )
        elif new_phase == 3:
            phase3_cfg = self.curriculum_cfg.get("phase_3", {})
            desc = (
                "Inverse-frequency rare-class rebalance "
                f"(ship={phase3_cfg.get('ship_oversample_factor', 2.0)}x, "
                f"oil={phase3_cfg.get('oil_spill_oversample_factor', 1.0)}x, "
                f"look_alike={phase3_cfg.get('look_alike_oversample_factor', 1.0)}x)"
            )
        else:
            phase1_cfg = self.curriculum_cfg.get("phase_1", {})
            desc = (
                "Standard sampling with inverse-frequency class weights "
                f"(ship={phase1_cfg.get('ship_oversample_factor', 2.0)}x, "
                f"oil={phase1_cfg.get('oil_spill_oversample_factor', 1.0)}x, "
                f"look_alike={phase1_cfg.get('look_alike_oversample_factor', 1.0)}x)"
            )
        if phase_epoch != epoch:
            logger.info(
                "=== Curriculum Phase %d (epoch %d, curriculum_epoch %d): %s ===",
                new_phase,
                epoch,
                phase_epoch,
                desc,
            )
        else:
            logger.info(
                "=== Curriculum Phase %d (epoch %d): %s ===",
                new_phase,
                epoch,
                desc,
            )
