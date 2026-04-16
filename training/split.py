import random
from statistics import median
from typing import Any, Dict, List, Tuple

from constants import CLASS_NAMES

LAND_CLASS_IDX = CLASS_NAMES.index("land")


def build_train_val_indices(
    metadata: List[Dict[str, Any]], train_split: float, seed: int
) -> Tuple[List[int], List[int]]:

    def _class_fraction(meta: Dict[str, Any], class_name: str) -> float:
        fractions = meta.get("pixel_fractions")
        if isinstance(fractions, dict):
            return float(fractions.get(class_name, 0.0))
        if isinstance(fractions, (list, tuple)):
            try:
                idx = CLASS_NAMES.index(class_name)
                return float(fractions[idx])
            except (ValueError, IndexError, TypeError):
                return 0.0
        return 0.0

    def _has_land(meta: Dict[str, Any]) -> bool:
        if "has_land" in meta:
            return bool(meta.get("has_land", False))
        counts = meta.get("pixel_counts")
        if isinstance(counts, dict):
            return bool(counts.get("land", 0))
        if isinstance(counts, (list, tuple)) and len(counts) > LAND_CLASS_IDX:
            return bool(counts[LAND_CLASS_IDX])
        return _class_fraction(meta, "land") > 0.0

    def _median_nonzero_fraction(
        records: List[Dict[str, Any]], class_name: str
    ) -> float:
        values = [
            _class_fraction(record, class_name)
            for record in records
            if _class_fraction(record, class_name) > 0.0
        ]
        return float(median(values)) if values else 0.0

    def _fraction_bin(value: float, nonzero_median: float) -> int:
        if value <= 0.0:
            return 0
        if nonzero_median <= 0.0:
            return 1
        return 1 if value <= nonzero_median else 2

    ship_median = _median_nonzero_fraction(metadata, "ship")
    look_median = _median_nonzero_fraction(metadata, "look_alike")
    land_median = _median_nonzero_fraction(metadata, "land")

    grouped_indices: Dict[Tuple[bool, bool, bool, bool, int, int, int], List[int]] = {}
    for idx, meta in enumerate(metadata):
        ship_fraction = _class_fraction(meta, "ship")
        look_fraction = _class_fraction(meta, "look_alike")
        land_fraction = _class_fraction(meta, "land")
        key = (
            bool(meta.get("has_ship", False)),
            bool(meta.get("has_oil_spill", False)),
            bool(meta.get("has_look_alike", False)),
            _has_land(meta),
            _fraction_bin(ship_fraction, ship_median),
            _fraction_bin(look_fraction, look_median),
            _fraction_bin(land_fraction, land_median),
        )
        grouped_indices.setdefault(key, []).append(idx)

    rng = random.Random(seed)
    train_indices: List[int] = []
    val_indices: List[int] = []

    for group in grouped_indices.values():
        shuffled = group[:]
        rng.shuffle(shuffled)

        if len(shuffled) == 1:
            train_count = 1
        else:
            train_count = int(round(len(shuffled) * train_split))
            train_count = max(1, min(len(shuffled) - 1, train_count))

        train_indices.extend(shuffled[:train_count])
        val_indices.extend(shuffled[train_count:])

    if not val_indices:
        fallback_idx = train_indices.pop()
        val_indices.append(fallback_idx)

    train_indices.sort()
    val_indices.sort()
    return train_indices, val_indices
