#!/usr/bin/env python3
"""Remove training artifacts for a clean restart.

Deletes caches, stats, checkpoints, logs, and predictions by default.
Does NOT touch the dataset, source code, or downloaded pretrained weights
(unless --include-pretrained is passed).

Examples:
  python cleanup.py              # dry-run (prints what would be removed)
  python cleanup.py --yes        # actually delete
  python cleanup.py --yes --all  # also wipe __pycache__
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

# Safe-to-delete training artifacts (relative to project root).
DEFAULT_PATHS: list[str] = [
    "checkpoints",
    "logs",
    "predictions",
    "evaluation_output",
    "data/sar_cache",
    "data/sar_stats",
    "data/ship_crops",
    "ship_crops",
    "sar_cache",
    "sar_stats",
    "sar_stats.json",
    "results_paper_ablation",
]

# Optional / aggressive extras.
PYCACHE_GLOBS = ("**/__pycache__", "**/.pytest_cache", "**/*.pyc")
PRETRAINED_PATHS = ("models/pretrained", "pretrained")


def _human_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{num_bytes}B"


def _path_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                pass
    return total


def collect_targets(
    *,
    include_pycache: bool,
    include_pretrained: bool,
) -> list[Path]:
    targets: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path) -> None:
        resolved = path.resolve()
        if resolved in seen:
            return
        # Never climb above project root.
        try:
            resolved.relative_to(ROOT)
        except ValueError:
            return
        if not path.exists():
            return
        seen.add(resolved)
        targets.append(path)

    for rel in DEFAULT_PATHS:
        add(ROOT / rel)

    if include_pycache:
        for pattern in PYCACHE_GLOBS:
            for match in ROOT.glob(pattern):
                add(match)

    if include_pretrained:
        for rel in PRETRAINED_PATHS:
            add(ROOT / rel)

    return sorted(targets, key=lambda p: str(p))


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
        return
    if path.is_dir():
        shutil.rmtree(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean training artifacts (caches, stats, checkpoints, logs, "
            "predictions) for a fresh run."
        )
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Actually delete files. Without this flag, only dry-run.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Also remove __pycache__ / .pytest_cache / *.pyc.",
    )
    parser.add_argument(
        "--include-pretrained",
        action="store_true",
        help=(
            "Also delete models/pretrained (forces re-download). "
            "Not recommended for routine restarts."
        ),
    )
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        help="Extra relative path under the project root to delete (repeatable).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    targets = collect_targets(
        include_pycache=bool(args.all),
        include_pretrained=bool(args.include_pretrained),
    )
    for rel in args.extra:
        candidate = (ROOT / rel).resolve()
        try:
            candidate.relative_to(ROOT)
        except ValueError:
            print(f"Refusing path outside project root: {rel}", file=sys.stderr)
            return 2
        if candidate.exists():
            targets.append(ROOT / rel)

    # De-dupe while preserving order.
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in targets:
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    targets = deduped

    if not targets:
        print("Nothing to clean (no matching artifacts found).")
        return 0

    total_bytes = sum(_path_size(p) for p in targets)
    mode = "DELETE" if args.yes else "DRY-RUN"
    print(f"[{mode}] Project root: {ROOT}")
    print(f"[{mode}] {len(targets)} path(s), ~{_human_size(total_bytes)}")
    for path in targets:
        kind = "dir" if path.is_dir() else "file"
        print(f"  - ({kind}, {_human_size(_path_size(path))}) {path.relative_to(ROOT)}")

    if not args.yes:
        print("\nDry-run only. Re-run with --yes to delete.")
        return 0

    failed = 0
    for path in targets:
        try:
            remove_path(path)
            print(f"  removed {path.relative_to(ROOT)}")
        except OSError as exc:
            failed += 1
            print(f"  FAILED {path.relative_to(ROOT)}: {exc}", file=sys.stderr)

    if failed:
        print(f"Finished with {failed} failure(s).", file=sys.stderr)
        return 1
    print("Cleanup complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
