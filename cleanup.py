from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent

GROUPS = [
    {
        "label": "Standard Pre-Run Cleanup",
        "paths": [
            "checkpoints",
            "logs",
            "predictions",
            "evaluation_output",
            ".pytest_cache",
        ],
    },
    {"label": "Checkpoints", "paths": ["checkpoints"]},
    {"label": "Logs", "paths": ["logs"]},
    {"label": "Predictions", "paths": ["predictions"]},
    {"label": "Evaluation output", "paths": ["evaluation_output"]},
    {"label": "Ablation results", "paths": ["ablations/results"]},
    {"label": "__pycache__ (all)", "paths": "__pycache__"},
    # --- MOVED CACHES OUT OF STANDARD CLEANUP ---
    {
        "label": "⚠️ DANGEROUS: SAR Dataset Cache (.npy files)",
        "paths": ["data/sar_cache"],
    },
    {
        "label": "⚠️ DANGEROUS: Ship Crop Library (.pkl files)",
        "paths": ["data/ship_crops"],
    },
    {"label": "⚠️ DANGEROUS: SAR Stats (.json files)", "paths": ["data/sar_stats"]},
]


def get_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def fmt_size(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


def count_files(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return 1
    return sum(1 for f in path.rglob("*") if f.is_file())


def find_pycaches() -> list[Path]:
    return sorted(ROOT.rglob("__pycache__"))


def iter_group_paths(group: dict) -> list[Path]:
    if group["paths"] == "__pycache__":
        return find_pycaches()
    return [ROOT / rel_path for rel_path in group["paths"]]


def get_group_info(group: dict) -> tuple[int, int]:
    paths = iter_group_paths(group)
    total_files = sum(count_files(path) for path in paths)
    total_size = sum(get_size(path) for path in paths)
    return total_files, total_size


# --- FIX: Completely remove the directory instead of just its contents ---
def empty_path(path: Path) -> int:
    if not path.exists():
        return 0

    freed = get_size(path)
    if path.is_file():
        path.unlink(missing_ok=True)
    else:
        # This will permanently delete the folder itself (like __pycache__)
        shutil.rmtree(path, ignore_errors=True)
    return freed


def delete_group(group: dict) -> int:
    return sum(empty_path(path) for path in iter_group_paths(group))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive cleanup utility for SAR Oil Spill artifacts"
    )
    return parser.parse_args()


def main() -> None:
    parse_args()
    print("\n  SAR Oil Spill - Cleanup Tool")
    print("  " + "=" * 40)
    print(
        "  Standard pre-run cleanup (Option 1) will NO LONGER delete your SAR caches."
    )
    print()

    for i, group in enumerate(GROUPS, 1):
        files, size = get_group_info(group)
        status = f"{files} files, {fmt_size(size)}" if files else "empty"
        print(f"  {i}. {group['label']:<45s} [{status}]")

    print()
    print("  a. Select all")
    print("  q. Quit")
    print()

    choice = input("  Enter numbers to delete (e.g. 1 7): ").strip().lower()

    if choice in ("q", ""):
        print("  Cancelled.")
        return

    if choice == "a":
        indices = list(range(len(GROUPS)))
    else:
        try:
            indices = [int(item) - 1 for item in choice.split()]
        except ValueError:
            print("  Invalid input.")
            return
        for idx in indices:
            if idx < 0 or idx >= len(GROUPS):
                print(f"  Invalid number: {idx + 1}")
                return

    print()
    print("  Will delete:")
    total_files = 0
    total_size = 0
    for idx in indices:
        files, size = get_group_info(GROUPS[idx])
        total_files += files
        total_size += size
        print(f"    * {GROUPS[idx]['label']}")

    print(f"\n  Total: {total_files} files, {fmt_size(total_size)}")
    confirm = input("  Confirm? [y/N]: ").strip().lower()

    if confirm != "y":
        print("  Cancelled.")
        return

    freed = 0
    for idx in indices:
        freed += delete_group(GROUPS[idx])
        print(f"  Cleared {GROUPS[idx]['label']}")

    print(f"\n  Done. Freed {fmt_size(freed)}.")


if __name__ == "__main__":
    main()
