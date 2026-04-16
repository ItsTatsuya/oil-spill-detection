import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional


def get_collapsed_file_lines(files: list[Path], threshold: int = 10) -> list[str]:
    """
    Smart collapsing for large groups of similarly-named files.
    Example: img_1.jpg ... img_10000.jpg becomes:
      📄 img_1.jpg
      📄 img_2.jpg
      📄 ...
      📄 img_10000.jpg
    Only collapses if the numbers are perfectly consecutive (no gaps).
    Preserves original filenames (including leading zeros like 0001).
    """
    if not files:
        return []

    # Regex: prefix + number + suffix (takes the first number group in the name)
    pattern = re.compile(r"^(.*?)(\d+)(.*)$")

    groups = defaultdict(list)
    lone_files = []

    for f in files:
        match = pattern.match(f.name)
        if match:
            prefix, num_str, suffix = match.groups()
            groups[(prefix, suffix)].append(
                (int(num_str), f.name)
            )  # store original name
        else:
            lone_files.append(f.name)

    lines = []

    # Lone files (no number pattern)
    for name in sorted(lone_files):
        lines.append(f"📄 {name}")

    # Numbered groups
    for key, items in groups.items():
        items.sort(key=lambda x: x[0])  # sort by numeric value
        nums = [x[0] for x in items]
        prefix, suffix = key
        first_name = items[0][1]
        second_name = items[1][1] if len(items) > 1 else None
        last_name = items[-1][1]

        if len(items) >= threshold and nums == list(range(nums[0], nums[-1] + 1)):
            # Collapse!
            lines.append(f"📄 {first_name}")
            if second_name:
                lines.append(f"📄 {second_name}")
            lines.append("📄 ...")
            lines.append(f"📄 {last_name}")
        else:
            # Show every file individually
            for _, name in items:
                lines.append(f"📄 {name}")

    return lines


def generate_directory_tree(
    root_dir: str = ".",
    output_file: str = "directory_structure.md",
    ignore_dirs: Optional[set] = None,
    collapse_threshold: int = 10,
    max_depth: Optional[int] = None,
) -> None:
    """
    Generates a beautiful, properly indented Markdown tree of your project.
    Now with intelligent collapsing for huge datasets (e.g. img_1 → img_10000).
    """
    if ignore_dirs is None:
        ignore_dirs = {
            ".git",
            "__pycache__",
            "venv",
            ".venv",
            "env",
            ".env",
            "node_modules",
            ".vscode",
            ".idea",
            "build",
            "dist",
            ".pytest_cache",
            ".DS_Store",
        }

    root_path = Path(root_dir).resolve()

    md_lines = [
        "# 📁 Project Directory Structure",
        "",
        f"**Root directory:** `{root_path.name}`",
        f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Collapse threshold:** {collapse_threshold}+ similar files",
        "",
        "```",
    ]

    def tree(path: Path, prefix: str = "", depth: int = 0) -> None:
        if max_depth is not None and depth > max_depth:
            return
        if path.is_dir() and path.name in ignore_dirs:
            return

        try:
            items = list(path.iterdir())
        except (PermissionError, OSError):
            return

        # Separate directories and files
        dirs = [
            item for item in items if item.is_dir() and item.name not in ignore_dirs
        ]
        files = [item for item in items if item.is_file()]

        # Filter junk files but keep .gitignore and requirements.txt
        files = [
            f
            for f in files
            if f.name == ".gitignore"
            or f.name == "requirements.txt"
            or not (f.name.startswith(".") or f.name in {".DS_Store"})
        ]

        # Sort for consistent output
        dirs.sort(key=lambda x: x.name.lower())
        files.sort(key=lambda x: x.name.lower())

        # Render directories first (folders before files)
        for i, item in enumerate(dirs):
            is_last = (i == len(dirs) - 1) and not files
            connector = "└── " if is_last else "├── "
            md_lines.append(f"{prefix}{connector}📁 {item.name}/")
            new_prefix = prefix + ("    " if is_last else "│   ")
            tree(item, new_prefix, depth + 1)

        # Render files (with smart collapsing applied)
        if files:
            file_display_lines = get_collapsed_file_lines(
                files, threshold=collapse_threshold
            )
            for j, display_line in enumerate(file_display_lines):
                is_last = j == len(file_display_lines) - 1
                connector = "└── " if is_last else "├── "
                md_lines.append(f"{prefix}{connector}{display_line}")

    # Start the tree from root
    md_lines.append(f"📁 {root_path.name}/")
    tree(root_path)
    md_lines.append("```")

    # Write the file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"✅ Directory structure successfully written to → {output_file}")
    print(f"📍 Scanned root: {root_path}")
    print(
        f"   (Files collapsed when ≥{collapse_threshold} similar numbered files were found)"
    )


if __name__ == "__main__":
    # Run it! Change collapse_threshold if you want stricter/looser collapsing
    generate_directory_tree(collapse_threshold=8)  # 8 is a good balance
