#!/usr/bin/env python3
"""Convert JSONL logs to AI-friendly formats.

Supported outputs:
- csv (default): flattened records with one row per event
- tsv: same as csv, tab-separated
- jsonl: normalized JSON Lines
- json: list of normalized records

Examples:
  python scripts/convert_jsonl_log.py logs/events.jsonl
  python scripts/convert_jsonl_log.py logs/events.jsonl -f csv -o logs/events.csv
  python scripts/convert_jsonl_log.py logs/events.jsonl -f json -o logs/events.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert JSONL logs to CSV/TSV/JSONL/JSON for AI consumption."
    )
    parser.add_argument("input", type=Path, help="Path to input .jsonl log file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output file path (default: same name with extension based on format)",
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=("csv", "tsv", "jsonl", "json"),
        default="csv",
        help="Output format (default: csv)",
    )
    parser.add_argument(
        "--flatten-sep",
        default=".",
        help="Separator for flattened nested keys (default: '.')",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional limit on number of parsed rows",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on malformed JSON lines instead of skipping",
    )
    parser.add_argument(
        "--include-raw",
        action="store_true",
        help="Include original JSON line in an extra 'raw_json' column/key",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="File encoding for input/output (default: utf-8)",
    )
    return parser.parse_args()


def default_output_path(input_path: Path, fmt: str) -> Path:
    extension = {
        "csv": ".csv",
        "tsv": ".tsv",
        "jsonl": ".normalized.jsonl",
        "json": ".normalized.json",
    }[fmt]
    return input_path.with_suffix(extension)


def flatten_record(value: Any, parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    """Flatten nested dicts into a single dict with dotted keys.

    Lists are JSON-encoded to preserve structure in tabular formats.
    """
    flat: dict[str, Any] = {}

    if isinstance(value, dict):
        for key, child in value.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else str(key)
            flat.update(flatten_record(child, new_key, sep=sep))
        return flat

    if isinstance(value, list):
        flat[parent_key or "value"] = json.dumps(value, ensure_ascii=False)
        return flat

    flat[parent_key or "value"] = value
    return flat


def read_jsonl(
    path: Path,
    *,
    encoding: str,
    flatten_sep: str,
    max_rows: int | None,
    strict: bool,
    include_raw: bool,
) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    skipped = 0

    with path.open("r", encoding=encoding) as source:
        for line_number, raw_line in enumerate(source, start=1):
            if max_rows is not None and len(records) >= max_rows:
                break

            line = raw_line.strip()
            if not line:
                continue

            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                if strict:
                    raise ValueError(
                        f"Invalid JSON at line {line_number} in {path}"
                    ) from None
                skipped += 1
                continue

            normalized = flatten_record(parsed, sep=flatten_sep)
            if include_raw:
                normalized["raw_json"] = line
            records.append(normalized)

    return records, skipped


def write_csv_like(
    records: list[dict[str, Any]], output_path: Path, encoding: str, delimiter: str
) -> None:
    all_fields: list[str] = []
    seen = set()

    for record in records:
        for key in record.keys():
            if key not in seen:
                seen.add(key)
                all_fields.append(key)

    with output_path.open("w", encoding=encoding, newline="") as target:
        writer = csv.DictWriter(target, fieldnames=all_fields, delimiter=delimiter)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def write_jsonl(
    records: list[dict[str, Any]], output_path: Path, encoding: str
) -> None:
    with output_path.open("w", encoding=encoding) as target:
        for record in records:
            target.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(records: list[dict[str, Any]], output_path: Path, encoding: str) -> None:
    with output_path.open("w", encoding=encoding) as target:
        json.dump(records, target, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()

    input_path: Path = args.input
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = args.output or default_output_path(input_path, args.format)
    records, skipped = read_jsonl(
        input_path,
        encoding=args.encoding,
        flatten_sep=args.flatten_sep,
        max_rows=args.max_rows,
        strict=args.strict,
        include_raw=args.include_raw,
    )

    if args.format == "csv":
        write_csv_like(records, output_path, args.encoding, delimiter=",")
    elif args.format == "tsv":
        write_csv_like(records, output_path, args.encoding, delimiter="\t")
    elif args.format == "jsonl":
        write_jsonl(records, output_path, args.encoding)
    else:
        write_json(records, output_path, args.encoding)

    print(f"Input:   {input_path}")
    print(f"Output:  {output_path}")
    print(f"Format:  {args.format}")
    print(f"Rows:    {len(records)}")
    print(f"Skipped: {skipped}")


if __name__ == "__main__":
    main()
