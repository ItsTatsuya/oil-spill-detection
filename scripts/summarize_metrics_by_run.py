from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _best_metric(record: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        value = _safe_float(record.get(key))
        if value is not None:
            return value
    return None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def summarize(
    rows: list[dict[str, Any]], run_id_filter: str | None = None
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        run_id = str(row.get("run_id") or "legacy")
        if run_id_filter is not None and run_id != run_id_filter:
            continue
        grouped.setdefault(run_id, []).append(row)

    summaries: list[dict[str, Any]] = []
    for run_id, records in grouped.items():
        steps = [
            int(r.get("step", 0))
            for r in records
            if isinstance(r.get("step"), (int, float))
        ]
        best_val_miou = None
        best_ship_iou = None
        latest_train_loss = None

        for record in records:
            val_miou = _best_metric(
                record, ["val/mean_iou", "val_full/mean_iou", "val_fast/mean_iou"]
            )
            if val_miou is not None:
                best_val_miou = (
                    val_miou if best_val_miou is None else max(best_val_miou, val_miou)
                )

            ship_iou = _best_metric(
                record, ["val_full/iou_ship", "val_fast/iou_ship", "val/iou_ship"]
            )
            if ship_iou is not None:
                best_ship_iou = (
                    ship_iou if best_ship_iou is None else max(best_ship_iou, ship_iou)
                )

            train_loss = _best_metric(record, ["train/loss_total", "loss_total"])
            if train_loss is not None:
                latest_train_loss = train_loss

        summaries.append(
            {
                "run_id": run_id,
                "records": len(records),
                "first_step": min(steps) if steps else None,
                "last_step": max(steps) if steps else None,
                "best_val_miou": best_val_miou,
                "best_ship_iou": best_ship_iou,
                "latest_train_loss": latest_train_loss,
            }
        )

    summaries.sort(
        key=lambda item: (item["last_step"] is not None, item["last_step"]),
        reverse=True,
    )
    return summaries


def format_table(summaries: list[dict[str, Any]]) -> str:
    if not summaries:
        return "No matching runs found."

    header = (
        f"{'run_id':<20} {'records':>8} {'step_range':>15} "
        f"{'best_mIoU(%)':>12} {'best_ship(%)':>12} {'last_train_loss':>15}"
    )
    lines = [header, "-" * len(header)]

    for item in summaries:
        step_range = "-"
        if item["first_step"] is not None and item["last_step"] is not None:
            step_range = f"{item['first_step']}..{item['last_step']}"

        best_miou_pct = "-"
        if item["best_val_miou"] is not None:
            best_miou_pct = f"{item['best_val_miou'] * 100.0:.2f}"

        best_ship_pct = "-"
        if item["best_ship_iou"] is not None:
            best_ship_pct = f"{item['best_ship_iou'] * 100.0:.2f}"

        last_train_loss = "-"
        if item["latest_train_loss"] is not None:
            last_train_loss = f"{item['latest_train_loss']:.4f}"

        lines.append(
            f"{item['run_id']:<20} {item['records']:>8} {step_range:>15} "
            f"{best_miou_pct:>12} {best_ship_pct:>12} {last_train_loss:>15}"
        )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize metrics.jsonl grouped by run_id"
    )
    parser.add_argument(
        "--metrics-file",
        type=str,
        default="logs/metrics.jsonl",
        help="Path to JSONL metrics file",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run_id filter",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON summary instead of table",
    )
    args = parser.parse_args()

    metrics_path = Path(args.metrics_file)
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    rows = load_jsonl(metrics_path)
    summaries = summarize(rows, run_id_filter=args.run_id)

    if args.json:
        print(json.dumps(summaries, indent=2))
    else:
        print(format_table(summaries))


if __name__ == "__main__":
    main()
