import json
import logging

import matplotlib
import numpy as np

matplotlib.use("Agg")  
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import torch

from constants import CLASS_COLORS_RGB, CLASS_NAMES, NUM_CLASSES

logger = logging.getLogger(__name__)

def _resolve_class_colors(config: Optional[dict] = None) -> list[list[int]]:
    dataset_cfg = (config or {}).get("dataset", {})
    resolved = [list(color) for color in CLASS_COLORS_RGB]
    for idx, class_name in enumerate(CLASS_NAMES):
        configured = dataset_cfg.get("label_color_override", {}).get(class_name)
        if configured is None:
            class_colors_cfg = dataset_cfg.get("class_colors", {})
            configured = class_colors_cfg.get(class_name)
        if configured is not None and len(configured) == 3:
            resolved[idx] = [int(v) for v in configured]
    return resolved


def class_mask_to_rgb(mask: np.ndarray, class_colors: Optional[list[list[int]]] = None) -> np.ndarray:
    H, W = mask.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    colors = class_colors or [list(color) for color in CLASS_COLORS_RGB]
    for c, color in enumerate(colors):
        class_pixels = mask == c
        rgb[class_pixels] = color
    return rgb


def create_class_legend(
    ax: plt.Axes,
    class_colors: Optional[list[list[int]]] = None,
) -> None:
    colors = class_colors or [list(color) for color in CLASS_COLORS_RGB]
    patches = [
        mpatches.Patch(
            color=[r / 255, g / 255, b / 255],
            label=name,
        )
        for name, (r, g, b) in zip(CLASS_NAMES, colors)
    ]
    ax.legend(
        handles=patches,
        loc="lower right",
        fontsize=8,
        framealpha=0.8,
        ncol=1,
    )


class PredictionVisualizer:

    def __init__(
        self,
        config: Optional[dict] = None,
        output_dir: str = "./evaluation_output/visualizations",
    ) -> None:
        self.config = config or {}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.class_colors = _resolve_class_colors(self.config)

    def visualize_prediction(
        self,
        image: Union[np.ndarray, torch.Tensor],
        ground_truth: Union[np.ndarray, torch.Tensor],
        prediction: Union[np.ndarray, torch.Tensor],
        save_path: str,
        sample_iou: Optional[float] = None,
        per_class_iou: Optional[Dict[str, float]] = None,
        title: Optional[str] = None,
    ) -> None:
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        if isinstance(ground_truth, torch.Tensor):
            ground_truth = ground_truth.cpu().numpy()
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.cpu().numpy()

        if image.ndim == 3 and image.shape[0] >= 1 and image.shape[0] <= 8:
            amplitude = image[0]  
        elif image.ndim == 3 and image.shape[2] >= 1 and image.shape[2] <= 8:
            amplitude = image[:, :, 0]  
        else:
            amplitude = image.squeeze()

        amp_min, amp_max = amplitude.min(), amplitude.max()
        if amp_max > amp_min:
            amplitude_display = (amplitude - amp_min) / (amp_max - amp_min)
        else:
            amplitude_display = amplitude

        gt_rgb = class_mask_to_rgb(
            ground_truth.astype(np.int64),
            class_colors=self.class_colors,
        )
        pred_rgb = class_mask_to_rgb(
            prediction.astype(np.int64),
            class_colors=self.class_colors,
        )

        if sample_iou is None:
            ious = []
            for c in range(NUM_CLASSES):
                tp = np.sum((prediction == c) & (ground_truth == c))
                fp = np.sum((prediction == c) & (ground_truth != c))
                fn = np.sum((prediction != c) & (ground_truth == c))
                if tp + fp + fn > 0:
                    ious.append(tp / (tp + fp + fn + 1e-6))
            sample_iou = float(np.mean(ious)) if ious else 0.0

        fig, axes = plt.subplots(1, 3, figsize=(18, 7))

        axes[0].imshow(amplitude_display, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("Input SAR Amplitude", fontsize=13, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(gt_rgb)
        axes[1].set_title("Ground Truth Mask", fontsize=13, fontweight="bold")
        axes[1].axis("off")
        create_class_legend(axes[1], class_colors=self.class_colors)

        axes[2].imshow(pred_rgb)
        axes[2].set_title(
            f"Predicted Mask (mIoU: {sample_iou * 100:.1f}%)",
            fontsize=13,
            fontweight="bold",
        )
        axes[2].axis("off")
        create_class_legend(axes[2], class_colors=self.class_colors)

        if per_class_iou:
            iou_lines = []
            for cname, iou_val in per_class_iou.items():
                if not np.isnan(iou_val):
                    iou_lines.append(f"{cname}: {iou_val * 100:.1f}%")
            if iou_lines:
                iou_text = "\n".join(iou_lines)
                axes[2].text(
                    0.02,
                    0.02,
                    iou_text,
                    transform=axes[2].transAxes,
                    fontsize=8,
                    va="bottom",
                    ha="left",
                    fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
                )

        main_title = "Oil Spill SAR Segmentation"
        if title:
            main_title += f" — {title}"
        plt.suptitle(main_title, fontsize=15, fontweight="bold", y=1.02)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        from PIL import Image as PILImage

        _img = PILImage.open(save_path)
        if _img.mode != "RGB":
            _rgb = _img.convert("RGB")
            _img.close()
            _rgb.save(save_path)
        else:
            _img.close()
        logger.debug(f"Prediction visualization saved: {save_path}")

    def visualize_sar_features(
        self,
        amplitude: np.ndarray,
        variance: np.ndarray,
        gradient: np.ndarray,
        save_path: str,
    ) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        channels = [amplitude, variance, gradient]
        titles = [
            "Channel 1: SAR Amplitude\nOil/look-alike: dark | Ships: bright",
            "Channel 2: Local Variance (9×9 window)\nOil: near-zero | Open sea: high",
            "Channel 3: Gradient Magnitude (Sobel)\nShips: sharp | Oil: diffuse",
        ]

        for ax, ch, title in zip(axes, channels, titles):
            im = ax.imshow(ch, cmap="viridis", vmin=0, vmax=1)
            ax.set_title(title, fontsize=11)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.suptitle(
            "Multi-Feature SAR Encoding for Oil Spill Discrimination",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"SAR feature visualization saved: {save_path}")

    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        save_path: str,
        normalize: bool = True,
    ) -> None:
        cm = confusion_matrix.astype(np.float64)

        if normalize:
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_display = np.where(row_sums > 0, cm / row_sums, 0.0)
            fmt = ".2%"
            title = "Normalized Confusion Matrix (Row %)"
        else:
            cm_display = cm
            fmt = "d"
            title = "Confusion Matrix (Pixel Count)"

        fig, ax = plt.subplots(figsize=(9, 8))
        im = ax.imshow(
            cm_display, interpolation="nearest", cmap="Blues", vmin=0, vmax=1
        )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        thresh = cm_display.max() / 2.0
        for i in range(cm_display.shape[0]):
            for j in range(cm_display.shape[1]):
                value = cm_display[i, j]
                text = f"{value:.2%}" if normalize else f"{int(value)}"

                is_critical = (
                    (i == 1 and j == 2)  
                    or (i == 2 and j == 1)  
                )
                color = (
                    "red" if is_critical else ("white" if value > thresh else "black")
                )
                weight = "bold" if is_critical else "normal"

                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    color=color,
                    fontweight=weight,
                    fontsize=10,
                )

        ax.set_xticks(range(NUM_CLASSES))
        ax.set_yticks(range(NUM_CLASSES))
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=11)
        ax.set_yticklabels(CLASS_NAMES, fontsize=11)
        ax.set_xlabel("Predicted Class", fontsize=12, fontweight="bold")
        ax.set_ylabel("True Class", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

        fig.text(
            0.5,
            -0.02,
            "Red cells: oil_spill ↔ look_alike confusion (safety-critical)",
            ha="center",
            fontsize=10,
            style="italic",
            color="red",
        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Confusion matrix saved: {save_path}")

    def plot_attention_weights(
        self,
        attention_weights: Dict[str, Any],
        sample_image: np.ndarray,
        save_path: str,
    ) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(sample_image, cmap="gray")
        axes[0].set_title("SAR Amplitude Input", fontsize=12)
        axes[0].axis("off")

        if "oil_attends_look" in attention_weights:
            attn_oil = attention_weights["oil_attends_look"]
            if isinstance(attn_oil, torch.Tensor):
                attn_oil = attn_oil.cpu().numpy()
            attn_oil = attn_oil.mean() if attn_oil.size > 1 else float(attn_oil)
            axes[1].text(
                0.5,
                0.5,
                f"Oil→Look Attention\nWeight: {attn_oil:.4f}",
                ha="center",
                va="center",
                fontsize=14,
                transform=axes[1].transAxes,
            )
        axes[1].set_title(
            "Oil Spill Query\nAttends to Look-alike Features", fontsize=12
        )
        axes[1].axis("off")

        if "look_attends_oil" in attention_weights:
            attn_look = attention_weights["look_attends_oil"]
            if isinstance(attn_look, torch.Tensor):
                attn_look = attn_look.cpu().numpy()
            attn_look = attn_look.mean() if attn_look.size > 1 else float(attn_look)
            axes[2].text(
                0.5,
                0.5,
                f"Look→Oil Attention\nWeight: {attn_look:.4f}",
                ha="center",
                va="center",
                fontsize=14,
                transform=axes[2].transAxes,
            )
        axes[2].set_title(
            "Look-alike Query\nAttends to Oil Spill Features", fontsize=12
        )
        axes[2].axis("off")

        plt.suptitle(
            "Confusion-Aware Cross-Query Attention Weights\n"
            "(Novel module: forces explicit oil spill vs. look-alike reasoning)",
            fontsize=13,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Attention weights visualization saved: {save_path}")

    def plot_training_curves(
        self,
        log_file: str,
        save_path: str,
        phase_boundaries: Optional[List[int]] = None,
    ) -> None:
        phase_boundaries = phase_boundaries or [50, 150]

        epochs = []
        train_losses = []
        val_mious = []
        oil_ious = []
        look_ious = []
        ship_ious = []

        log_path = Path(log_file)
        if not log_path.exists():
            logger.warning(f"Log file not found: {log_file}")
            return

        with open(log_path, "r") as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    step = record.get("step", 0)

                    if "train/loss_total" in record:
                        epochs.append(step)
                        train_losses.append(record["train/loss_total"])

                    val_mean_iou = record.get(
                        "val_full/mean_iou",
                        record.get("val/mean_iou", record.get("val_fast/mean_iou")),
                    )
                    if val_mean_iou is not None:
                        val_mious.append((step, val_mean_iou))

                    oil_key = (
                        "val_full/iou_oil_spill"
                        if "val_full/iou_oil_spill" in record
                        else (
                            "val/iou_oil_spill"
                            if "val/iou_oil_spill" in record
                            else "val/oil_spill_iou"
                        )
                    )
                    look_key = (
                        "val_full/iou_look_alike"
                        if "val_full/iou_look_alike" in record
                        else (
                            "val/iou_look_alike"
                            if "val/iou_look_alike" in record
                            else "val/look_alike_iou"
                        )
                    )
                    ship_key = (
                        "val_full/iou_ship"
                        if "val_full/iou_ship" in record
                        else ("val/iou_ship" if "val/iou_ship" in record else "val/ship_iou")
                    )
                    if oil_key in record:
                        oil_ious.append((step, record[oil_key]))
                    if look_key in record:
                        look_ious.append((step, record[look_key]))
                    if ship_key in record:
                        ship_ious.append((step, record[ship_key]))
                except Exception:
                    continue

        if not epochs:
            logger.warning("No training data found in log file.")
            return

        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        if train_losses:
            axes[0].plot(
                epochs, train_losses, "b-", linewidth=1.5, label="Train Loss", alpha=0.8
            )
        axes[0].set_ylabel("Loss", fontsize=12)
        axes[0].set_title("Training Loss", fontsize=13, fontweight="bold")
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        if val_mious:
            v_epochs, v_mious = zip(*val_mious)
            axes[1].plot(
                v_epochs,
                [m * 100 for m in v_mious],
                "g-o",
                linewidth=2,
                markersize=4,
                label="Val mIoU",
                alpha=0.9,
            )
        if oil_ious:
            o_epochs, o_vals = zip(*oil_ious)
            axes[1].plot(
                o_epochs,
                [v * 100 for v in o_vals],
                color="#00a7a7",
                linewidth=1.5,
                label="Oil IoU",
                alpha=0.8,
            )
        if look_ious:
            l_epochs, l_vals = zip(*look_ious)
            axes[1].plot(
                l_epochs,
                [v * 100 for v in l_vals],
                color="#c0392b",
                linewidth=1.5,
                label="Look-alike IoU",
                alpha=0.8,
            )
        if ship_ious:
            s_epochs, s_vals = zip(*ship_ious)
            axes[1].plot(
                s_epochs,
                [v * 100 for v in s_vals],
                color="#8e5a2b",
                linewidth=1.5,
                label="Ship IoU",
                alpha=0.8,
            )
        axes[1].set_ylabel("mIoU (%)", fontsize=12)
        axes[1].set_xlabel("Epoch", fontsize=12)
        axes[1].set_title("Validation mIoU", fontsize=13, fontweight="bold")
        axes[1].grid(True, alpha=0.3)

        axes[1].axhline(
            y=65.06,
            color="r",
            linestyle="--",
            alpha=0.7,
            label="DeepLabv3+ Baseline (65.06%)",
            linewidth=1.5,
        )

        axes[1].legend(fontsize=10)

        phase_labels = ["→ Phase 2\n(Ship Enriched)", "→ Phase 3\n(Confusion Penalty)"]
        colors = ["orange", "purple"]
        for i, (boundary, label, color) in enumerate(
            zip(phase_boundaries, phase_labels, colors)
        ):
            for ax in axes:
                ax.axvline(
                    x=boundary, color=color, linestyle="--", alpha=0.6, linewidth=1.5
                )
            axes[1].text(
                boundary + 1,
                axes[1].get_ylim()[0] + 2,
                label,
                fontsize=9,
                color=color,
                ha="left",
            )

        phase_regions = [
            (0, phase_boundaries[0], "Phase 1\nStandard"),
            (phase_boundaries[0], phase_boundaries[1], "Phase 2\nShip Enriched"),
            (
                phase_boundaries[1],
                max(epochs) if epochs else 300,
                "Phase 3\nConfusion Penalty",
            ),
        ]
        for start, end, label in phase_regions:
            mid = (start + end) / 2
            axes[0].text(
                mid,
                axes[0].get_ylim()[1] * 0.95,
                label,
                ha="center",
                va="top",
                fontsize=9,
                alpha=0.6,
                style="italic",
            )

        plt.suptitle(
            "Training Curves with Curriculum Phase Transitions",
            fontsize=14,
            fontweight="bold",
            y=1.01,
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Training curves saved: {save_path}")
