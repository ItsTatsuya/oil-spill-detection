"""
Evaluation script for SegFormer-B2 oil spill detection — PyTorch implementation.

Improvements over the original:
- Uses shared modules (model/metrics.py, model/prediction.py, utils.py, config.py)
- Pixel accuracy, frequency-weighted IoU, Cohen's Kappa metrics
- Vectorised bootstrap CI (np.bincount — 10–100× faster)
- Predictor created once for accurate inference timing
- Ship post-processing driven by EvalConfig
- AMP (autocast) inference for speed
"""

import os
import time
import logging
import argparse

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.amp import autocast

from config import EvalConfig, DataConfig
from data.data_loader import load_dataset
from model.prediction import MultiScalePredictor
from model.metrics import (
    compute_iou,
    compute_precision_recall_f1,
    compute_bootstrap_ci,
    compute_pixel_accuracy,
    compute_frequency_weighted_iou,
    compute_cohens_kappa,
)

logger = logging.getLogger('oil_spill')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(levelname)s  %(message)s')

BASELINE_RESULTS = {
    'mean_iou': 65.06,
    'class_ious': {
        'Sea Surface': 96.43, 'Oil Spill': 53.38,
        'Look-alike': 55.40, 'Ship': 27.63, 'Land': 92.44,
    },
}
CLASS_NAMES = ['Sea Surface', 'Oil Spill', 'Look-alike', 'Ship', 'Land']


# ---------------------------------------------------------------------------
# Ship post-processing
# ---------------------------------------------------------------------------
def apply_ship_enhancement(predictions: np.ndarray, ecfg: EvalConfig,
                            ship_class_idx: int = 3) -> np.ndarray:
    """Morphology-based ship detection refinement.

    Parameters
    ----------
    predictions : np.ndarray  [B, num_classes, H, W]  probabilities (float32)
    """
    enhanced = predictions.copy()
    B = predictions.shape[0]

    for i in range(B):
        ship_probs = predictions[i, ship_class_idx]          # [H, W]
        ship_mask  = (ship_probs > ecfg.ship_probability_threshold).astype(np.uint8)

        if np.sum(ship_mask) == 0:
            continue

        kernel = np.ones((5, 5), np.uint8)
        opened = cv2.morphologyEx(ship_mask, cv2.MORPH_OPEN,
                                  kernel=np.ones((2, 2), np.uint8))
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

        num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(
            closed, connectivity=8
        )
        enhanced_mask = np.zeros_like(ship_mask)
        for j in range(1, num_labels):
            area = stats[j, cv2.CC_STAT_AREA]
            if area >= ecfg.ship_min_size:
                comp = (labels_im == j).astype(np.uint8)
                if area < 100:
                    iters = 2 if area < 30 else 1
                    comp  = cv2.dilate(comp, kernel, iterations=iters)
                enhanced_mask = np.logical_or(enhanced_mask, comp)

        enhanced_mask = enhanced_mask.astype(np.uint8)

        # Boost ship probabilities
        sp = enhanced[i, ship_class_idx].copy()
        sp[enhanced_mask == 1] = np.clip(
            sp[enhanced_mask == 1] * ecfg.ship_boost_factor, 0.0, 1.0
        )
        enhanced[i, ship_class_idx] = sp

        # Suppress look-alike where ship is detected
        la = enhanced[i, 2].copy()
        la[enhanced_mask == 1] *= ecfg.ship_suppression_factor
        enhanced[i, 2] = la

        # Re-normalise along class axis
        s = enhanced[i].sum(axis=0, keepdims=True)
        s = np.where(s > 0, s, 1.0)
        enhanced[i] /= s

    return enhanced


# ---------------------------------------------------------------------------
# Inference timing
# ---------------------------------------------------------------------------
def measure_inference_time(model, device, num_runs: int = 5, batch_size: int = 1,
                            multi_scale: bool = False, use_tta: bool = False):
    """Warm-up then time inference in both AMP and FP32 modes."""
    use_amp = torch.cuda.is_available()
    h = model.expected_height
    w = model.expected_width
    c = model.expected_channels
    dummy = torch.randn(batch_size, c, h, w, device=device)

    predictor = MultiScalePredictor(
        model, scales=[0.75, 1.0, 1.25], batch_size=batch_size, use_tta=use_tta,
        device=device,
    ) if multi_scale else None

    def run_once(inp):
        if predictor is not None:
            return predictor.predict(inp)
        with torch.inference_mode():
            return model(inp)

    # AMP timing
    for _ in range(3):
        with autocast('cuda', enabled=use_amp):
            run_once(dummy)
    torch.cuda.synchronize() if use_amp else None
    t0 = time.time()
    for _ in range(num_runs):
        with autocast('cuda', enabled=use_amp):
            run_once(dummy)
    torch.cuda.synchronize() if use_amp else None
    mp_ms = (time.time() - t0) * 1000 / num_runs

    # FP32 timing
    dummy_fp32 = dummy.float()
    for _ in range(3):
        run_once(dummy_fp32)
    torch.cuda.synchronize() if use_amp else None
    t0 = time.time()
    for _ in range(num_runs):
        run_once(dummy_fp32)
    torch.cuda.synchronize() if use_amp else None
    fp32_ms = (time.time() - t0) * 1000 / num_runs

    return {
        'mixed_precision_ms': mp_ms,
        'float32_ms': fp32_ms,
        'speedup': fp32_ms / mp_ms if mp_ms > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def evaluate_model(
    model_path: str,
    test_loader,
    ecfg: EvalConfig,
    batch_size: int = 8,
    apply_ship_postprocessing: bool = True,
    use_tta: bool = True,
    device=None,
):
    """Full evaluation. Returns a results dict or None on failure."""
    from model.model import OilSpillSegformer

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- Load model --------------------------------------------------------
    if not os.path.exists(model_path):
        alt = 'segformer_b2_final.pt'
        if os.path.exists(alt):
            model_path = alt
        else:
            logger.error("Model not found at %s", model_path)
            return None

    try:
        model = OilSpillSegformer(
            input_shape=(384, 384, 1),
            num_classes=5,
            drop_rate=0.0,
            use_cbam=False,
            pretrained_weights=None,
        ).to(device)

        state = torch.load(model_path, map_location=device)
        # Support both raw state_dict and checkpoint dicts
        if 'model' in state:
            state = state['model']
        model.load_state_dict(state, strict=True)
        model.eval()
        print(f"Loaded model from {model_path}")
    except Exception as e:
        logger.error("Model loading failed: %s", e)
        return None

    # ---- Predictor ---------------------------------------------------------
    scales    = ecfg.scales_with_tta if use_tta else ecfg.scales_no_tta
    predictor = MultiScalePredictor(
        model, scales=scales, batch_size=batch_size,
        use_tta=use_tta, device=device,
    )

    use_amp = torch.cuda.is_available()
    all_preds_np, all_true_np          = [], []
    per_image_true, per_image_pred     = [], []

    for images, labels in tqdm(test_loader, desc="Evaluating", unit="batch"):
        images  = images.to(device)
        labels  = labels.to(device)

        try:
            with autocast('cuda', enabled=use_amp):
                logits = predictor.predict(images)   # [B, C, H, W]

            # predictor.predict() returns log-probs — use exp() to recover probabilities
            probs_np = torch.exp(logits.float()).cpu().numpy()           # [B, C, H, W]
            true_np  = labels.cpu().numpy()                              # [B, H, W]

            if apply_ship_postprocessing:
                try:
                    probs_np = apply_ship_enhancement(probs_np, ecfg)
                except Exception as e:
                    logger.warning("Ship postprocessing failed: %s", e)

            all_preds_np.append(probs_np)
            all_true_np.append(true_np)

            preds_cls = probs_np.argmax(axis=1)   # [B, H, W]
            for b in range(true_np.shape[0]):
                per_image_true.append(true_np[b].astype(np.int32))
                per_image_pred.append(preds_cls[b].astype(np.int32))

        except Exception as e:
            logger.warning("Batch failed: %s", e)
            continue

    if not all_preds_np:
        return None

    all_preds_np = np.concatenate(all_preds_np, axis=0)   # [N, C, H, W]
    all_true_np  = np.concatenate(all_true_np,  axis=0)   # [N, H, W]

    # ---- Core metrics ------------------------------------------------------
    mean_iou, class_ious, confusion_mat = compute_iou(all_true_np, all_preds_np)
    precision, recall, f1               = compute_precision_recall_f1(confusion_mat)
    pixel_acc = compute_pixel_accuracy(confusion_mat)
    fwiou     = compute_frequency_weighted_iou(confusion_mat)
    kappa     = compute_cohens_kappa(confusion_mat)

    # ---- Bootstrap CI ------------------------------------------------------
    print("Computing bootstrap confidence interval …")
    ci_low, ci_high = compute_bootstrap_ci(
        per_image_true, per_image_pred,
        num_classes=5,
        n_bootstrap=ecfg.bootstrap_iterations,
        confidence=ecfg.bootstrap_confidence,
    )

    # ---- Inference timing --------------------------------------------------
    inf_time = measure_inference_time(model, device, multi_scale=True,  use_tta=use_tta)
    ss_time  = measure_inference_time(model, device, multi_scale=False, use_tta=False)

    results = {
        'mean_iou':               mean_iou * 100,
        'mean_iou_ci':            (ci_low * 100, ci_high * 100),
        'pixel_accuracy':         pixel_acc * 100,
        'frequency_weighted_iou': fwiou * 100,
        'cohens_kappa':           kappa,
        'class_ious':      {CLASS_NAMES[i]: class_ious[i] * 100 for i in range(5)},
        'class_precision': {CLASS_NAMES[i]: precision[i]  * 100 for i in range(5)},
        'class_recall':    {CLASS_NAMES[i]: recall[i]     * 100 for i in range(5)},
        'class_f1':        {CLASS_NAMES[i]: f1[i]         * 100 for i in range(5)},
        'confusion_matrix':      confusion_mat,
        'inference_time_ms':     inf_time,
        'single_scale_time_ms':  ss_time,
    }
    return results


# ---------------------------------------------------------------------------
# Plotting & saving
# ---------------------------------------------------------------------------
def plot_class_ious(results, title="Class-wise IoU Comparison"):
    plt.figure(figsize=(12, 6))
    x     = np.arange(len(CLASS_NAMES))
    n     = len(results)
    width = 0.8 / (n + 1)

    for i, (name, res) in enumerate(results.items()):
        vals = [res['class_ious'][cn] for cn in CLASS_NAMES]
        plt.bar(x + i * width - 0.4 + width / 2, vals, width, label=name)

    bl = [BASELINE_RESULTS['class_ious'][cn] for cn in CLASS_NAMES]
    plt.bar(x + n * width - 0.4 + width / 2, bl, width,
            label='Baseline', color='gray', alpha=0.7)

    plt.xlabel('Classes')
    plt.ylabel('IoU (%)')
    plt.title(title)
    plt.xticks(x, CLASS_NAMES, rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('class_ious_comparison.png', dpi=300)
    plt.close()


def save_results(results, output_file="evaluation_results.md"):
    with open(output_file, 'w') as f:
        f.write("# Oil Spill Detection Model Evaluation Results\n\n")
        f.write(f"Evaluation Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Summary Metrics\n\n")
        f.write("| Model | Mean IoU (%) | 95% CI | Pixel Acc (%) | FWIoU (%) | Kappa |\n")
        f.write("|-------|-------------|--------|---------------|-----------|-------|\n")
        for name, res in results.items():
            ci = res.get('mean_iou_ci', (0, 0))
            f.write(f"| {name} | {res['mean_iou']:.2f} | "
                    f"{ci[0]:.2f}–{ci[1]:.2f} | "
                    f"{res.get('pixel_accuracy', 0):.2f} | "
                    f"{res.get('frequency_weighted_iou', 0):.2f} | "
                    f"{res.get('cohens_kappa', 0):.4f} |\n")
        f.write(f"| Baseline | {BASELINE_RESULTS['mean_iou']:.2f} | — | — | — | — |\n")

        f.write("\n## Class-wise IoU\n\n")
        header_classes = " | ".join(CLASS_NAMES)
        f.write(f"| Model | {header_classes} |\n")
        f.write("|-------" + "|------" * 5 + "|\n")
        for name, res in results.items():
            vals = " | ".join(f"{res['class_ious'][cn]:.2f}" for cn in CLASS_NAMES)
            f.write(f"| {name} | {vals} |\n")
        bl_vals = " | ".join(f"{BASELINE_RESULTS['class_ious'][cn]:.2f}"
                             for cn in CLASS_NAMES)
        f.write(f"| Baseline | {bl_vals} |\n")

        f.write("\n## Class-wise Precision / Recall / F1\n\n")
        f.write("| Class | Precision (%) | Recall (%) | F1 (%) |\n")
        f.write("|-------|--------------|-----------|--------|\n")
        for name, res in results.items():
            for cn in CLASS_NAMES:
                f.write(f"| {cn} | "
                        f"{res['class_precision'][cn]:.2f} | "
                        f"{res['class_recall'][cn]:.2f} | "
                        f"{res['class_f1'][cn]:.2f} |\n")

        f.write("\n## Inference Time\n\n")
        f.write("| Model | Single-Scale (ms) | Multi-Scale (ms) | Speedup |\n")
        f.write("|-------|-------------------|------------------|--------|\n")
        for name, res in results.items():
            f.write(f"| {name} | "
                    f"{res['single_scale_time_ms']['mixed_precision_ms']:.2f} | "
                    f"{res['inference_time_ms']['mixed_precision_ms']:.2f} | "
                    f"{res['inference_time_ms']['speedup']:.2f}× |\n")

        f.write("\n![Class-wise IoU Comparison](class_ious_comparison.png)\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    ecfg   = EvalConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser(description='Evaluate oil spill detection models')
    parser.add_argument('--batch_size', type=int, default=ecfg.batch_size)
    parser.add_argument('--disable_ship_postprocessing', action='store_true')
    parser.add_argument('--model_path', type=str, default=ecfg.model_path)
    parser.add_argument('--disable_tta', action='store_true')
    parser.add_argument('--tta_num_augs', type=int, default=ecfg.tta_num_augmentations)
    args = parser.parse_args()

    dcfg = DataConfig()
    test_loader, _, _ = load_dataset(
        data_dir=dcfg.data_dir, split='test',
        batch_size=args.batch_size,
        val_split=dcfg.val_split,
        num_workers=dcfg.num_workers,
    )

    use_tta    = not args.disable_tta
    apply_ship = not args.disable_ship_postprocessing

    model_results = evaluate_model(
        args.model_path, test_loader, ecfg=ecfg,
        batch_size=args.batch_size,
        apply_ship_postprocessing=apply_ship,
        use_tta=use_tta,
        device=device,
    )

    if model_results is None:
        print("Evaluation failed.")
        return

    model_name = 'SegFormer-B2'
    results    = {model_name: model_results}

    ci_lo, ci_hi = model_results['mean_iou_ci']
    print(f"\nResults for {model_name}:")
    print(f"Mean IoU:  {model_results['mean_iou']:.2f}%  "
          f"(95% CI: {ci_lo:.2f}% – {ci_hi:.2f}%)")
    print(f"Pixel Acc: {model_results.get('pixel_accuracy', 0):.2f}%")
    print(f"FWIoU:     {model_results.get('frequency_weighted_iou', 0):.2f}%")
    print(f"Kappa:     {model_results.get('cohens_kappa', 0):.4f}")
    print()
    print(f"{'Class':<15} {'IoU':>7} {'Precision':>11} {'Recall':>9} {'F1':>7}")
    print("-" * 55)
    for cn in CLASS_NAMES:
        print(f"{cn:<15} "
              f"{model_results['class_ious'][cn]:>6.2f}% "
              f"{model_results['class_precision'][cn]:>10.2f}% "
              f"{model_results['class_recall'][cn]:>8.2f}% "
              f"{model_results['class_f1'][cn]:>6.2f}%")
    print(f"\nSingle-scale: "
          f"{model_results['single_scale_time_ms']['mixed_precision_ms']:.2f} ms")
    print(f"Multi-scale:  "
          f"{model_results['inference_time_ms']['mixed_precision_ms']:.2f} ms")

    plot_class_ious(results)
    save_results(results)


if __name__ == "__main__":
    main()
