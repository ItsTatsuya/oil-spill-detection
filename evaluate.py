"""
Evaluation script for SegFormer-B2 oil spill detection.

Improvements over the original:
- Uses shared modules (model/metrics.py, model/prediction.py, utils.py, config.py)
- Adds pixel accuracy, frequency-weighted IoU, and Cohen's Kappa metrics
- Vectorised bootstrap CI (10–100× faster than original pixel-level loop)
- Predictor created once outside timing loop for accurate inference measurement
- Ship post-processing parameters driven by EvalConfig
- Proper exception logging instead of silent swallowing
"""

import os
import time
import logging
import argparse

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import silent_tf_import

tf = silent_tf_import()
from tensorflow.keras import mixed_precision  # type: ignore

policy = mixed_precision.global_policy()
print(f"Mixed precision policy: {policy.name}")

from data.data_loader import load_dataset
from config import EvalConfig, DataConfig
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

# Define baseline results for comparison
BASELINE_RESULTS = {
    'mean_iou': 65.06,
    'class_ious': {
        'Sea Surface': 96.43,
        'Oil Spill': 53.38,
        'Look-alike': 55.40,
        'Ship': 27.63,
        'Land': 92.44,
    },
}

CLASS_NAMES = ['Sea Surface', 'Oil Spill', 'Look-alike', 'Ship', 'Land']


# ---------------------------------------------------------------------------
# Ship post-processing
# ---------------------------------------------------------------------------
def apply_ship_enhancement(predictions, ecfg: EvalConfig, ship_class_idx=3):
    """Morphology-based ship detection refinement with configurable thresholds."""
    import cv2

    predictions_np = predictions.numpy()
    batch_size = predictions_np.shape[0]
    enhanced = np.copy(predictions_np)

    for i in range(batch_size):
        ship_probs = predictions_np[i, :, :, ship_class_idx]
        ship_mask = (ship_probs > ecfg.ship_probability_threshold).astype(np.uint8)

        if np.sum(ship_mask) == 0:
            continue

        kernel = np.ones((5, 5), np.uint8)
        opened = cv2.morphologyEx(ship_mask, cv2.MORPH_OPEN, kernel=np.ones((2, 2), np.uint8))
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)

        enhanced_mask = np.zeros_like(ship_mask)
        for j in range(1, num_labels):
            area = stats[j, cv2.CC_STAT_AREA]
            if area >= ecfg.ship_min_size:
                comp = (labels == j).astype(np.uint8)
                if area < 100:
                    iters = 2 if area < 30 else 1
                    comp = cv2.dilate(comp, kernel, iterations=iters)
                enhanced_mask = np.logical_or(enhanced_mask, comp)

        enhanced_mask = enhanced_mask.astype(np.uint8)

        ship_probs_enh = np.copy(ship_probs)
        ship_probs_enh[enhanced_mask == 1] *= ecfg.ship_boost_factor
        ship_probs_enh = np.clip(ship_probs_enh, 0, 1)
        enhanced[i, :, :, ship_class_idx] = ship_probs_enh

        # Suppress look-alike where ship is detected
        lookalike_idx = 2
        la_probs = enhanced[i, :, :, lookalike_idx]
        la_probs[enhanced_mask == 1] *= ecfg.ship_suppression_factor
        enhanced[i, :, :, lookalike_idx] = la_probs

        # Re-normalise
        s = np.sum(enhanced[i], axis=-1, keepdims=True)
        enhanced[i] /= s

    return tf.convert_to_tensor(enhanced, dtype=predictions.dtype)


# ---------------------------------------------------------------------------
# Inference timing
# ---------------------------------------------------------------------------
def measure_inference_time(model, num_runs=5, batch_size=1, multi_scale=False, use_tta=False):
    input_shape = model.input_shape[1:3]
    channels = model.input_shape[3]
    dummy = tf.random.normal((batch_size, *input_shape, channels))

    # Create predictor ONCE (outside loop)
    if multi_scale:
        predictor = MultiScalePredictor(
            model, scales=[0.75, 1.0, 1.25], batch_size=batch_size, use_tta=use_tta,
        )
    else:
        predictor = None

    def run_once(inp):
        if predictor is not None:
            return predictor.predict(inp)
        return model(inp, training=False)

    # --- Mixed precision ---
    mp_input = tf.cast(dummy, policy.compute_dtype)
    for _ in range(3):
        run_once(mp_input)
    t0 = time.time()
    for _ in range(num_runs):
        run_once(mp_input)
    mp_ms = (time.time() - t0) * 1000 / num_runs

    # --- FP32 ---
    fp32_input = tf.cast(dummy, tf.float32)
    orig_policy = mixed_precision.global_policy()
    mixed_precision.set_global_policy('float32')
    for _ in range(3):
        run_once(fp32_input)
    t0 = time.time()
    for _ in range(num_runs):
        run_once(fp32_input)
    fp32_ms = (time.time() - t0) * 1000 / num_runs
    mixed_precision.set_global_policy(orig_policy)

    return {
        'mixed_precision_ms': mp_ms,
        'float32_ms': fp32_ms,
        'speedup': fp32_ms / mp_ms if mp_ms > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def evaluate_model(
    model_path, test_dataset, ecfg: EvalConfig, batch_size=8,
    apply_ship_postprocessing=True, use_tta=True,
):
    from model.model import OilSpillSegformer, create_pretrained_weight_loader

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    if not os.path.exists(model_path):
        alt = 'segformer_b2_final.weights.h5'
        if os.path.exists(alt):
            model_path = alt

    orig_policy = mixed_precision.global_policy()
    if orig_policy.name != 'mixed_float16':
        mixed_precision.set_global_policy('mixed_float16')

    try:
        model = OilSpillSegformer(
            input_shape=(384, 384, 1), num_classes=5,
            drop_rate=0.0, use_cbam=False, pretrained_weights=None,
        )
        _ = model(tf.zeros((1, 384, 384, 1), dtype=tf.float32), training=False)
        try:
            model.load_weights(model_path)
        except Exception:
            weight_loader = create_pretrained_weight_loader()
            if not weight_loader(model, model_path):
                raise RuntimeError("All weight-loading methods failed")
        if orig_policy.name != 'mixed_float16':
            mixed_precision.set_global_policy(orig_policy)
    except Exception as e:
        logger.error("Model loading failed: %s", e)
        if orig_policy.name != 'mixed_float16':
            mixed_precision.set_global_policy(orig_policy)
        return None

    # Create predictor ONCE
    scales = ecfg.scales_with_tta if use_tta else ecfg.scales_no_tta
    predictor = MultiScalePredictor(
        model, scales=scales, batch_size=batch_size, use_tta=use_tta,
    )

    # Detect if dataset is already batched
    is_already_batched = False
    try:
        sample_images, _ = next(iter(test_dataset))
        if len(sample_images.shape) == 4 and sample_images.shape[0] > 1:
            is_already_batched = True
    except StopIteration:
        pass

    if is_already_batched:
        test_batches = test_dataset
    else:
        test_batches = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    all_true, all_preds = [], []
    per_image_true, per_image_pred = [], []

    for images, labels in tqdm(test_batches, desc="Evaluating", unit="batch"):
        try:
            preds = predictor.predict(images)
            if apply_ship_postprocessing:
                try:
                    preds = apply_ship_enhancement(preds, ecfg, ship_class_idx=3)
                except Exception as e:
                    logger.warning("Ship postprocessing failed: %s", e)
            all_true.append(labels)
            all_preds.append(preds)

            labels_np = tf.squeeze(labels, axis=-1).numpy().astype(np.int32)
            pred_np = tf.argmax(preds, axis=-1).numpy().astype(np.int32)
            for b in range(labels_np.shape[0]):
                per_image_true.append(labels_np[b])
                per_image_pred.append(pred_np[b])
        except Exception as e:
            logger.warning("Batch failed: %s", e)
            continue

    if not all_true:
        return None

    true_labels = tf.concat(all_true, axis=0)
    predictions = tf.concat(all_preds, axis=0)

    # Core metrics
    mean_iou, class_ious, confusion_mat = compute_iou(true_labels, predictions, num_classes=5)
    precision, recall, f1 = compute_precision_recall_f1(confusion_mat)

    # New metrics
    pixel_acc = compute_pixel_accuracy(confusion_mat)
    fwiou = compute_frequency_weighted_iou(confusion_mat)
    kappa = compute_cohens_kappa(confusion_mat)

    # Bootstrap CI (vectorised)
    print("Computing bootstrap confidence interval …")
    ci_low, ci_high = compute_bootstrap_ci(
        per_image_true, per_image_pred, num_classes=5,
        n_bootstrap=ecfg.bootstrap_iterations, confidence=ecfg.bootstrap_confidence,
    )

    # Inference timing
    inf_time = measure_inference_time(model, multi_scale=True, use_tta=use_tta)
    ss_time = measure_inference_time(model, multi_scale=False, use_tta=False)

    results = {
        'mean_iou': mean_iou * 100,
        'mean_iou_ci': (ci_low * 100, ci_high * 100),
        'pixel_accuracy': pixel_acc * 100,
        'frequency_weighted_iou': fwiou * 100,
        'cohens_kappa': kappa,
        'class_ious':      {CLASS_NAMES[i]: class_ious[i] * 100 for i in range(5)},
        'class_precision': {CLASS_NAMES[i]: precision[i]  * 100 for i in range(5)},
        'class_recall':    {CLASS_NAMES[i]: recall[i]     * 100 for i in range(5)},
        'class_f1':        {CLASS_NAMES[i]: f1[i]         * 100 for i in range(5)},
        'confusion_matrix': confusion_mat,
        'inference_time_ms': inf_time,
        'single_scale_time_ms': ss_time,
    }
    return results


# ---------------------------------------------------------------------------
# Plotting & saving
# ---------------------------------------------------------------------------
def plot_class_ious(results, title="Class-wise IoU Comparison"):
    plt.figure(figsize=(12, 6))
    class_names = list(next(iter(results.values()))['class_ious'].keys())
    x = np.arange(len(class_names))
    width = 0.8 / (len(results) + 1)

    for i, (name, res) in enumerate(results.items()):
        vals = [res['class_ious'][cn] for cn in class_names]
        plt.bar(x + i * width - 0.4 + width / 2, vals, width, label=name)

    bl = [BASELINE_RESULTS['class_ious'][cn] for cn in class_names]
    plt.bar(x + len(results) * width - 0.4 + width / 2, bl, width,
            label='Baseline', color='gray', alpha=0.7)

    plt.xlabel('Classes')
    plt.ylabel('IoU (%)')
    plt.title(title)
    plt.xticks(x, class_names, rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for i, (_, res) in enumerate(results.items()):
        vals = [res['class_ious'][cn] for cn in class_names]
        for j, v in enumerate(vals):
            plt.text(j + i * width - 0.4 + width / 2, v + 0.5, f"{v:.1f}",
                     ha='center', fontsize=8)
    for j, v in enumerate(bl):
        plt.text(j + len(results) * width - 0.4 + width / 2, v + 0.5, f"{v:.1f}",
                 ha='center', fontsize=8)

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
        bl = " | ".join(f"{BASELINE_RESULTS['class_ious'][cn]:.2f}" for cn in CLASS_NAMES)
        f.write(f"| Baseline | {bl} |\n")

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
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    ecfg = EvalConfig()

    parser = argparse.ArgumentParser(description='Evaluate oil spill detection models')
    parser.add_argument('--batch_size', type=int, default=ecfg.batch_size)
    parser.add_argument('--disable_ship_postprocessing', action='store_true')
    parser.add_argument('--model_path', type=str, default=ecfg.model_path)
    parser.add_argument('--disable_tta', action='store_true')
    parser.add_argument('--tta_num_augs', type=int, default=ecfg.tta_num_augmentations)
    args = parser.parse_args()

    test_dataset, _, _ = load_dataset(data_dir='dataset', split='test', batch_size=args.batch_size)

    use_tta = not args.disable_tta
    apply_ship = not args.disable_ship_postprocessing

    model_results = evaluate_model(
        args.model_path, test_dataset, ecfg=ecfg,
        batch_size=args.batch_size,
        apply_ship_postprocessing=apply_ship,
        use_tta=use_tta,
    )

    if model_results is None:
        print("Evaluation failed.")
        return

    model_name = 'SegFormer-B2'
    results = {model_name: model_results}

    ci_lo, ci_hi = model_results['mean_iou_ci']
    print(f"\nResults for {model_name}:")
    print(f"Mean IoU:  {model_results['mean_iou']:.2f}%  (95% CI: {ci_lo:.2f}% – {ci_hi:.2f}%)")
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
    print(f"\nSingle-scale: {model_results['single_scale_time_ms']['mixed_precision_ms']:.2f} ms")
    print(f"Multi-scale:  {model_results['inference_time_ms']['mixed_precision_ms']:.2f} ms")

    plot_class_ious(results)
    save_results(results)


if __name__ == "__main__":
    main()
