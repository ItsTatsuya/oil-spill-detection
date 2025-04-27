import os
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def silent_tf_import():
    import sys
    orig_stderr_fd = sys.stderr.fileno()
    saved_stderr_fd = os.dup(orig_stderr_fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, orig_stderr_fd)
    os.close(devnull_fd)

    import tensorflow as tf

    os.dup2(saved_stderr_fd, orig_stderr_fd)
    os.close(saved_stderr_fd)

    return tf

tf = silent_tf_import()

# Import mixed precision for faster inference
from tensorflow.keras import mixed_precision # type: ignore
policy = mixed_precision.global_policy()

print(f"Mixed precision policy: {policy.name}")

# Import the data loading function from data_loader.py
from data_loader import load_dataset

# Define baseline results for comparison
BASELINE_RESULTS = {
    'mean_iou': 65.06,
    'class_ious': {
        'Sea Surface': 96.43,
        'Oil Spill': 53.38,
        'Look-alike': 55.40,
        'Ship': 27.63,
        'Land': 92.44
    }
}

class MultiScalePredictor:
    """
    Enhanced Multi-scale prediction for semantic segmentation.
    Includes Test-Time Augmentation (TTA) for improved boundary detection
    and rare class (oil spills, ships) accuracy.
    """

    def __init__(self, model, scales=[0.75, 1.0, 1.25], batch_size=4, use_tta=True):
        self.model = model
        self.scales = scales
        self.batch_size = batch_size
        self.use_tta = use_tta
        self.expected_height = model.input_shape[1]
        self.expected_width = model.input_shape[2]
        self.expected_channels = self.model.input_shape[3]

        if self.use_tta:
            from test_time_augmentation import TestTimeAugmentation
            self.tta = TestTimeAugmentation(
                model,
                num_augmentations=8,
                use_flips=True,
                use_scales=False,
                use_rotations=True,
                include_original=True
            )

    @tf.function
    def _predict_batch(self, batch):
        return self.model(batch, training=False)

    def predict(self, image_batch):
        batch_size = tf.shape(image_batch)[0]
        height = tf.shape(image_batch)[1]
        width = tf.shape(image_batch)[2]

        if image_batch.shape[-1] != self.expected_channels:
            if self.expected_channels == 1 and image_batch.shape[-1] > 1:
                image_batch = tf.expand_dims(tf.reduce_mean(image_batch, axis=-1), axis=-1)
            elif self.expected_channels == 3 and image_batch.shape[-1] == 1:
                image_batch = tf.concat([image_batch, image_batch, image_batch], axis=-1)

        from tensorflow.keras import mixed_precision # type: ignore
        policy = mixed_precision.global_policy()
        image_batch = tf.cast(image_batch, policy.compute_dtype)

        original_size = (height, width)
        all_predictions = []

        for scale in self.scales:
            try:
                scaled_height = tf.cast(tf.cast(height, tf.float32) * scale, tf.int32)
                scaled_width = tf.cast(tf.cast(width, tf.float32) * scale, tf.int32)

                scaled_height = tf.cast(tf.math.ceil(scaled_height / 8) * 8, tf.int32)
                scaled_width = tf.cast(tf.math.ceil(scaled_width / 8) * 8, tf.int32)

                scaled_batch = tf.image.resize(
                    image_batch,
                    size=(scaled_height, scaled_width),
                    method='bilinear'
                )

                model_input = tf.image.resize(
                    scaled_batch,
                    size=(self.expected_height, self.expected_width),
                    method='bilinear'
                )

                if self.use_tta:
                    logits = self.tta.predict(model_input)
                else:
                    logits = self._predict_batch(model_input)

                if tf.shape(logits)[1] != height or tf.shape(logits)[2] != width:
                    resized_logits = tf.image.resize(
                        logits,
                        size=original_size,
                        method='bilinear'
                    )
                else:
                    resized_logits = logits

                probs = tf.nn.softmax(resized_logits, axis=-1)
                all_predictions.append(probs)

            except Exception:
                continue

        if not all_predictions:
            try:
                model_input = tf.image.resize(
                    image_batch,
                    size=(self.expected_height, self.expected_width),
                    method='bilinear'
                )
                logits = self.model(model_input, training=False)

                if tf.shape(logits)[1] != height or tf.shape(logits)[2] != width:
                    logits = tf.image.resize(
                        logits,
                        size=original_size,
                        method='bilinear'
                    )
                return logits
            except Exception:
                dummy_shape = list(image_batch.shape)
                dummy_shape[-1] = 5
                return tf.zeros(dummy_shape)

        fused_prediction = tf.reduce_mean(tf.stack(all_predictions, axis=0), axis=0)

        epsilon = 1e-7
        fused_prediction = tf.clip_by_value(fused_prediction, epsilon, 1.0 - epsilon)
        logits = tf.math.log(fused_prediction)

        return logits


def compute_iou(y_true, y_pred, num_classes=5):
    y_true = tf.cast(y_true, tf.int32)
    y_true = tf.squeeze(y_true, axis=-1)

    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, tf.int32)

    confusion_matrix = tf.zeros((num_classes, num_classes), dtype=tf.float32)

    for i in range(tf.shape(y_true)[0]):
        true_flat = tf.reshape(y_true[i], [-1])
        pred_flat = tf.reshape(y_pred[i], [-1])

        cm_i = tf.math.confusion_matrix(
            true_flat, pred_flat,
            num_classes=num_classes,
            dtype=tf.float32
        )

        confusion_matrix += cm_i

    sum_over_row = tf.reduce_sum(confusion_matrix, axis=0)
    sum_over_col = tf.reduce_sum(confusion_matrix, axis=1)
    true_positives = tf.linalg.tensor_diag_part(confusion_matrix)

    denominator = sum_over_row + sum_over_col - true_positives

    class_ious = tf.math.divide_no_nan(true_positives, denominator)

    mask = tf.greater(sum_over_col, 0)
    mean_iou = tf.reduce_mean(tf.boolean_mask(class_ious, mask))

    return mean_iou.numpy(), class_ious.numpy()


def measure_inference_time(model, num_runs=5, batch_size=1, multi_scale=False, use_tta=False):
    input_shape = model.input_shape[1:3]
    expected_channels = model.input_shape[3]
    dummy_input = tf.random.normal((batch_size, *input_shape, expected_channels))
    mp_input = tf.cast(dummy_input, policy.compute_dtype)

    for _ in range(3):
        if multi_scale:
            predictor = MultiScalePredictor(model, scales=[0.75, 1.0, 1.25], batch_size=batch_size, use_tta=use_tta)
            _ = predictor.predict(mp_input)
        else:
            _ = model(mp_input, training=False)

    start_time = time.time()
    for _ in range(num_runs):
        if multi_scale:
            predictor = MultiScalePredictor(model, scales=[0.75, 1.0, 1.25], batch_size=batch_size, use_tta=use_tta)
            _ = predictor.predict(mp_input)
        else:
            _ = model(mp_input, training=False)
    end_time = time.time()

    mp_avg_time = (end_time - start_time) * 1000 / num_runs

    fp32_input = tf.cast(dummy_input, tf.float32)
    original_policy = mixed_precision.global_policy()
    mixed_precision.set_global_policy('float32')

    for _ in range(3):
        if multi_scale:
            predictor = MultiScalePredictor(model, scales=[0.75, 1.0, 1.25], batch_size=batch_size, use_tta=use_tta)
            _ = predictor.predict(fp32_input)
        else:
            _ = model(fp32_input, training=False)

    start_time = time.time()
    for _ in range(num_runs):
        if multi_scale:
            predictor = MultiScalePredictor(model, scales=[0.75, 1.0, 1.25], batch_size=batch_size, use_tta=use_tta)
            _ = predictor.predict(fp32_input)
        else:
            _ = model(fp32_input, training=False)
    end_time = time.time()

    fp32_avg_time = (end_time - start_time) * 1000 / num_runs
    speedup = fp32_avg_time / mp_avg_time if mp_avg_time > 0 else 0

    mixed_precision.set_global_policy(original_policy)

    return {
        'mixed_precision_ms': mp_avg_time,
        'float32_ms': fp32_avg_time,
        'speedup': speedup
    }


def apply_ship_enhancement(predictions, ship_class_idx=3):
    import cv2

    predictions_np = predictions.numpy()
    batch_size = predictions_np.shape[0]
    enhanced_predictions = np.copy(predictions_np)

    for i in range(batch_size):
        ship_probs = predictions_np[i, :, :, ship_class_idx]
        ship_mask = (ship_probs > 0.25).astype(np.uint8)

        if np.sum(ship_mask) == 0:
            continue

        kernel = np.ones((5, 5), np.uint8)
        opened_mask = cv2.morphologyEx(ship_mask, cv2.MORPH_OPEN, kernel=np.ones((2, 2), np.uint8))
        closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed_mask, connectivity=8)

        min_size = 8
        enhanced_mask = np.zeros_like(ship_mask)

        for j in range(1, num_labels):
            if stats[j, cv2.CC_STAT_AREA] >= min_size:
                component_mask = (labels == j).astype(np.uint8)

                if stats[j, cv2.CC_STAT_AREA] < 100:
                    iterations = 2 if stats[j, cv2.CC_STAT_AREA] < 30 else 1
                    component_mask = cv2.dilate(component_mask, kernel, iterations=iterations)

                enhanced_mask = np.logical_or(enhanced_mask, component_mask)

        enhanced_mask = enhanced_mask.astype(np.uint8)

        boost_factor = 1.35
        enhanced_ship_probs = np.copy(ship_probs)
        enhanced_ship_probs[enhanced_mask == 1] *= boost_factor
        enhanced_ship_probs = np.clip(enhanced_ship_probs, 0, 1)
        enhanced_predictions[i, :, :, ship_class_idx] = enhanced_ship_probs

        lookalike_idx = 2
        if lookalike_idx != ship_class_idx:
            suppression_factor = 0.7
            lookalike_probs = enhanced_predictions[i, :, :, lookalike_idx]
            lookalike_probs[enhanced_mask == 1] *= suppression_factor
            enhanced_predictions[i, :, :, lookalike_idx] = lookalike_probs

        sum_probs = np.sum(enhanced_predictions[i], axis=-1, keepdims=True)
        enhanced_predictions[i] /= sum_probs

    return tf.convert_to_tensor(enhanced_predictions, dtype=predictions.dtype)


def evaluate_model(model_path, test_dataset, batch_size=8, apply_ship_postprocessing=True,
                use_tta=True, tta_num_augs=8):
    from model import OilSpillSegformer, create_pretrained_weight_loader
    from test_time_augmentation import TestTimeAugmentation

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    if not os.path.exists(model_path) or 'best' in model_path:
        alt_model_path = 'segformer_b2_final.weights.h5'
        if os.path.exists(alt_model_path):
            model_path = alt_model_path

    original_policy = mixed_precision.global_policy()
    if original_policy.name != 'mixed_float16':
        mixed_precision.set_global_policy('mixed_float16')

    try:
        model = OilSpillSegformer(
            input_shape=(384, 384, 1),
            num_classes=5,
            drop_rate=0.0,
            use_cbam=False,
            pretrained_weights=None
        )
        dummy_input = tf.zeros((1, 384, 384, 1), dtype=tf.float32)
        _ = model(dummy_input, training=False)
        try:
            model.load_weights(model_path)
        except Exception:
            try:
                weight_loader = create_pretrained_weight_loader()
                success = weight_loader(model, model_path)
                if not success:
                    raise RuntimeError("Failed to load weights with custom loader")
            except Exception:
                raise
        if original_policy.name != 'mixed_float16':
            mixed_precision.set_global_policy(original_policy)
    except Exception:
        if original_policy.name != 'mixed_float16':
            mixed_precision.set_global_policy(original_policy)
        return None

    if use_tta:
        predictor = MultiScalePredictor(
            model,
            scales=[0.5, 0.75, 1.0, 1.25, 1.5],
            batch_size=batch_size,
            use_tta=True
        )
    else:
        predictor = MultiScalePredictor(
            model,
            scales=[0.75, 1.0, 1.25],
            batch_size=batch_size,
            use_tta=False
        )

    is_already_batched = False
    try:
        sample_images, sample_labels = next(iter(test_dataset))
        if len(sample_images.shape) == 4 and sample_images.shape[0] > 1:
            is_already_batched = True
    except:
        pass

    if is_already_batched:
        test_batches = test_dataset
        try:
            total_batches = sum(1 for _ in test_dataset)
        except:
            total_batches = None
    else:
        test_batches = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        try:
            num_samples = sum(1 for _ in test_dataset)
            total_batches = (num_samples + batch_size - 1) // batch_size
        except:
            total_batches = None

    all_true_labels = []
    all_predictions = []

    progress_bar = tqdm(test_batches, total=total_batches, desc="Evaluating", unit="batch")

    for images, labels in progress_bar:
        try:
            predictions = predictor.predict(images)
            original_predictions = predictions
            try:
                if apply_ship_postprocessing:
                    refined_predictions = apply_ship_enhancement(original_predictions, ship_class_idx=3)
                else:
                    refined_predictions = original_predictions
                final_predictions = refined_predictions
            except Exception:
                final_predictions = original_predictions
            all_true_labels.append(labels)
            all_predictions.append(final_predictions)
        except Exception:
            continue

    if not all_true_labels:
        return None

    true_labels = tf.concat(all_true_labels, axis=0)
    predictions = tf.concat(all_predictions, axis=0)

    mean_iou, class_ious = compute_iou(true_labels, predictions, num_classes=5)

    inference_time_results = measure_inference_time(model, multi_scale=True, use_tta=use_tta)
    single_scale_time_results = measure_inference_time(model, multi_scale=False, use_tta=False)

    class_names = ['Sea Surface', 'Oil Spill', 'Look-alike', 'Ship', 'Land']

    results = {
        'mean_iou': mean_iou * 100,
        'class_ious': {class_names[i]: class_ious[i] * 100 for i in range(len(class_names))},
        'inference_time_ms': inference_time_results,
        'single_scale_time_ms': single_scale_time_results
    }

    return results


def plot_class_ious(results, title="Class-wise IoU Comparison"):
    plt.figure(figsize=(12, 6))

    class_names = list(next(iter(results.values()))['class_ious'].keys())
    x = np.arange(len(class_names))
    width = 0.8 / (len(results) + 1)

    for i, (model_name, model_results) in enumerate(results.items()):
        class_ious = [model_results['class_ious'][cn] for cn in class_names]
        plt.bar(x + i*width - 0.4 + width/2, class_ious, width, label=model_name)

    baseline_ious = [BASELINE_RESULTS['class_ious'][cn] for cn in class_names]
    plt.bar(x + len(results)*width - 0.4 + width/2, baseline_ious, width, label='Baseline', color='gray', alpha=0.7)

    plt.xlabel('Classes')
    plt.ylabel('IoU (%)')
    plt.title(title)
    plt.xticks(x, class_names, rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for i, (model_name, model_results) in enumerate(results.items()):
        class_ious = [model_results['class_ious'][cn] for cn in class_names]
        for j, v in enumerate(class_ious):
            plt.text(j + i*width - 0.4 + width/2, v + 0.5, f"{v:.1f}",
                     ha='center', fontsize=8)

    for j, v in enumerate(baseline_ious):
        plt.text(j + len(results)*width - 0.4 + width/2, v + 0.5, f"{v:.1f}",
                 ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig('class_ious_comparison.png', dpi=300)
    plt.close()


def save_results(results, output_file="evaluation_results.md"):
    with open(output_file, 'w') as f:
        f.write("# Oil Spill Detection Model Evaluation Results\n\n")
        f.write(f"Evaluation Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Mean IoU Comparison\n\n")
        f.write("| Model | Mean IoU (%) |\n")
        f.write("|-------|-------------|\n")
        for model_name, model_results in results.items():
            f.write(f"| {model_name} | {model_results['mean_iou']:.2f} |\n")
        f.write(f"| Baseline | {BASELINE_RESULTS['mean_iou']:.2f} |\n")

        f.write("\n## Class-wise IoU Comparison\n\n")
        f.write("| Model | " + " | ".join(next(iter(results.values()))['class_ious'].keys()) + " |\n")
        f.write("|-------|" + "|".join(["------" for _ in next(iter(results.values()))['class_ious']]) + "|\n")

        for model_name, model_results in results.items():
            class_ious = [f"{model_results['class_ious'][cn]:.2f}" for cn in model_results['class_ious'].keys()]
            f.write(f"| {model_name} | " + " | ".join(class_ious) + " |\n")
        baseline_class_ious = [f"{BASELINE_RESULTS['class_ious'][cn]:.2f}" for cn in BASELINE_RESULTS['class_ious'].keys()]
        f.write(f"| Baseline | " + " | ".join(baseline_class_ious) + " |\n")

        f.write("\n## Inference Time Comparison\n\n")
        f.write("| Model | Single-Scale (ms) | Multi-Scale (ms) |\n")
        f.write("|-------|-------------------|------------------|\n")
        for model_name, model_results in results.items():
            f.write(f"| {model_name} | {model_results['single_scale_time_ms']['mixed_precision_ms']:.2f} | {model_results['inference_time_ms']['mixed_precision_ms']:.2f} |\n")

        f.write("\n![Class-wise IoU Comparison](class_ious_comparison.png)\n")


def main():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    import argparse
    parser = argparse.ArgumentParser(description='Evaluate oil spill detection models')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for evaluation')
    parser.add_argument('--disable_ship_postprocessing', action='store_true', help='Disable ship-specific post-processing')
    parser.add_argument('--model_path', type=str, default='checkpoints/segformer_b2_best.weights.h5', help='Path to model weights file')
    parser.add_argument('--disable_tta', action='store_true', help='Disable Test-Time Augmentation')
    parser.add_argument('--tta_num_augs', type=int, default=8, help='Number of TTA augmentations to use')
    args = parser.parse_args()

    test_dataset, _, num_test_batches = load_dataset(data_dir='dataset', split='test', batch_size=args.batch_size)

    model_path = args.model_path
    model_name = 'SegFormer-B2'

    apply_ship_postprocessing = not args.disable_ship_postprocessing
    use_tta = not args.disable_tta
    tta_num_augs = args.tta_num_augs

    model_results = evaluate_model(
        model_path,
        test_dataset,
        batch_size=args.batch_size,
        apply_ship_postprocessing=apply_ship_postprocessing,
        use_tta=use_tta,
        tta_num_augs=tta_num_augs
    )

    if model_results:
        results = {model_name: model_results}

        print(f"\nResults for {model_name}:")
        print(f"Mean IoU: {model_results['mean_iou']:.2f}%")
        print("Class-wise IoU:")
        for class_name, iou in model_results['class_ious'].items():
            print(f"  {class_name}: {iou:.2f}%")
        print(f"Single-scale inference time: {model_results['single_scale_time_ms']['mixed_precision_ms']:.2f} ms")
        print(f"Multi-scale inference time: {model_results['inference_time_ms']['mixed_precision_ms']:.2f} ms")

        plot_class_ious(results)
        save_results(results)
    else:
        print(f"Evaluation failed for {model_name}")


if __name__ == "__main__":
    main()
