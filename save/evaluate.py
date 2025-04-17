"""
Oil Spill Detection - Model Evaluation Script

This script evaluates the trained DeepLabv3+ model on the oil spill detection test dataset.
It computes class-wise IoU metrics, mean IoU, and inference time.
The results are compared with a baseline model and saved as a markdown file.

Features:
- Multi-scale inference (50%, 75%, 100%)
- Class-wise IoU measurement
- Inference time benchmarking
"""

import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

# Set TensorFlow logging level to reduce verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Import the data loading function from data_loader.py
from data_loader import load_dataset


class MultiScalePredictor:
    """
    Multi-scale prediction for semantic segmentation.
    Accepts images at multiple scales and combines predictions.
    """
    def __init__(self, model, scales=[0.5, 0.75, 1.0]):
        self.model = model
        self.scales = scales

    def predict(self, image_batch):
        """
        Predict using multiple scales and fuse results.

        Args:
            image_batch: Batch of images with shape [batch_size, height, width, channels]

        Returns:
            Fused predictions with shape [batch_size, height, width, num_classes]
        """
        batch_size = tf.shape(image_batch)[0]
        height = tf.shape(image_batch)[1]
        width = tf.shape(image_batch)[2]

        # Placeholder for all scaled predictions
        all_predictions = []

        # Make predictions at each scale
        for scale in self.scales:
            # Resize images to current scale
            scaled_height = tf.cast(tf.cast(height, tf.float32) * scale, tf.int32)
            scaled_width = tf.cast(tf.cast(width, tf.float32) * scale, tf.int32)

            # Ensure dimensions are multiples of 8 for better performance
            scaled_height = tf.cast(tf.math.ceil(scaled_height / 8) * 8, tf.int32)
            scaled_width = tf.cast(tf.math.ceil(scaled_width / 8) * 8, tf.int32)

            scaled_images = tf.image.resize(
                image_batch,
                size=(scaled_height, scaled_width),
                method='bilinear'
            )

            # Get predictions for scaled images
            scaled_preds = self.model(scaled_images, training=False)

            # Resize predictions back to original size
            resized_preds = tf.image.resize(
                scaled_preds,
                size=(height, width),
                method='bilinear'
            )

            # Apply softmax to get probabilities
            probs = tf.nn.softmax(resized_preds, axis=-1)

            all_predictions.append(probs)

        # Average predictions from all scales
        fused_prediction = tf.reduce_mean(tf.stack(all_predictions, axis=0), axis=0)

        return fused_prediction


def compute_iou(y_true, y_pred, num_classes=5):
    """
    Compute IoU metrics for each class and mean IoU.

    Args:
        y_true: Ground truth labels, shape [batch, height, width, 1], values 0-4
        y_pred: Predicted class probabilities, shape [batch, height, width, num_classes]
        num_classes: Number of classes

    Returns:
        Tuple of (mean_iou, class_ious)
    """
    # Convert y_pred from probabilities to class indices
    y_pred_classes = tf.argmax(y_pred, axis=-1)
    y_pred_classes = tf.cast(y_pred_classes, tf.int32)

    # Squeeze y_true to match y_pred_classes shape
    y_true = tf.cast(tf.squeeze(y_true, axis=-1), tf.int32)

    # Initialize MeanIoU metric
    mean_iou_metric = tf.keras.metrics.MeanIoU(num_classes=num_classes)

    # Update state with predictions
    mean_iou_metric.update_state(y_true, y_pred_classes)

    # Calculate mean IoU
    mean_iou = mean_iou_metric.result().numpy()

    # Calculate class-wise IoU
    confusion_matrix = mean_iou_metric.total_cm.numpy()

    # IoU = true_positive / (true_positive + false_positive + false_negative)
    class_ious = []
    for i in range(num_classes):
        true_positive = confusion_matrix[i, i]
        false_positive = np.sum(confusion_matrix[:, i]) - true_positive
        false_negative = np.sum(confusion_matrix[i, :]) - true_positive

        iou = true_positive / (true_positive + false_positive + false_negative + 1e-7)
        class_ious.append(iou)

    return mean_iou, class_ious


def measure_inference_time(model, input_shape=(1, 320, 320, 3), multi_scale=True, num_runs=10):
    """
    Measure inference time for a single image.

    Args:
        model: The model to evaluate
        input_shape: Input shape for test images
        multi_scale: Whether to use multi-scale inference
        num_runs: Number of runs for averaging

    Returns:
        Average inference time in milliseconds
    """
    # Create a dummy input
    dummy_input = tf.random.normal(input_shape)

    # Create multi-scale predictor if needed
    if multi_scale:
        predictor = MultiScalePredictor(model, scales=[0.5, 0.75, 1.0])

    # Warmup runs
    for _ in range(5):
        if multi_scale:
            _ = predictor.predict(dummy_input)
        else:
            _ = model(dummy_input, training=False)

    # Measure inference time
    start_time = time.time()

    for _ in range(num_runs):
        if multi_scale:
            _ = predictor.predict(dummy_input)
        else:
            _ = model(dummy_input, training=False)

    end_time = time.time()

    # Calculate average time in milliseconds
    avg_time_ms = (end_time - start_time) * 1000 / num_runs

    return avg_time_ms


def evaluate_model(model_path, test_dataset, batch_size=8):
    """
    Evaluate the DeepLabv3+ model on the test dataset.

    Args:
        model_path: Path to the saved model
        test_dataset: The test dataset
        batch_size: Batch size for evaluation

    Returns:
        Dictionary containing evaluation results
    """
    # Load the model
    print(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)

    # Create a multi-scale predictor
    predictor = MultiScalePredictor(model, scales=[0.5, 0.75, 1.0])

    # Prepare the test dataset
    test_batches = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Initialize list to store ground truth and predictions
    all_true_labels = []
    all_predictions = []

    # Process each batch
    print("Evaluating on test dataset...")
    for images, labels in tqdm(test_batches):
        # Get predictions using multi-scale fusion
        predictions = predictor.predict(images)

        # Store labels and predictions
        all_true_labels.append(labels)
        all_predictions.append(predictions)

    # Concatenate all batches
    true_labels = tf.concat(all_true_labels, axis=0)
    predictions = tf.concat(all_predictions, axis=0)

    # Compute IoU metrics
    mean_iou, class_ious = compute_iou(true_labels, predictions, num_classes=5)

    # Compute inference time
    inference_time = measure_inference_time(model, multi_scale=True)
    single_scale_time = measure_inference_time(model, multi_scale=False)

    # Prepare results
    class_names = ['Sea Surface', 'Oil Spill', 'Look-alike', 'Ship', 'Land']

    results = {
        'mean_iou': mean_iou * 100,  # Convert to percentage
        'class_ious': {class_names[i]: class_ious[i] * 100 for i in range(len(class_names))},
        'inference_time_ms': inference_time,
        'single_scale_time_ms': single_scale_time
    }

    return results


def create_results_markdown(results, baseline_miou=65.06, baseline_ious=[96.43, 53.38, 55.40, 27.63, 92.44]):
    """
    Create a markdown file with evaluation results.

    Args:
        results: Dictionary with evaluation results
        baseline_miou: Baseline mean IoU percentage
        baseline_ious: List of baseline class IoUs
    """
    class_names = ['Sea Surface', 'Oil Spill', 'Look-alike', 'Ship', 'Land']

    # Create a markdown string
    markdown = "# DeepLabv3+ Model Evaluation Results\n\n"

    # Add date and time
    import datetime
    now = datetime.datetime.now()
    markdown += f"Evaluation performed on: {now.strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    # Model configuration
    markdown += "## Model Configuration\n\n"
    markdown += "- **Model**: DeepLabv3+ with EfficientNet-B4 backbone\n"
    markdown += "- **Input Size**: 320x320x3\n"
    markdown += "- **Output Classes**: 5 (Sea Surface, Oil Spill, Look-alike, Ship, Land)\n"
    markdown += "- **Inference Method**: Multi-scale fusion (50%, 75%, 100%)\n\n"

    # Performance metrics
    markdown += "## Performance Metrics\n\n"

    # IoU comparison table
    markdown += "### IoU Metrics Comparison\n\n"
    markdown += "| Class | Our Model (%) | Baseline (%) | Difference (%) |\n"
    markdown += "|-------|--------------|--------------|----------------|\n"

    our_ious = [results['class_ious'][name] for name in class_names]

    for i, class_name in enumerate(class_names):
        diff = our_ious[i] - baseline_ious[i]
        diff_str = f"{diff:.2f}"
        if diff > 0:
            diff_str = f"+{diff_str}"

        markdown += f"| {class_name} | {our_ious[i]:.2f} | {baseline_ious[i]:.2f} | {diff_str} |\n"

    # Mean IoU comparison
    diff_miou = results['mean_iou'] - baseline_miou
    diff_str = f"{diff_miou:.2f}"
    if diff_miou > 0:
        diff_str = f"+{diff_str}"

    markdown += f"| **Mean IoU** | **{results['mean_iou']:.2f}** | **{baseline_miou:.2f}** | **{diff_str}** |\n\n"

    # Inference time
    markdown += "### Inference Time\n\n"
    markdown += f"- **Multi-scale Inference**: {results['inference_time_ms']:.2f} ms per image\n"
    markdown += f"- **Single-scale Inference**: {results['single_scale_time_ms']:.2f} ms per image\n\n"

    # Analysis
    markdown += "## Analysis\n\n"

    # Compare with baseline
    if results['mean_iou'] > baseline_miou:
        markdown += f"Our model shows an improvement of {diff_miou:.2f}% in mean IoU compared to the baseline.\n\n"
    else:
        markdown += f"Our model's mean IoU is {-diff_miou:.2f}% lower than the baseline.\n\n"

    # Class-wise analysis
    best_improvement = max([(our_ious[i] - baseline_ious[i], class_names[i]) for i in range(len(class_names))])
    worst_change = min([(our_ious[i] - baseline_ious[i], class_names[i]) for i in range(len(class_names))])

    markdown += f"The largest improvement is in the '{best_improvement[1]}' class with a {best_improvement[0]:.2f}% increase in IoU.\n"

    if worst_change[0] < 0:
        markdown += f"The largest decrease is in the '{worst_change[1]}' class with a {-worst_change[0]:.2f}% reduction in IoU.\n"
    else:
        markdown += f"All classes show improvement over the baseline.\n"

    # Save to markdown file
    with open('evaluation_results.md', 'w') as f:
        f.write(markdown)

    print(f"Results saved to evaluation_results.md")


def main():
    # Configure GPU memory growth to avoid OOM errors
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Using {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"GPU memory growth configuration error: {e}")
    else:
        print("Using CPU")

    # Load the test dataset
    print("Loading test dataset...")
    test_dataset = load_dataset(data_dir='dataset', split='test')

    # Check if the model file exists
    model_path = 'improved_deeplabv3plus.h5'
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        print("Please run training first or make sure the model file is in the correct location.")
        return

    # Evaluate the model
    results = evaluate_model(model_path, test_dataset)

    # Create and save the markdown results
    create_results_markdown(results)

    # Print a summary
    print("\nEvaluation Summary:")
    print(f"Mean IoU: {results['mean_iou']:.2f}%")
    print("Class-wise IoU:")
    for class_name, iou in results['class_ious'].items():
        print(f"  {class_name}: {iou:.2f}%")
    print(f"Multi-scale inference time: {results['inference_time_ms']:.2f} ms per image")


if __name__ == "__main__":
    main()
