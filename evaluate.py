import os
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import glob

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
print(f"Current mixed precision policy: {policy.name}")

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

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

def apply_crf(images, predictions):
    """
    Apply Conditional Random Field to refine boundaries for oil spills and ships.

    Args:
        images: Batch of input images with shape [batch_size, height, width, channels]
        predictions: Tensor of shape [batch_size, height, width, num_classes] with class probabilities

    Returns:
        Enhanced predictions with refined boundaries
    """
    batch_size = tf.shape(predictions)[0]
    height, width = tf.shape(predictions)[1], tf.shape(predictions)[2]

    # Convert to numpy for processing
    images_np = images.numpy()
    predictions_np = predictions.numpy()

    # Create output array with same shape as predictions
    enhanced_preds = np.zeros_like(predictions_np)

    for i in range(batch_size):
        try:
            # Get the image and ensure it's C-contiguous
            img_np = np.ascontiguousarray(images_np[i])

            # Convert to uint8 for CRF processing
            img = (img_np * 255).astype(np.uint8)

            # Make sure we have a 3-channel image for CRF
            if img.shape[-1] == 1:
                # Expand single channel to 3 channels
                img = np.repeat(img, 3, axis=-1)
            elif len(img.shape) == 2:
                # Add channel dimension and repeat to 3 channels
                img = np.repeat(np.expand_dims(img, axis=-1), 3, axis=-1)

            # Get predictions for this image and ensure they're C-contiguous
            pred_np = np.ascontiguousarray(predictions_np[i])

            # Ensure the correct shape for densecrf input (classes, height, width)
            probs = np.ascontiguousarray(pred_np.transpose(2, 0, 1))

            # Apply CRF
            d = dcrf.DenseCRF2D(width, height, 5)
            U = unary_from_softmax(probs)
            d.setUnaryEnergy(U)

            # Add pairwise terms (appearance and smoothness)
            d.addPairwiseGaussian(sxy=3, compat=3)
            d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=img, compat=10)

            # Perform inference
            Q = d.inference(5)

            # Reshape back to expected format
            result = np.array(Q).reshape(5, height, width).transpose(1, 2, 0)
            enhanced_preds[i] = result

        except Exception as e:
            print(f"Error applying CRF to image {i}: {e}")
            # If CRF fails, use the original prediction
            enhanced_preds[i] = predictions_np[i]

    return tf.convert_to_tensor(enhanced_preds, dtype=predictions.dtype)


class MultiScalePredictor:
    """
    Enhanced Multi-scale prediction for semantic segmentation.
    Now includes Test-Time Augmentation (TTA) for improved boundary detection
    and rare class (oil spills, ships) accuracy.
    """

    def __init__(self, model, scales=[0.75, 1.0, 1.25], batch_size=4, use_tta=True):
        """
        Initialize with optimized scales for SAR imagery and TTA support.

        Args:
            model: The model to use for prediction
            scales: List of scales for multi-scale prediction (optimized for 384x384)
            batch_size: Batch size for prediction (reduced to 4 for 8GB VRAM)
            use_tta: Whether to use Test-Time Augmentation
        """
        self.model = model
        self.scales = scales
        self.batch_size = batch_size
        self.use_tta = use_tta
        self.expected_height = model.input_shape[1]
        self.expected_width = model.input_shape[2]

        # Get expected input channels from model
        self.expected_channels = self.model.input_shape[3]
        print(f"Model expects input with {self.expected_channels} channels")

        # Initialize TTA if needed
        if self.use_tta:
            from test_time_augmentation import TestTimeAugmentation
            self.tta = TestTimeAugmentation(
                model,
                num_augmentations=5,  # Use 5 augmentations by default
                use_flips=True,       # Use horizontal/vertical flips
                use_scales=False,     # Scales are handled by multi-scale prediction
                use_rotations=True,   # Use rotations (helps with ship orientation)
                include_original=True # Always include original image
            )
            print(f"TTA enabled with 5 augmentations for improved rare class detection")

        print(f"MultiScalePredictor initialized with scales {scales}, batch_size={batch_size}")

    @tf.function
    def _predict_batch(self, batch):
        return self.model(batch, training=False)

    def predict(self, image_batch):
        """
        Predict using multiple scales and TTA for optimal accuracy.

        Args:
            image_batch: Batch of images with shape [batch_size, height, width, channels]

        Returns:
            Enhanced predictions with shape [batch_size, height, width, num_classes]
        """
        batch_size = tf.shape(image_batch)[0]
        height = tf.shape(image_batch)[1]
        width = tf.shape(image_batch)[2]

        # Ensure input has the correct number of channels for the model
        if image_batch.shape[-1] != self.expected_channels:
            print(f"Input shape after channel adjustment: {image_batch.shape}, dtype: {image_batch.dtype}")
            if self.expected_channels == 1 and image_batch.shape[-1] > 1:
                # Convert multi-channel to single channel
                image_batch = tf.expand_dims(tf.reduce_mean(image_batch, axis=-1), axis=-1)
            elif self.expected_channels == 3 and image_batch.shape[-1] == 1:
                # Convert grayscale to RGB by duplicating the channel
                image_batch = tf.concat([image_batch, image_batch, image_batch], axis=-1)

        # Cast input to the policy's compute dtype (float16 for mixed precision)
        from tensorflow.keras import mixed_precision # type: ignore
        policy = mixed_precision.global_policy()
        image_batch = tf.cast(image_batch, policy.compute_dtype)

        # Store original size for later use
        original_size = (height, width)

        # Placeholder for all scaled predictions
        all_predictions = []

        # Make predictions at each scale
        for scale in self.scales:
            try:
                # Calculate scaled dimensions
                scaled_height = tf.cast(tf.cast(height, tf.float32) * scale, tf.int32)
                scaled_width = tf.cast(tf.cast(width, tf.float32) * scale, tf.int32)

                # Make sure the dimensions are divisible by 8 (for the model's architecture)
                scaled_height = tf.cast(tf.math.ceil(scaled_height / 8) * 8, tf.int32)
                scaled_width = tf.cast(tf.math.ceil(scaled_width / 8) * 8, tf.int32)

                # Resize the input images to the scaled dimensions
                scaled_batch = tf.image.resize(
                    image_batch,
                    size=(scaled_height, scaled_width),
                    method='bilinear'
                )

                # Resize to the model's expected input dimensions
                model_input = tf.image.resize(
                    scaled_batch,
                    size=(self.expected_height, self.expected_width),
                    method='bilinear'
                )

                print(f"Resized images at scale {scale}: shape={model_input.shape}, dtype={model_input.dtype}")

                # For each scale, use TTA if enabled
                if self.use_tta:
                    # Apply TTA - this will handle multiple augmentations internally
                    logits = self.tta.predict(model_input)
                else:
                    # Regular prediction without TTA
                    logits = self._predict_batch(model_input)

                # Resize prediction back to original size
                if tf.shape(logits)[1] != height or tf.shape(logits)[2] != width:
                    resized_logits = tf.image.resize(
                        logits,
                        size=original_size,
                        method='bilinear'
                    )
                else:
                    resized_logits = logits

                # Convert to probabilities
                probs = tf.nn.softmax(resized_logits, axis=-1)
                all_predictions.append(probs)

            except Exception as e:
                print(f"Error at scale {scale}: {e}")
                import traceback
                traceback.print_exc()
                # Skip this scale if there's an error
                continue

        # Check if we have any successful predictions
        if not all_predictions:
            print("No successful predictions at any scale. Attempting direct prediction...")
            try:
                # Resize to model's expected input dimensions
                model_input = tf.image.resize(
                    image_batch,
                    size=(self.expected_height, self.expected_width),
                    method='bilinear'
                )
                # Try direct prediction
                logits = self.model(model_input, training=False)

                # Resize prediction back to original size if needed
                if tf.shape(logits)[1] != height or tf.shape(logits)[2] != width:
                    logits = tf.image.resize(
                        logits,
                        size=original_size,
                        method='bilinear'
                    )
                return logits
            except Exception as e:
                print(f"Direct prediction also failed: {e}")
                # Create a dummy prediction with the right shape
                dummy_shape = list(image_batch.shape)
                dummy_shape[-1] = 5  # Assuming 5 classes for oil spill segmentation
                return tf.zeros(dummy_shape)

        # Average predictions from all scales
        fused_prediction = tf.reduce_mean(tf.stack(all_predictions, axis=0), axis=0)

        # Convert back to logits for consistency with model output
        epsilon = 1e-7
        fused_prediction = tf.clip_by_value(fused_prediction, epsilon, 1.0 - epsilon)
        logits = tf.math.log(fused_prediction)

        return logits


def compute_iou(y_true, y_pred, num_classes=5):
    """
    Compute mean IoU and class-wise IoU.

    Args:
        y_true: Ground truth labels with shape [batch_size, height, width, 1]
        y_pred: Predictions with shape [batch_size, height, width, num_classes]
        num_classes: Number of classes

    Returns:
        mean_iou: Mean IoU across all classes
        class_ious: IoU for each class
    """
    # Convert y_true to integer class indices and remove last dimension
    y_true = tf.cast(y_true, tf.int32)
    y_true = tf.squeeze(y_true, axis=-1)

    # Convert y_pred from logits/probabilities to class indices
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, tf.int32)

    # Initialize confusion matrix
    confusion_matrix = tf.zeros((num_classes, num_classes), dtype=tf.float32)

    # Update confusion matrix for each image in the batch
    for i in range(tf.shape(y_true)[0]):
        true_flat = tf.reshape(y_true[i], [-1])
        pred_flat = tf.reshape(y_pred[i], [-1])

        # Compute confusion matrix for this image
        cm_i = tf.math.confusion_matrix(
            true_flat, pred_flat,
            num_classes=num_classes,
            dtype=tf.float32
        )

        # Add to overall confusion matrix
        confusion_matrix += cm_i

    # Calculate IoU for each class
    # IoU = true_positive / (true_positive + false_positive + false_negative)
    sum_over_row = tf.reduce_sum(confusion_matrix, axis=0)
    sum_over_col = tf.reduce_sum(confusion_matrix, axis=1)
    true_positives = tf.linalg.tensor_diag_part(confusion_matrix)

    # sum_over_row + sum_over_col - true_positives = TP + FP + FN
    denominator = sum_over_row + sum_over_col - true_positives

    # The IoU is set to 0 if the denominator is 0
    class_ious = tf.math.divide_no_nan(true_positives, denominator)

    # Calculate mean IoU excluding classes that don't appear in ground truth
    mask = tf.greater(sum_over_col, 0)
    mean_iou = tf.reduce_mean(tf.boolean_mask(class_ious, mask))

    return mean_iou.numpy(), class_ious.numpy()


def measure_inference_time(model, num_runs=5, batch_size=1, multi_scale=False, use_tta=False):
    """
    Measure average inference time for a model.

    Args:
        model: The model to measure
        num_runs: Number of inference runs to average over
        batch_size: Batch size for inference
        multi_scale: Whether to use multi-scale prediction
        use_tta: Whether to use Test-Time Augmentation

    Returns:
        Average inference time in milliseconds and comparison results
    """
    # Create a dummy input with the model's expected input shape
    input_shape = model.input_shape[1:3]
    expected_channels = model.input_shape[3]  # Get expected channel count from model
    dummy_input = tf.random.normal((batch_size, *input_shape, expected_channels))

    # Test with mixed precision (float16)
    print(f"Testing inference with mixed precision ({policy.compute_dtype})...")

    # Cast input to mixed precision dtype
    mp_input = tf.cast(dummy_input, policy.compute_dtype)

    # Warmpup
    for _ in range(3):
        if multi_scale:
            # Use optimized scales [0.75, 1.0, 1.25] for better context
            predictor = MultiScalePredictor(model, scales=[0.75, 1.0, 1.25], batch_size=batch_size, use_tta=use_tta)
            _ = predictor.predict(mp_input)
        else:
            _ = model(mp_input, training=False)

    # Measure time with mixed precision
    start_time = time.time()
    for _ in range(num_runs):
        if multi_scale:
            predictor = MultiScalePredictor(model, scales=[0.75, 1.0, 1.25], batch_size=batch_size, use_tta=use_tta)
            _ = predictor.predict(mp_input)
        else:
            _ = model(mp_input, training=False)
    end_time = time.time()

    # Calculate average time in milliseconds for mixed precision
    mp_avg_time = (end_time - start_time) * 1000 / num_runs
    print(f"Mixed precision average inference time: {mp_avg_time:.2f} ms")

    # Test with float32 for comparison
    print(f"Testing inference with float32...")

    # Cast input to float32
    fp32_input = tf.cast(dummy_input, tf.float32)

    # Create a temporary policy context for float32
    original_policy = mixed_precision.global_policy()
    mixed_precision.set_global_policy('float32')

    # Warmup with float32
    for _ in range(3):
        if multi_scale:
            predictor = MultiScalePredictor(model, scales=[0.75, 1.0, 1.25], batch_size=batch_size, use_tta=use_tta)
            _ = predictor.predict(fp32_input)
        else:
            _ = model(fp32_input, training=False)

    # Measure time with float32
    start_time = time.time()
    for _ in range(num_runs):
        if multi_scale:
            predictor = MultiScalePredictor(model, scales=[0.75, 1.0, 1.25], batch_size=batch_size, use_tta=use_tta)
            _ = predictor.predict(fp32_input)
        else:
            _ = model(fp32_input, training=False)
    end_time = time.time()

    # Calculate average time in milliseconds for float32
    fp32_avg_time = (end_time - start_time) * 1000 / num_runs
    print(f"Float32 average inference time: {fp32_avg_time:.2f} ms")

    # Calculate speedup
    speedup = fp32_avg_time / mp_avg_time if mp_avg_time > 0 else 0
    print(f"Mixed precision speedup: {speedup:.2f}x")

    # Restore original policy
    mixed_precision.set_global_policy(original_policy)

    # Return mixed precision time as the primary result along with comparison data
    return {
        'mixed_precision_ms': mp_avg_time,
        'float32_ms': fp32_avg_time,
        'speedup': speedup
    }


def apply_ship_enhancement(predictions, ship_class_idx=3):
    """
    Apply morphological operations to enhance ship detection.

    Args:
        predictions: Tensor of shape [batch_size, height, width, num_classes] with class probabilities
        ship_class_idx: Index of the ship class (default 3)

    Returns:
        Enhanced predictions with improved ship detection
    """
    import cv2

    # Convert to numpy for OpenCV operations
    predictions_np = predictions.numpy()
    batch_size = predictions_np.shape[0]
    enhanced_predictions = np.copy(predictions_np)

    # Process each image in the batch
    for i in range(batch_size):
        # Extract ship probabilities
        ship_probs = predictions_np[i, :, :, ship_class_idx]

        # Convert to binary mask (threshold at 0.3)
        ship_mask = (ship_probs > 0.3).astype(np.uint8)

        # Skip if no ships detected
        if np.sum(ship_mask) == 0:
            continue

        # Apply morphological closing to connect nearby ship pixels
        kernel = np.ones((3, 3), np.uint8)
        closed_mask = cv2.morphologyEx(ship_mask, cv2.MORPH_CLOSE, kernel)

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed_mask, connectivity=8)

        # Filter out very small components (likely noise)
        min_size = 5  # Minimum size in pixels
        enhanced_mask = np.zeros_like(ship_mask)

        for j in range(1, num_labels):  # Skip background (label 0)
            if stats[j, cv2.CC_STAT_AREA] >= min_size:
                component_mask = (labels == j).astype(np.uint8)

                # For medium-sized components, apply slight dilation to improve connectivity
                if stats[j, cv2.CC_STAT_AREA] < 50:
                    component_mask = cv2.dilate(component_mask, kernel, iterations=1)

                enhanced_mask = np.logical_or(enhanced_mask, component_mask)

        # Convert back to uint8
        enhanced_mask = enhanced_mask.astype(np.uint8)

        # Apply the enhanced mask to the predictions
        # Calculate the boost factor based on confidence
        boost_factor = 1.2  # Boost ship probabilities by 20%

        # Create a copy of the ship probabilities
        enhanced_ship_probs = np.copy(ship_probs)

        # Boost probabilities where the enhanced mask is 1
        enhanced_ship_probs[enhanced_mask == 1] *= boost_factor

        # Clip to ensure probabilities remain in [0, 1]
        enhanced_ship_probs = np.clip(enhanced_ship_probs, 0, 1)

        # Update ship class probabilities in the predictions
        enhanced_predictions[i, :, :, ship_class_idx] = enhanced_ship_probs

        # Renormalize the probabilities to ensure they sum to 1
        # This maintains the probabilistic interpretation
        sum_probs = np.sum(enhanced_predictions[i], axis=-1, keepdims=True)
        enhanced_predictions[i] /= sum_probs

    return tf.convert_to_tensor(enhanced_predictions, dtype=predictions.dtype)


def evaluate_model(model_path, test_dataset, batch_size=4, apply_ship_postprocessing=True,
                use_tta=True, tta_num_augs=5):
    """
    Evaluate the SegFormer-B2 model on the test dataset.

    Args:
        model_path: Path to the saved model weights
        test_dataset: The test dataset
        batch_size: Batch size for evaluation (reduced for larger model)
        apply_ship_postprocessing: Whether to apply ship-specific post-processing
        use_tta: Whether to use Test-Time Augmentation
        tta_num_augs: Number of TTA augmentations to use

    Returns:
        Dictionary containing evaluation results
    """
    # Import model definition
    from model import OilSpillSegformer, create_pretrained_weight_loader
    from loss import HybridSegmentationLoss
    from test_time_augmentation import TestTimeAugmentation

    # Configure TensorFlow for optimal performance
    print(f"Using TensorFlow with default thread configuration")

    # Optimize GPU memory usage
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Configure TensorFlow to use less GPU memory initially and grow as needed
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU memory growth enabled for {len(gpus)} GPUs")
        except Exception as e:
            print(f"Error configuring GPU: {e}")

    # Use the final weights file if the best weights file is causing issues
    if not os.path.exists(model_path) or 'best' in model_path:
        alt_model_path = 'segformer_b2_final.weights.h5'
        if os.path.exists(alt_model_path):
            print(f"Using alternative weights file: {alt_model_path}")
            model_path = alt_model_path

    # Force mixed precision to match training conditions
    original_policy = mixed_precision.global_policy()
    if original_policy.name != 'mixed_float16':
        print(f"Temporarily setting mixed precision to mixed_float16 for model loading")
        mixed_precision.set_global_policy('mixed_float16')

    # Load model from checkpointed weights file
    print(f"Loading model from checkpoint: {model_path}")
    try:
        # Create a new SegFormer-B2 model with the appropriate configuration
        model = OilSpillSegformer(
            input_shape=(384, 384, 1),  # 384x384 input size
            num_classes=5,              # 5 classes (Sea Surface, Oil Spill, Look-alike, Ship, Land)
            drop_rate=0.0,              # Disable dropout for inference
            use_cbam=False,             # Disable CBAM to match training configuration
            pretrained_weights=None     # No need for pretrained weights since we're loading checkpoint
        )

        # Call the model once to build it
        dummy_input = tf.zeros((1, 384, 384, 1), dtype=tf.float32)
        _ = model(dummy_input, training=False)
        print("Model built successfully")

        # Use custom weight loader from the model module
        weight_loader = create_pretrained_weight_loader()
        success = weight_loader(model, model_path)

        if success:
            print("Weights loaded successfully with custom weight loader")
        else:
            raise RuntimeError("Failed to load weights with custom loader")

        # Restore original policy
        if original_policy.name != 'mixed_float16':
            mixed_precision.set_global_policy(original_policy)
            print(f"Restored original mixed precision policy: {original_policy.name}")
    except Exception as e:
        print(f"Error loading SegFormer-B2 model: {e}")

        # Restore original policy if there was an error
        if original_policy.name != 'mixed_float16':
            mixed_precision.set_global_policy(original_policy)

        return None

    # Create predictor according to evaluation settings
    if use_tta:
        print(f"Using Test-Time Augmentation with {tta_num_augs} augmentations for better mIoU")
        # Use MultiScalePredictor with TTA enabled for consistency with training eval
        predictor = MultiScalePredictor(
            model,
            scales=[0.75, 1.0, 1.25],  # Same scales as used in training
            batch_size=batch_size,
            use_tta=True
        )
    else:
        # Use multi-scale predictor without TTA
        predictor = MultiScalePredictor(
            model,
            scales=[0.75, 1.0, 1.25],  # Same scales as used in training
            batch_size=batch_size,
            use_tta=False
        )

    # Check if the dataset is already batched
    is_already_batched = False
    try:
        # Get the first batch to check its shape
        sample_images, sample_labels = next(iter(test_dataset))
        print(f"Dataset element shape: images={sample_images.shape}, labels={sample_labels.shape}")
        if len(sample_images.shape) == 4 and sample_images.shape[0] > 1:
            is_already_batched = True
            print("Dataset appears to be already batched")
    except:
        print("Could not determine if dataset is batched, assuming it's not")

    # Prepare the test dataset
    if is_already_batched:
        test_batches = test_dataset
    else:
        test_batches = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Initialize list to store ground truth and predictions
    all_true_labels = []
    all_predictions = []

    # Process each batch
    print("Evaluating on test dataset with SegFormer-B2 model...")
    for images, labels in tqdm(test_batches):
        try:
            # Get predictions using the predictor
            predictions = predictor.predict(images)

            # Apply ship-specific post-processing if enabled
            if apply_ship_postprocessing:
                predictions = apply_ship_enhancement(predictions, ship_class_idx=3)

            # Store labels and predictions (skip CRF for now as it may be causing issues)
            all_true_labels.append(labels)
            all_predictions.append(predictions)
        except Exception as e:
            print(f"Error processing batch: {e}")
            continue

    # Check if we have any successful predictions
    if not all_true_labels:
        print("No successful predictions were made. Evaluation failed.")
        return None

    # Concatenate all batches
    true_labels = tf.concat(all_true_labels, axis=0)
    predictions = tf.concat(all_predictions, axis=0)

    # Compute IoU metrics
    mean_iou, class_ious = compute_iou(true_labels, predictions, num_classes=5)

    # Compute inference time
    inference_time_results = measure_inference_time(model, multi_scale=True, use_tta=use_tta)
    single_scale_time_results = measure_inference_time(model, multi_scale=False, use_tta=use_tta)

    # Prepare results
    class_names = ['Sea Surface', 'Oil Spill', 'Look-alike', 'Ship', 'Land']

    results = {
        'mean_iou': mean_iou * 100,  # Convert to percentage
        'class_ious': {class_names[i]: class_ious[i] * 100 for i in range(len(class_names))},
        'inference_time_ms': inference_time_results,
        'single_scale_time_ms': single_scale_time_results
    }

    return results


def plot_class_ious(results, title="Class-wise IoU Comparison"):
    """
    Plot class-wise IoU comparison between models.

    Args:
        results: Dictionary of model results
        title: Plot title
    """
    plt.figure(figsize=(12, 6))

    class_names = list(next(iter(results.values()))['class_ious'].keys())
    x = np.arange(len(class_names))
    width = 0.8 / (len(results) + 1)  # +1 for baseline

    # Plot model results
    for i, (model_name, model_results) in enumerate(results.items()):
        class_ious = [model_results['class_ious'][cn] for cn in class_names]
        plt.bar(x + i*width - 0.4 + width/2, class_ious, width, label=model_name)

    # Plot baseline results
    baseline_ious = [BASELINE_RESULTS['class_ious'][cn] for cn in class_names]
    plt.bar(x + len(results)*width - 0.4 + width/2, baseline_ious, width, label='Baseline', color='gray', alpha=0.7)

    plt.xlabel('Classes')
    plt.ylabel('IoU (%)')
    plt.title(title)
    plt.xticks(x, class_names, rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add values on top of bars for models
    for i, (model_name, model_results) in enumerate(results.items()):
        class_ious = [model_results['class_ious'][cn] for cn in class_names]
        for j, v in enumerate(class_ious):
            plt.text(j + i*width - 0.4 + width/2, v + 0.5, f"{v:.1f}",
                     ha='center', fontsize=8)

    # Add values on top of bars for baseline
    for j, v in enumerate(baseline_ious):
        plt.text(j + len(results)*width - 0.4 + width/2, v + 0.5, f"{v:.1f}",
                 ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig('class_ious_comparison.png', dpi=300)
    plt.close()

    print(f"Class-wise IoU comparison plot saved as class_ious_comparison.png")


def save_results(results, output_file="evaluation_results.md"):
    """
    Save evaluation results to a markdown file.

    Args:
        results: Dictionary of model results
        output_file: Output file name
    """
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

    print(f"Evaluation results saved to {output_file}")


def main():
    """Main evaluation function."""
    # Configure GPU memory growth to avoid OOM errors
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Using {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"GPU memory configuration error: {e}")
    else:
        print("Using CPU")

    # Allow specifying a specific model via command line argument
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate oil spill detection models')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for evaluation (reduced for larger model)')
    parser.add_argument('--disable_ship_postprocessing', action='store_true',
                       help='Disable ship-specific post-processing')
    parser.add_argument('--model_path', type=str, default='checkpoints/segformer_b2_best.weights.h5',
                       help='Path to model weights file')
    parser.add_argument('--disable_tta', action='store_true', help='Disable Test-Time Augmentation')
    parser.add_argument('--tta_num_augs', type=int, default=5, help='Number of TTA augmentations to use')
    args = parser.parse_args()

    # Load the test dataset with 384x384 input size
    print("Loading test dataset...")
    test_dataset, _, num_test_batches = load_dataset(data_dir='dataset', split='test', batch_size=args.batch_size)
    print(f"Loaded test dataset with {num_test_batches} batches with batch_size={args.batch_size}")

    # Update the model name and path for SegFormer-B2
    model_path = args.model_path
    model_name = 'SegFormer-B2'
    print(f"\nEvaluating model: {model_name} from {model_path}")

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

        # Print results
        print(f"\nResults for {model_name}:")
        print(f"Mean IoU: {model_results['mean_iou']:.2f}%")
        print("Class-wise IoU:")
        for class_name, iou in model_results['class_ious'].items():
            print(f"  {class_name}: {iou:.2f}%")
        print(f"Single-scale inference time: {model_results['single_scale_time_ms']['mixed_precision_ms']:.2f} ms")
        print(f"Multi-scale inference time: {model_results['inference_time_ms']['mixed_precision_ms']:.2f} ms")

        # Plot class-wise IoU comparison
        plot_class_ious(results)

        # Save results to markdown file
        save_results(results)
    else:
        print(f"Evaluation failed for {model_name}")


if __name__ == "__main__":
    main()
