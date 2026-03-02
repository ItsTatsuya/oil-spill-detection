"""
Shared multi-scale predictor with optional TTA for oil spill segmentation.

Consolidates the duplicated MultiScalePredictor from train.py and evaluate.py.
"""

import logging
import tensorflow as tf

logger = logging.getLogger('oil_spill')


class MultiScalePredictor:
    """
    Multi-scale prediction with optional Test-Time Augmentation (TTA).

    Parameters
    ----------
    model : tf.keras.Model
    scales : list of float
    batch_size : int
    use_tta : bool
        When True, applies TTA at each scale via TestTimeAugmentation.
    tta_num_augmentations : int
    """

    def __init__(
        self,
        model,
        scales=(0.75, 1.0, 1.25),
        batch_size=4,
        use_tta=False,
        tta_num_augmentations=8,
    ):
        self.model = model
        self.scales = list(scales)
        self.batch_size = batch_size
        self.use_tta = use_tta
        self.expected_height = model.input_shape[1]
        self.expected_width = model.input_shape[2]
        self.expected_channels = model.input_shape[3]

        if self.use_tta:
            from data.test_time_augmentation import TestTimeAugmentation
            self.tta = TestTimeAugmentation(
                model,
                num_augmentations=tta_num_augmentations,
                use_flips=True,
                use_scales=False,
                use_rotations=True,
                include_original=True,
            )

    @tf.function
    def _predict_batch(self, batch):
        return self.model(batch, training=False)

    def predict(self, image_batch):
        batch_size = tf.shape(image_batch)[0]
        height = tf.shape(image_batch)[1]
        width = tf.shape(image_batch)[2]

        # Fix channel mismatch
        if image_batch.shape[-1] != self.expected_channels:
            if self.expected_channels == 1 and image_batch.shape[-1] > 1:
                image_batch = tf.expand_dims(tf.reduce_mean(image_batch, axis=-1), axis=-1)
            elif self.expected_channels == 3 and image_batch.shape[-1] == 1:
                image_batch = tf.concat([image_batch] * 3, axis=-1)

        policy = tf.keras.mixed_precision.global_policy()
        image_batch = tf.cast(image_batch, policy.compute_dtype)

        original_size = (height, width)
        all_predictions = []

        for scale in self.scales:
            try:
                scaled_h = tf.cast(tf.cast(height, tf.float32) * scale, tf.int32)
                scaled_w = tf.cast(tf.cast(width, tf.float32) * scale, tf.int32)
                # Round up to multiple of 8 for SegFormer compatibility
                scaled_h = tf.cast(tf.math.ceil(scaled_h / 8) * 8, tf.int32)
                scaled_w = tf.cast(tf.math.ceil(scaled_w / 8) * 8, tf.int32)

                scaled_batch = tf.image.resize(
                    image_batch, (scaled_h, scaled_w), method='bilinear'
                )
                model_input = tf.image.resize(
                    scaled_batch,
                    (self.expected_height, self.expected_width),
                    method='bilinear',
                )

                if self.use_tta:
                    logits = self.tta.predict(model_input)
                else:
                    logits = self._predict_batch(model_input)

                logits = tf.cast(logits, tf.float32)

                if tf.shape(logits)[1] != height or tf.shape(logits)[2] != width:
                    logits = tf.image.resize(logits, original_size, method='bilinear')

                probs = tf.nn.softmax(logits, axis=-1)
                all_predictions.append(probs)

            except Exception as e:
                logger.warning("Scale %.2f failed: %s", scale, e)
                continue

        # Fallback: plain forward pass
        if not all_predictions:
            try:
                model_input = tf.image.resize(
                    image_batch,
                    (self.expected_height, self.expected_width),
                    method='bilinear',
                )
                logits = self.model(model_input, training=False)
                if tf.shape(logits)[1] != height or tf.shape(logits)[2] != width:
                    logits = tf.image.resize(logits, original_size, method='bilinear')
                return logits
            except Exception:
                dummy = list(image_batch.shape)
                dummy[-1] = 5
                return tf.zeros(dummy)

        # Average probabilities → convert back to logits
        fused = tf.reduce_mean(tf.stack(all_predictions, axis=0), axis=0)
        eps = 1e-7
        fused = tf.clip_by_value(fused, eps, 1.0 - eps)
        logits = tf.math.log(fused)
        return logits
