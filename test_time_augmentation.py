import numpy as np
import tensorflow as tf

class TestTimeAugmentation:
    """
    Test-Time Augmentation (TTA) for semantic segmentation models.
    Applies multiple transformations during inference and averages predictions
    for more robust results, particularly for rare classes.

    This significantly improves mIoU for oil spill and ship detection.
    """

    def __init__(self, model, num_augmentations=8, use_flips=True, use_scales=True,
                 use_rotations=True, include_original=True):
        """
        Initialize the TTA class.

        Args:
            model: The TensorFlow model to use for prediction
            num_augmentations: Maximum number of augmentations to apply
            use_flips: Whether to use horizontal/vertical flips
            use_scales: Whether to use multi-scale predictions
            use_rotations: Whether to use rotations
            include_original: Whether to include the original image
        """
        self.model = model
        self.num_augmentations = num_augmentations
        self.use_flips = use_flips
        self.use_scales = use_scales
        self.use_rotations = use_rotations
        self.include_original = include_original

        # Create scaling factors
        self.scales = [0.75, 1.0, 1.25] if use_scales else [1.0]

        # Define rotation angles in radians
        self.rotations = [0, np.pi/12, -np.pi/12, np.pi/6, -np.pi/6] if use_rotations else [0]

        print(f"TTA configured with {num_augmentations} augmentations")

    def _augment_image(self, image, aug_type):
        """
        Apply a specific augmentation to an image.

        Args:
            image: Input image tensor
            aug_type: Type of augmentation to apply

        Returns:
            Augmented image and transformation info for reversal
        """
        # Get expected input shape from model
        expected_height, expected_width = self.model.input_shape[1:3]

        # Apply different augmentations based on type
        if aug_type == 'original':
            return image, {'type': 'original'}

        elif aug_type == 'h_flip':
            return tf.image.flip_left_right(image), {'type': 'h_flip'}

        elif aug_type == 'v_flip':
            return tf.image.flip_up_down(image), {'type': 'v_flip'}

        elif aug_type.startswith('rotate_'):
            angle = float(aug_type.split('_')[1])
            from augmentation import rotate_image
            return rotate_image(image, angle, 'bilinear'), {'type': 'rotate', 'angle': -angle}

        elif aug_type.startswith('scale_'):
            scale = float(aug_type.split('_')[1])

            # Original image dimensions
            orig_height, orig_width = tf.shape(image)[0], tf.shape(image)[1]

            # Scale image
            scaled_height = tf.cast(tf.cast(orig_height, tf.float32) * scale, tf.int32)
            scaled_width = tf.cast(tf.cast(orig_width, tf.float32) * scale, tf.int32)

            # Resize image to scaled dimensions
            scaled_img = tf.image.resize(image, [scaled_height, scaled_width],
                                          method='bilinear')

            # Resize back to expected input dimensions
            resized_img = tf.image.resize(scaled_img, [expected_height, expected_width],
                                          method='bilinear')

            return resized_img, {'type': 'scale', 'scale': scale,
                                'orig_shape': [orig_height, orig_width]}

        # Combinations
        elif aug_type == 'h_flip_v_flip':
            flipped = tf.image.flip_left_right(image)
            return tf.image.flip_up_down(flipped), {'type': 'h_flip_v_flip'}

        else:
            print(f"Unknown augmentation type: {aug_type}")
            return image, {'type': 'original'}

    def _reverse_augmentation(self, pred, aug_info):
        """
        Reverse an augmentation applied to a prediction.

        Args:
            pred: Model prediction tensor
            aug_info: Augmentation information for reversal

        Returns:
            De-augmented prediction
        """
        aug_type = aug_info['type']

        if aug_type == 'original':
            return pred

        elif aug_type == 'h_flip':
            return tf.image.flip_left_right(pred)

        elif aug_type == 'v_flip':
            return tf.image.flip_up_down(pred)

        elif aug_type == 'rotate':
            angle = aug_info['angle']
            from augmentation import rotate_image
            return rotate_image(pred, angle, 'bilinear')

        elif aug_type == 'scale':
            # For scale, we need to resize back to original dimensions
            orig_shape = aug_info['orig_shape']
            return tf.image.resize(pred, orig_shape, method='bilinear')

        elif aug_type == 'h_flip_v_flip':
            unflipped_v = tf.image.flip_up_down(pred)
            return tf.image.flip_left_right(unflipped_v)

        else:
            print(f"Unknown augmentation type for reversal: {aug_type}")
            return pred

    def _generate_augmentation_types(self):
        """
        Generate a list of augmentation types to apply.

        Returns:
            List of augmentation type strings
        """
        aug_types = []

        # Add original image if requested
        if self.include_original:
            aug_types.append('original')

        # Add flips
        if self.use_flips:
            aug_types.extend(['h_flip', 'v_flip', 'h_flip_v_flip'])

        # Add rotations
        if self.use_rotations:
            for angle in self.rotations:
                if angle != 0:  # Skip 0 rotation as it's the same as original
                    aug_types.append(f'rotate_{angle}')

        # Add scales
        if self.use_scales:
            for scale in self.scales:
                if scale != 1.0:  # Skip scale 1.0 as it's the same as original
                    aug_types.append(f'scale_{scale}')

        # Limit to requested number of augmentations
        if len(aug_types) > self.num_augmentations:
            # Always keep 'original' if it's included
            if self.include_original:
                orig = ['original']
                others = [t for t in aug_types if t != 'original']
                selected_others = others[:self.num_augmentations-1]
                aug_types = orig + selected_others
            else:
                aug_types = aug_types[:self.num_augmentations]

        return aug_types

    def predict(self, images):
        """
        Apply TTA to a batch of images and get averaged predictions.

        Args:
            images: Batch of input images

        Returns:
            Averaged prediction after TTA
        """
        # Generate augmentation types
        aug_types = self._generate_augmentation_types()
        print(f"Applying {len(aug_types)} TTA transformations: {aug_types}")

        # Store predictions from each augmentation
        all_predictions = []

        # Process each augmentation
        for aug_type in aug_types:
            # Apply augmentation to the batch
            batch_aug_info = []
            batch_aug_images = []

            for i in range(tf.shape(images)[0]):
                aug_image, aug_info = self._augment_image(images[i], aug_type)
                batch_aug_images.append(aug_image)
                batch_aug_info.append(aug_info)

            # Stack augmented images into a batch
            batch_aug_images = tf.stack(batch_aug_images, axis=0)

            # Get predictions for augmented batch
            predictions = self.model(batch_aug_images, training=False)

            # Reverse augmentations for predictions
            batch_reversed_preds = []

            for i in range(tf.shape(predictions)[0]):
                reversed_pred = self._reverse_augmentation(predictions[i], batch_aug_info[i])
                batch_reversed_preds.append(reversed_pred)

            # Stack reversed predictions into a batch
            batch_reversed_preds = tf.stack(batch_reversed_preds, axis=0)

            # Add to all predictions
            all_predictions.append(batch_reversed_preds)

        # Average all predictions
        # Convert logits to probabilities for proper averaging
        all_probs = [tf.nn.softmax(pred, axis=-1) for pred in all_predictions]
        avg_probs = tf.reduce_mean(all_probs, axis=0)

        # Convert back to logits
        epsilon = 1e-7
        avg_probs = tf.clip_by_value(avg_probs, epsilon, 1.0 - epsilon)
        avg_logits = tf.math.log(avg_probs / (1.0 - avg_probs + epsilon))

        return avg_logits
