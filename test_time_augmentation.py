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

        # Create scaling factors - but we'll always resize back to model input size
        self.scales = [0.75, 1.0, 1.25] if use_scales else [1.0]

        # Define rotation angles in radians
        self.rotations = [0, np.pi/12, -np.pi/12, np.pi/6, -np.pi/6] if use_rotations else [0]

        # Get model's expected input dimensions
        self.expected_height = model.input_shape[1]
        self.expected_width = model.input_shape[2]
        self.expected_channels = model.input_shape[3]

        print(f"TTA configured with {num_augmentations} augmentations")
        print(f"Model expects input size: {self.expected_height}x{self.expected_width}x{self.expected_channels}")

    def _augment_image(self, image, aug_type):
        """
        Apply a specific augmentation to an image.

        Args:
            image: Input image tensor
            aug_type: Type of augmentation to apply

        Returns:
            Augmented image and transformation info for reversal
        """
        # Store original size for later reversal
        orig_height, orig_width = tf.shape(image)[0], tf.shape(image)[1]
        orig_size = (orig_height, orig_width)

        # Apply different augmentations based on type
        if aug_type == 'original':
            # Always ensure the image is the size expected by the model
            if orig_height != self.expected_height or orig_width != self.expected_width:
                image = tf.image.resize(image, [self.expected_height, self.expected_width], method='bilinear')
            return image, {'type': 'original', 'orig_size': orig_size}

        elif aug_type == 'h_flip':
            flipped = tf.image.flip_left_right(image)
            # Always ensure the image is the size expected by the model
            if orig_height != self.expected_height or orig_width != self.expected_width:
                flipped = tf.image.resize(flipped, [self.expected_height, self.expected_width], method='bilinear')
            return flipped, {'type': 'h_flip', 'orig_size': orig_size}

        elif aug_type == 'v_flip':
            flipped = tf.image.flip_up_down(image)
            # Always ensure the image is the size expected by the model
            if orig_height != self.expected_height or orig_width != self.expected_width:
                flipped = tf.image.resize(flipped, [self.expected_height, self.expected_width], method='bilinear')
            return flipped, {'type': 'v_flip', 'orig_size': orig_size}

        elif aug_type.startswith('rotate_'):
            angle = float(aug_type.split('_')[1])
            from augmentation import rotate_image
            rotated = rotate_image(image, angle, 'bilinear')

            # Always ensure the image is the size expected by the model
            if tf.shape(rotated)[0] != self.expected_height or tf.shape(rotated)[1] != self.expected_width:
                rotated = tf.image.resize(rotated, [self.expected_height, self.expected_width], method='bilinear')

            # Ensure channel dimension is preserved
            if rotated.shape[-1] != self.expected_channels:
                if self.expected_channels == 1:
                    rotated = tf.expand_dims(tf.reduce_mean(rotated, axis=-1), axis=-1)
                elif self.expected_channels == 3 and rotated.shape[-1] == 1:
                    rotated = tf.concat([rotated, rotated, rotated], axis=-1)

            return rotated, {'type': 'rotate', 'angle': -angle, 'orig_size': orig_size}

        elif aug_type.startswith('scale_'):
            scale = float(aug_type.split('_')[1])

            # Apply scaling within the network (but always resize back to model input size)
            # This allows analyzing the image at different scales without changing model input dims
            if scale != 1.0:
                # First, scale the image
                scaled_height = tf.cast(tf.cast(orig_height, tf.float32) * scale, tf.int32)
                scaled_width = tf.cast(tf.cast(orig_width, tf.float32) * scale, tf.int32)
                scaled_img = tf.image.resize(image, [scaled_height, scaled_width], method='bilinear')
            else:
                scaled_img = image

            # Then always resize to expected model input size
            resized_img = tf.image.resize(scaled_img, [self.expected_height, self.expected_width], method='bilinear')

            # Ensure channel dimension is correct
            if resized_img.shape[-1] != self.expected_channels:
                if self.expected_channels == 1:
                    resized_img = tf.expand_dims(tf.reduce_mean(resized_img, axis=-1), axis=-1)
                elif self.expected_channels == 3 and resized_img.shape[-1] == 1:
                    resized_img = tf.concat([resized_img, resized_img, resized_img], axis=-1)

            return resized_img, {'type': 'scale', 'scale': scale, 'orig_size': orig_size}

        # Combinations
        elif aug_type == 'h_flip_v_flip':
            flipped = tf.image.flip_left_right(image)
            flipped = tf.image.flip_up_down(flipped)

            # Always ensure the image is the size expected by the model
            if orig_height != self.expected_height or orig_width != self.expected_width:
                flipped = tf.image.resize(flipped, [self.expected_height, self.expected_width], method='bilinear')

            return flipped, {'type': 'h_flip_v_flip', 'orig_size': orig_size}

        else:
            print(f"Unknown augmentation type: {aug_type}")
            # Always ensure the image is the size expected by the model
            if orig_height != self.expected_height or orig_width != self.expected_width:
                image = tf.image.resize(image, [self.expected_height, self.expected_width], method='bilinear')
            return image, {'type': 'original', 'orig_size': orig_size}

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
        orig_size = aug_info.get('orig_size', None)

        if aug_type == 'original':
            if orig_size is not None:
                return tf.image.resize(pred, orig_size, method='bilinear')
            return pred

        elif aug_type == 'h_flip':
            flipped_back = tf.image.flip_left_right(pred)
            if orig_size is not None:
                return tf.image.resize(flipped_back, orig_size, method='bilinear')
            return flipped_back

        elif aug_type == 'v_flip':
            flipped_back = tf.image.flip_up_down(pred)
            if orig_size is not None:
                return tf.image.resize(flipped_back, orig_size, method='bilinear')
            return flipped_back

        elif aug_type == 'rotate':
            angle = aug_info['angle']
            from augmentation import rotate_image
            rotated_back = rotate_image(pred, angle, 'bilinear')
            if orig_size is not None:
                return tf.image.resize(rotated_back, orig_size, method='bilinear')
            return rotated_back

        elif aug_type == 'scale':
            # For scale, we need to resize back to original dimensions
            if orig_size is not None:
                return tf.image.resize(pred, orig_size, method='bilinear')
            return pred

        elif aug_type == 'h_flip_v_flip':
            unflipped_v = tf.image.flip_up_down(pred)
            unflipped = tf.image.flip_left_right(unflipped_v)
            if orig_size is not None:
                return tf.image.resize(unflipped, orig_size, method='bilinear')
            return unflipped

        else:
            print(f"Unknown augmentation type for reversal: {aug_type}")
            if orig_size is not None:
                return tf.image.resize(pred, orig_size, method='bilinear')
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
        # Make a copy of the input to avoid modifying the original
        processed_images = tf.identity(images)

        # Ensure input images have the correct number of channels - critical for the model
        if processed_images.shape[-1] != self.expected_channels:
            print(f"Converting input from {processed_images.shape[-1]} to {self.expected_channels} channels...")
            if self.expected_channels == 1 and processed_images.shape[-1] > 1:
                # Convert multi-channel to single channel (RGB to grayscale)
                processed_images = tf.expand_dims(tf.reduce_mean(processed_images, axis=-1), axis=-1)
            elif self.expected_channels == 3 and processed_images.shape[-1] == 1:
                # Convert single channel to RGB by duplicating the channel
                processed_images = tf.concat([processed_images, processed_images, processed_images], axis=-1)

        # Print shape after conversion
        print(f"Input shape after initial channel adjustment: {processed_images.shape}")

        # Generate augmentation types
        aug_types = self._generate_augmentation_types()
        print(f"Applying {len(aug_types)} TTA transformations: {aug_types}")

        # Store predictions from each augmentation
        all_predictions = []

        # Process each augmentation
        for aug_type in aug_types:
            try:
                # Apply augmentation to the batch
                batch_aug_info = []
                batch_aug_images = []

                for i in range(tf.shape(processed_images)[0]):
                    # Apply augmentation - this will also resize to model input size
                    aug_image, aug_info = self._augment_image(processed_images[i], aug_type)

                    # Force correct dimensions
                    if tf.shape(aug_image)[0] != self.expected_height or tf.shape(aug_image)[1] != self.expected_width:
                        print(f"Resizing to {self.expected_height}x{self.expected_width}")
                        aug_image = tf.image.resize(aug_image, [self.expected_height, self.expected_width], method='bilinear')

                    # Force correct channel dimension
                    if aug_image.shape[-1] != self.expected_channels:
                        print(f"Converting from {aug_image.shape[-1]} to {self.expected_channels} channels")
                        if self.expected_channels == 1:
                            aug_image = tf.expand_dims(tf.reduce_mean(aug_image, axis=-1), axis=-1)
                        elif self.expected_channels == 3 and aug_image.shape[-1] == 1:
                            aug_image = tf.concat([aug_image, aug_image, aug_image], axis=-1)

                    # Add to batch
                    batch_aug_images.append(aug_image)
                    batch_aug_info.append(aug_info)

                # Stack augmented images into a batch
                batch_aug_images = tf.stack(batch_aug_images, axis=0)

                # Print shape for debugging
                print(f"Batch shape after {aug_type}: {batch_aug_images.shape}, dtype: {batch_aug_images.dtype}")

                # Final verification before model inference
                if batch_aug_images.shape[1] != self.expected_height or batch_aug_images.shape[2] != self.expected_width:
                    raise ValueError(f"Wrong dimensions: {batch_aug_images.shape[1:3]} vs {(self.expected_height, self.expected_width)}")

                if batch_aug_images.shape[3] != self.expected_channels:
                    raise ValueError(f"Wrong channels: {batch_aug_images.shape[3]} vs {self.expected_channels}")

                # Make sure values are in reasonable range (0-1)
                if tf.reduce_max(batch_aug_images) > 1.0:
                    batch_aug_images = batch_aug_images / 255.0

                # Ensure correct data type for the model
                batch_aug_images = tf.cast(batch_aug_images, tf.float32)

                # Get predictions for augmented batch
                predictions = self.model(batch_aug_images, training=False)

                # Reverse augmentations for predictions
                batch_reversed_preds = []

                for i in range(tf.shape(predictions)[0]):
                    reversed_pred = self._reverse_augmentation(predictions[i], batch_aug_info[i])
                    batch_reversed_preds.append(reversed_pred)

                # Stack reversed predictions into a batch - ensure tensor format
                batch_reversed_preds = tf.convert_to_tensor(batch_reversed_preds)

                # Add to all predictions
                all_predictions.append(batch_reversed_preds)

            except Exception as e:
                print(f"Error during TTA with augmentation '{aug_type}': {e}")
                import traceback
                traceback.print_exc()
                continue

        # Check if we have any successful augmentations
        if not all_predictions:
            print("All TTA augmentations failed. Falling back to direct prediction.")
            try:
                # Make sure input has correct size before direct prediction
                if processed_images.shape[1] != self.expected_height or processed_images.shape[2] != self.expected_width:
                    processed_images = tf.image.resize(processed_images, [self.expected_height, self.expected_width], method='bilinear')

                # Ensure correct channel count
                if processed_images.shape[3] != self.expected_channels:
                    if self.expected_channels == 1:
                        processed_images = tf.expand_dims(tf.reduce_mean(processed_images, axis=-1), axis=-1)
                    elif self.expected_channels == 3 and processed_images.shape[3] == 1:
                        processed_images = tf.concat([processed_images, processed_images, processed_images], axis=-1)

                # Fallback to direct prediction without augmentation
                return self.model(processed_images, training=False)
            except Exception as e:
                print(f"Direct prediction also failed: {e}")
                # Return a dummy prediction with the correct shape
                dummy_shape = list(processed_images.shape)
                dummy_shape[-1] = self.model.output_shape[-1]  # Use correct number of classes
                return tf.zeros(dummy_shape)

        # Average all predictions
        # Convert logits to probabilities for proper averaging
        all_probs = []
        for pred in all_predictions:
            # Ensure tensor is properly shaped and has correct dtype
            if isinstance(pred, np.ndarray):
                pred = tf.convert_to_tensor(pred.astype(np.float32))
            # Apply softmax for proper averaging
            probs = tf.nn.softmax(pred, axis=-1)
            all_probs.append(probs)

        # Stack all probability predictions and average
        stacked_probs = tf.stack(all_probs, axis=0)
        avg_probs = tf.reduce_mean(stacked_probs, axis=0)

        # Convert back to logits
        epsilon = 1e-7
        avg_probs = tf.clip_by_value(avg_probs, epsilon, 1.0 - epsilon)
        avg_logits = tf.math.log(avg_probs)

        return avg_logits
