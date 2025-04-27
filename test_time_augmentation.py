import numpy as np
import tensorflow as tf

class TestTimeAugmentation:
    def __init__(self, model, num_augmentations=8, use_flips=True, use_scales=True,
                 use_rotations=True, include_original=True):
        self.model = model
        self.num_augmentations = num_augmentations
        self.use_flips = use_flips
        self.use_scales = use_scales
        self.use_rotations = use_rotations
        self.include_original = include_original

        self.scales = [0.75, 1.0, 1.25] if use_scales else [1.0]

        self.rotations = [0, np.pi/12, -np.pi/12, np.pi/6, -np.pi/6] if use_rotations else [0]

        self.expected_height = model.input_shape[1]
        self.expected_width = model.input_shape[2]
        self.expected_channels = model.input_shape[3]

        self.compute_dtype = tf.keras.mixed_precision.global_policy().compute_dtype
        if self.compute_dtype == 'mixed_float16':
            self.compute_dtype = tf.float16
        else:
            self.compute_dtype = tf.float32

        print(f"TTA configured with {num_augmentations} augmentations")

    def _augment_image(self, image, aug_type):
        image = tf.cast(image, tf.float32)

        orig_height, orig_width = tf.shape(image)[0], tf.shape(image)[1]
        orig_size = (orig_height, orig_width)

        if aug_type == 'original':
            if orig_height != self.expected_height or orig_width != self.expected_width:
                image = tf.image.resize(image, [self.expected_height, self.expected_width], method='bilinear')
            return tf.cast(image, self.compute_dtype), {'type': 'original', 'orig_size': orig_size}

        elif aug_type == 'h_flip':
            flipped = tf.image.flip_left_right(image)
            if orig_height != self.expected_height or orig_width != self.expected_width:
                flipped = tf.image.resize(flipped, [self.expected_height, self.expected_width], method='bilinear')
            return tf.cast(flipped, self.compute_dtype), {'type': 'h_flip', 'orig_size': orig_size}

        elif aug_type == 'v_flip':
            flipped = tf.image.flip_up_down(image)
            if orig_height != self.expected_height or orig_width != self.expected_width:
                flipped = tf.image.resize(flipped, [self.expected_height, self.expected_width], method='bilinear')
            return tf.cast(flipped, self.compute_dtype), {'type': 'v_flip', 'orig_size': orig_size}

        elif aug_type.startswith('rotate_'):
            angle = float(aug_type.split('_')[1])
            from augmentation import rotate_image
            rotated = rotate_image(image, angle, 'bilinear')

            if tf.shape(rotated)[0] != self.expected_height or tf.shape(rotated)[1] != self.expected_width:
                rotated = tf.image.resize(rotated, [self.expected_height, self.expected_width], method='bilinear')

            if rotated.shape[-1] != self.expected_channels:
                if self.expected_channels == 1:
                    rotated = tf.expand_dims(tf.reduce_mean(rotated, axis=-1), axis=-1)
                elif self.expected_channels == 3 and rotated.shape[-1] == 1:
                    rotated = tf.concat([rotated, rotated, rotated], axis=-1)

            return tf.cast(rotated, self.compute_dtype), {'type': 'rotate', 'angle': -angle, 'orig_size': orig_size}

        elif aug_type.startswith('scale_'):
            scale = float(aug_type.split('_')[1])

            if scale != 1.0:
                scaled_height = tf.cast(tf.cast(orig_height, tf.float32) * scale, tf.int32)
                scaled_width = tf.cast(tf.cast(orig_width, tf.float32) * scale, tf.int32)
                scaled_img = tf.image.resize(image, [scaled_height, scaled_width], method='bilinear')
            else:
                scaled_img = image

            resized_img = tf.image.resize(scaled_img, [self.expected_height, self.expected_width], method='bilinear')

            if resized_img.shape[-1] != self.expected_channels:
                if self.expected_channels == 1:
                    resized_img = tf.expand_dims(tf.reduce_mean(resized_img, axis=-1), axis=-1)
                elif self.expected_channels == 3 and resized_img.shape[-1] == 1:
                    resized_img = tf.concat([resized_img, resized_img, resized_img], axis=-1)

            return tf.cast(resized_img, self.compute_dtype), {'type': 'scale', 'scale': scale, 'orig_size': orig_size}

        elif aug_type == 'h_flip_v_flip':
            flipped = tf.image.flip_left_right(image)
            flipped = tf.image.flip_up_down(flipped)

            if orig_height != self.expected_height or orig_width != self.expected_width:
                flipped = tf.image.resize(flipped, [self.expected_height, self.expected_width], method='bilinear')

            return tf.cast(flipped, self.compute_dtype), {'type': 'h_flip_v_flip', 'orig_size': orig_size}

        else:
            if orig_height != self.expected_height or orig_width != self.expected_width:
                image = tf.image.resize(image, [self.expected_height, self.expected_width], method='bilinear')
            return tf.cast(image, self.compute_dtype), {'type': 'original', 'orig_size': orig_size}

    def _reverse_augmentation(self, pred, aug_info):
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
            if orig_size is not None:
                return tf.image.resize(pred, orig_size, method='bilinear')
            return pred

    def _generate_augmentation_types(self):
        aug_types = []

        if self.include_original:
            aug_types.append('original')

        if self.use_flips:
            aug_types.extend(['h_flip', 'v_flip', 'h_flip_v_flip'])

        if self.use_rotations:
            for angle in self.rotations:
                if angle != 0:
                    aug_types.append(f'rotate_{angle}')

        if self.use_scales:
            for scale in self.scales:
                if scale != 1.0:
                    aug_types.append(f'scale_{scale}')

        if len(aug_types) > self.num_augmentations:
            if self.include_original:
                orig = ['original']
                others = [t for t in aug_types if t != 'original']
                selected_others = others[:self.num_augmentations-1]
                aug_types = orig + selected_others
            else:
                aug_types = aug_types[:self.num_augmentations]

        return aug_types

    def predict(self, images):
        processed_images = tf.cast(images, tf.float32)

        if processed_images.shape[-1] != self.expected_channels:
            if self.expected_channels == 1 and processed_images.shape[-1] > 1:
                processed_images = tf.expand_dims(tf.reduce_mean(processed_images, axis=-1), axis=-1)
            elif self.expected_channels == 3 and processed_images.shape[-1] == 1:
                processed_images = tf.concat([processed_images, processed_images, processed_images], axis=-1)

        aug_types = self._generate_augmentation_types()

        all_predictions = []

        for aug_type in aug_types:
            try:
                batch_aug_info = []
                batch_aug_images = []

                for i in range(tf.shape(processed_images)[0]):
                    aug_image, aug_info = self._augment_image(processed_images[i], aug_type)

                    if tf.shape(aug_image)[0] != self.expected_height or tf.shape(aug_image)[1] != self.expected_width:
                        aug_image = tf.image.resize(aug_image, [self.expected_height, self.expected_width], method='bilinear')
                        aug_image = tf.cast(aug_image, self.compute_dtype)

                    if aug_image.shape[-1] != self.expected_channels:
                        if self.expected_channels == 1:
                            aug_image = tf.expand_dims(tf.reduce_mean(aug_image, axis=-1), axis=-1)
                        elif self.expected_channels == 3 and aug_image.shape[-1] == 1:
                            aug_image = tf.concat([aug_image, aug_image, aug_image], axis=-1)
                        aug_image = tf.cast(aug_image, self.compute_dtype)

                    batch_aug_images.append(aug_image)
                    batch_aug_info.append(aug_info)

                batch_aug_images = tf.stack(batch_aug_images, axis=0)

                if batch_aug_images.shape[1] != self.expected_height or batch_aug_images.shape[2] != self.expected_width:
                    raise ValueError(f"Wrong dimensions: {batch_aug_images.shape[1:3]} vs {(self.expected_height, self.expected_width)}")

                if batch_aug_images.shape[3] != self.expected_channels:
                    raise ValueError(f"Wrong channels: {batch_aug_images.shape[3]} vs {self.expected_channels}")

                if tf.reduce_max(batch_aug_images) > 1.0:
                    batch_aug_images = batch_aug_images / 255.0

                batch_aug_images = tf.cast(batch_aug_images, self.compute_dtype)

                predictions = self.model(batch_aug_images, training=False)

                predictions = tf.cast(predictions, tf.float32)

                batch_reversed_preds = []

                for i in range(tf.shape(predictions)[0]):
                    reversed_pred = self._reverse_augmentation(predictions[i], batch_aug_info[i])
                    batch_reversed_preds.append(reversed_pred)

                batch_reversed_preds = tf.convert_to_tensor(batch_reversed_preds)

                all_predictions.append(batch_reversed_preds)

            except Exception as e:
                continue

        if not all_predictions:
            try:
                if processed_images.shape[1] != self.expected_height or processed_images.shape[2] != self.expected_width:
                    processed_images = tf.image.resize(processed_images, [self.expected_height, self.expected_width], method='bilinear')

                if processed_images.shape[3] != self.expected_channels:
                    if self.expected_channels == 1:
                        processed_images = tf.expand_dims(tf.reduce_mean(processed_images, axis=-1), axis=-1)
                    elif self.expected_channels == 3 and processed_images.shape[3] == 1:
                        processed_images = tf.concat([processed_images, processed_images, processed_images], axis=-1)

                processed_images = tf.cast(processed_images, self.compute_dtype)

                return self.model(processed_images, training=False)
            except Exception as e:
                dummy_shape = list(processed_images.shape)
                dummy_shape[-1] = self.model.output_shape[-1]
                return tf.zeros(dummy_shape)

        all_probs = []
        for pred in all_predictions:
            if isinstance(pred, np.ndarray):
                pred = tf.convert_to_tensor(pred.astype(np.float32))
            else:
                pred = tf.cast(pred, tf.float32)

            probs = tf.nn.softmax(pred, axis=-1)
            all_probs.append(probs)

        stacked_probs = tf.stack(all_probs, axis=0)
        avg_probs = tf.reduce_mean(stacked_probs, axis=0)

        epsilon = 1e-7
        avg_probs = tf.clip_by_value(avg_probs, epsilon, 1.0 - epsilon)
        avg_logits = tf.math.log(avg_probs)

        return avg_logits
