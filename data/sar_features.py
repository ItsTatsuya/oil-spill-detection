import json
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from tqdm import tqdm

from utils.config import (
    resolve_channel_schema_version,
    resolve_dataset_signature,
    resolve_model_num_channels,
    resolve_sar_channel_names,
    resolve_train_split,
)

logger = logging.getLogger(__name__)


def compute_glcm_contrast(
    amplitude: np.ndarray,
    window_size: int = 7,
    levels: int = 32,
    distances: list[int] | None = None,
    angles: list[float] | None = None,
    # stride=8 keeps texture spatial detail (old default 64 was too coarse for slicks).
    stride: int = 8,
) -> np.ndarray:
    if distances is None:
        distances = [1]
    if angles is None:
        angles = [0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0]

    h, w = amplitude.shape
    pad = window_size // 2
    a_min, a_max = float(amplitude.min()), float(amplitude.max())
    if a_max <= a_min:
        return np.zeros_like(amplitude, dtype=np.float32)

    amp_quant = ((amplitude - a_min) / (a_max - a_min) * (levels - 1)).astype(np.uint8)
    amp_padded = np.pad(amp_quant, pad, mode="reflect")
    contrast_map = np.zeros((h, w), dtype=np.float32)

    sample_h = max(1, int(np.ceil(h / stride)))
    sample_w = max(1, int(np.ceil(w / stride)))
    coarse = np.zeros((sample_h, sample_w), dtype=np.float32)

    for oi, i in enumerate(range(0, h, stride)):
        for oj, j in enumerate(range(0, w, stride)):
            patch = amp_padded[i : i + window_size, j : j + window_size]
            glcm = graycomatrix(
                patch,
                distances=distances,
                angles=angles,
                levels=levels,
                symmetric=True,
                normed=True,
            )
            coarse[oi, oj] = float(graycoprops(glcm, "contrast").mean())

    contrast_map = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_LINEAR)

    return contrast_map


class SARFeatureEncoder:

    VARIANCE_WINDOW = 9

    def __init__(
        self, config: Optional[dict] = None, stats_path: str = "./data/sar_stats.json"
    ) -> None:
        self.config = config or {}
        self.stats_path = Path(stats_path)
        self.stats_path.parent.mkdir(parents=True, exist_ok=True)

        self._amplitude_p1: Optional[float] = None
        self._amplitude_p99: Optional[float] = None
        self._variance_max: Optional[float] = None
        self._gradient_max: Optional[float] = None
        self._glcm_max: Optional[float] = None
        self._fitted = False
        self.channel_names = resolve_sar_channel_names(self.config)

    @property
    def amplitude_p1(self) -> Optional[float]:
        return self._amplitude_p1

    @property
    def amplitude_p99(self) -> Optional[float]:
        return self._amplitude_p99

    @property
    def variance_max(self) -> Optional[float]:
        return self._variance_max

    @variance_max.setter
    def variance_max(self, value: float) -> None:
        self._variance_max = value

    @property
    def gradient_max(self) -> Optional[float]:
        return self._gradient_max

    @gradient_max.setter
    def gradient_max(self, value: float) -> None:
        self._gradient_max = value

    @property
    def fitted(self) -> bool:
        return self._fitted

    @fitted.setter
    def fitted(self, value: bool) -> None:
        self._fitted = value

    @property
    def glcm_max(self) -> Optional[float]:
        return self._glcm_max

    @glcm_max.setter
    def glcm_max(self, value: float) -> None:
        self._glcm_max = value

    def _resolve_stats_metadata(self) -> dict:
        dataset_cfg = self.config.get("dataset", {})
        dataset_root = Path(dataset_cfg.get("root", "./dataset")).expanduser().resolve()
        glcm_stride = int(self.config.get("sar_features", {}).get("glcm_stride", 8))
        return {
            "signature": resolve_dataset_signature(self.config),
            "dataset_root": str(dataset_root),
            "train_split": float(resolve_train_split(self.config)),
            "split_seed": int(dataset_cfg.get("split_seed", 42)),
            "input_channels": int(resolve_model_num_channels(self.config)),
            "channel_schema_version": resolve_channel_schema_version(self.config),
            "channel_names": list(self.channel_names),
            "glcm_stride": glcm_stride,
        }

    def _validate_loaded_metadata(self, stats: dict) -> None:
        if not self.config:
            return

        expected = self._resolve_stats_metadata()
        file_signature = stats.get("signature")
        if file_signature is None:
            logger.warning(
                "SAR stats file %s has no signature metadata. "
                "Legacy stats can load, but provenance cannot be validated.",
                self.stats_path,
            )
            return

        if str(file_signature) != expected["signature"]:
            raise RuntimeError(
                "SAR stats signature mismatch. "
                f"File={file_signature!r}, expected={expected['signature']!r}. "
                "Use the stats file generated by the exact training split/config, "
                "or delete the stale file and regenerate it."
            )

        file_dataset_root = stats.get("dataset_root")
        if file_dataset_root is not None and str(file_dataset_root) != expected["dataset_root"]:
            raise RuntimeError(
                "SAR stats dataset root mismatch. "
                f"File={file_dataset_root!r}, expected={expected['dataset_root']!r}."
            )

        file_glcm_stride = stats.get("glcm_stride")
        if file_glcm_stride is not None and int(file_glcm_stride) != int(
            expected["glcm_stride"]
        ):
            raise RuntimeError(
                "SAR stats glcm_stride mismatch. "
                f"File={file_glcm_stride!r}, expected={expected['glcm_stride']!r}. "
                "Delete the stats file and regenerate after changing glcm_stride."
            )
        if file_glcm_stride is None and "glcm_contrast" in self.channel_names:
            logger.warning(
                "SAR stats file %s has no glcm_stride metadata. "
                "Assuming it matches config glcm_stride=%s.",
                self.stats_path,
                expected["glcm_stride"],
            )

    def _compute_local_variance(self, amplitude: np.ndarray) -> np.ndarray:
        k = self.VARIANCE_WINDOW
        amp64 = amplitude.astype(np.float64)
        mean = cv2.blur(amp64, (k, k))
        mean_sq = cv2.blur(amp64**2, (k, k))
        variance = np.maximum(mean_sq - mean**2, 0.0).astype(np.float32)
        return variance

    def _compute_gradient_magnitude(self, amplitude: np.ndarray) -> np.ndarray:
        img_f64 = amplitude.astype(np.float64)
        gx = cv2.Sobel(img_f64, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img_f64, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(gx**2 + gy**2).astype(np.float32)
        return magnitude

    def set_default_stats(
        self,
        amplitude_p1: float = 0.0,
        amplitude_p99: float = 1.0,
        variance_max: float = 0.01,
        gradient_max: float = 1.0,
        glcm_max: float = 1.0,
    ) -> None:
        if not (0.0 <= amplitude_p1 < amplitude_p99 <= 1.0):
            raise ValueError(
                "amplitude percentiles must satisfy 0 <= p1 < p99 <= 1, "
                f"got p1={amplitude_p1}, p99={amplitude_p99}"
            )
        if variance_max <= 0:
            raise ValueError(f"variance_max must be positive, got {variance_max}")
        if gradient_max <= 0:
            raise ValueError(f"gradient_max must be positive, got {gradient_max}")
        if glcm_max <= 0:
            raise ValueError(f"glcm_max must be positive, got {glcm_max}")
        self._amplitude_p1 = float(amplitude_p1)
        self._amplitude_p99 = float(amplitude_p99)
        self._variance_max = float(variance_max)
        self._gradient_max = float(gradient_max)
        self._glcm_max = float(glcm_max)
        self._fitted = True
        logger.info(
            f"SARFeatureEncoder: using default stats "
            f"(amplitude_p1={amplitude_p1}, amplitude_p99={amplitude_p99}, "
            f"variance_max={variance_max}, gradient_max={gradient_max}, glcm_max={glcm_max}). "
            "Run fit() on real training data for best results."
        )

    def fit(self, image_paths: list) -> None:
        logger.info(
            f"Computing SAR feature statistics from {len(image_paths)} training images..."
        )

        all_amplitude_samples: list[np.ndarray] = []
        all_variance_maxes: list = []
        all_gradient_maxes: list = []
        all_glcm_maxes: list = []

        amp_sample_size = int(
            self.config.get("sar_features", {}).get("amplitude_sample_size", 4096)
        )

        pbar = tqdm(
            image_paths,
            total=len(image_paths),
            desc="SAR stats",
            leave=False,
        )
        for path in pbar:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.warning(f"Could not read image: {path}")
                continue
            amplitude = img.astype(np.float32) / 255.0

            flat_amp = amplitude.reshape(-1)
            if flat_amp.size > amp_sample_size:
                step = max(1, flat_amp.size // amp_sample_size)
                amp_sample = flat_amp[::step][:amp_sample_size]
            else:
                amp_sample = flat_amp
            all_amplitude_samples.append(amp_sample.astype(np.float32, copy=False))

            variance = self._compute_local_variance(amplitude)
            gradient = self._compute_gradient_magnitude(amplitude)
            # Must match transform() stride — otherwise glcm_max is calibrated on a
            # different spatial sampling than inference and silently mis-scales texture.
            glcm_stride = int(
                self.config.get("sar_features", {}).get("glcm_stride", 8)
            )
            glcm_contrast = compute_glcm_contrast(amplitude, stride=glcm_stride)

            all_variance_maxes.append(float(variance.max()))
            all_gradient_maxes.append(float(gradient.max()))
            all_glcm_maxes.append(float(glcm_contrast.max()))

        pbar.close()

        if not all_variance_maxes:
            raise RuntimeError(
                "No images could be read; cannot compute SAR statistics."
            )

        amplitude_values = np.concatenate(all_amplitude_samples)
        self._amplitude_p1 = float(np.percentile(amplitude_values, 1))
        self._amplitude_p99 = float(np.percentile(amplitude_values, 99))
        if self._amplitude_p99 <= self._amplitude_p1:
            self._amplitude_p1 = 0.0
            self._amplitude_p99 = 1.0

        self._variance_max = float(np.percentile(all_variance_maxes, 99)) + 1e-8
        self._gradient_max = float(np.percentile(all_gradient_maxes, 99)) + 1e-8
        self._glcm_max = float(np.percentile(all_glcm_maxes, 99)) + 1e-8
        self._fitted = True

        logger.info(
            f"SAR statistics: amplitude_p1={self._amplitude_p1:.6f}, "
            f"amplitude_p99={self._amplitude_p99:.6f}, "
            f"variance_max={self._variance_max:.6f}, "
            f"gradient_max={self._gradient_max:.6f}, "
            f"glcm_max={self._glcm_max:.6f}"
        )
        self.save_stats()

    def transform(self, amplitude: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError(
                "SARFeatureEncoder must be fitted before use. "
                "Call fit() on training images or load_stats() from a saved file."
            )

        if amplitude.ndim == 3:
            amplitude = amplitude.squeeze(-1)
        assert amplitude.ndim == 2, f"Expected 2D array, got shape {amplitude.shape}"

        amp = amplitude.clip(0.0, 1.0).astype(np.float32, copy=False)
        needed = set(self.channel_names)
        feature_map: dict[str, np.ndarray] = {
            "amplitude": amp,
        }
        # Only compute expensive features that are actually requested so an
        # amplitude-repeat 3-channel baseline stays cheap.
        if "local_variance" in needed:
            variance = self._compute_local_variance(amplitude)
            feature_map["local_variance"] = (
                (variance / self._variance_max)
                .clip(0.0, 1.0)
                .astype(np.float32, copy=False)
            )
        if "gradient_magnitude" in needed:
            gradient = self._compute_gradient_magnitude(amplitude)
            feature_map["gradient_magnitude"] = (
                (gradient / self._gradient_max)
                .clip(0.0, 1.0)
                .astype(np.float32, copy=False)
            )
        if "glcm_contrast" in needed:
            glcm_stride = int(
                self.config.get("sar_features", {}).get("glcm_stride", 8)
            )
            glcm_contrast = compute_glcm_contrast(amplitude, stride=glcm_stride)
            feature_map["glcm_contrast"] = (
                (glcm_contrast / self._glcm_max)
                .clip(0.0, 1.0)
                .astype(np.float32, copy=False)
            )

        try:
            ordered = [feature_map[name] for name in self.channel_names]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported SAR channel requested: {exc.args[0]!r}. "
                f"Supported channels: {sorted(feature_map)} "
                f"(requested schema: {self.channel_names})"
            ) from exc
        features = np.stack(ordered, axis=-1)

        assert features.shape[-1] == len(self.channel_names), (
            f"Expected {len(self.channel_names)} channels, got {features.shape}"
        )
        assert features.dtype == np.float32, f"Expected float32, got {features.dtype}"

        return features

    def save_stats(self) -> None:
        if not self._fitted:
            raise RuntimeError("Cannot save stats before fitting. Call fit() first.")

        stats = {
            "schema_version": 3,
            "amplitude_p1": self._amplitude_p1,
            "amplitude_p99": self._amplitude_p99,
            "variance_max": self._variance_max,
            "gradient_max": self._gradient_max,
            "glcm_max": self._glcm_max,
            "channel_names": list(self.channel_names),
        }
        if self.config:
            stats.update(self._resolve_stats_metadata())
        tmp_path = self.stats_path.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump(stats, f, indent=2)
        tmp_path.replace(self.stats_path)
        logger.info(f"SAR statistics saved to {self.stats_path}")

    def load_stats(self) -> None:
        if not self.stats_path.exists():
            raise FileNotFoundError(
                f"SAR stats file not found at {self.stats_path}. "
                "Run fit() on training data first."
            )

        with open(self.stats_path, "r") as f:
            stats = json.load(f)

        required_keys = ("variance_max", "gradient_max", "glcm_max")
        missing = [key for key in required_keys if key not in stats]
        if missing:
            raise RuntimeError(
                f"SAR stats file is missing required keys: {missing}. "
                "Delete the file and regenerate stats with a fresh training run."
            )

        self._validate_loaded_metadata(stats)

        self._amplitude_p1 = float(
            stats.get("amplitude_p1", stats.get("ratio_p1", 0.0))
        )
        self._amplitude_p99 = float(
            stats.get("amplitude_p99", stats.get("ratio_p99", 1.0))
        )
        if self._amplitude_p99 <= self._amplitude_p1:
            logger.warning(
                "Invalid amplitude percentile range in stats file (p1=%.6f, p99=%.6f). Falling back to [0, 1].",
                self._amplitude_p1,
                self._amplitude_p99,
            )
            self._amplitude_p1 = 0.0
            self._amplitude_p99 = 1.0

        self._variance_max = stats["variance_max"]
        self._gradient_max = stats["gradient_max"]
        self._glcm_max = stats["glcm_max"]
        self._fitted = True
        logger.info(f"SAR statistics loaded from {self.stats_path}")

    def visualize(self, amplitude: np.ndarray, save_path: str) -> None:
        import matplotlib.pyplot as plt

        if not self._fitted:
            raise RuntimeError("Call fit() or load_stats() before visualizing.")

        features = self.transform(amplitude)

        title_map = {
            "amplitude": "SAR Amplitude\n(Oil/look-alike: dark; Ships: bright)",
            "local_variance": "Local Variance (9x9)\n(Oil: near-zero; Open sea: high)",
            "gradient_magnitude": "Gradient Magnitude\n(Ships: sharp; Oil: diffuse)",
            "glcm_contrast": "GLCM Contrast\n(2nd-order texture cue)",
        }
        cmap_map = {
            "amplitude": "gray",
            "local_variance": "viridis",
            "gradient_magnitude": "plasma",
            "glcm_contrast": "magma",
        }
        fig, axes = plt.subplots(
            1,
            len(self.channel_names),
            figsize=(6 * len(self.channel_names), 6),
        )
        if len(self.channel_names) == 1:
            axes = [axes]

        for ax, ch_name, ch in zip(
            axes, self.channel_names, range(len(self.channel_names))
        ):
            cmap = cmap_map.get(ch_name, "viridis")
            im = ax.imshow(features[:, :, ch], cmap=cmap, vmin=0, vmax=1)
            ax.set_title(title_map.get(ch_name, ch_name), fontsize=11)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.suptitle("SAR Multi-Feature Encoding", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"SAR feature visualization saved to {save_path}")
