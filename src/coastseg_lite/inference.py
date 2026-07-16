"""The fixed CoastSeg RGB SegFormer inference kernel."""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
# A user shell may request the separate `tf_keras` compatibility package. This
# project pins TensorFlow/Keras 2.15 together and must use that bundled Keras.
os.environ["TF_USE_LEGACY_KERAS"] = "0"


@dataclass(frozen=True)
class PreparedImage:
    model_input: np.ndarray
    source_shape: tuple[int, int]


def standardize(image: np.ndarray) -> np.ndarray:
    """Match Doodleverse's whole-image adjusted standardization."""
    pixel_count = image.shape[0] * image.shape[1]
    adjusted_std = max(float(np.std(image)), 1.0 / np.sqrt(pixel_count))
    standardized = (image - float(np.mean(image))) / adjusted_std
    if standardized.ndim == 2:
        standardized = np.dstack((standardized, standardized, standardized))
    return standardized.astype(np.float32)


def prepare_image(
    image_rgb: np.ndarray, target_size: tuple[int, int] = (512, 512)
) -> PreparedImage:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, got {image_rgb.shape}")
    if image_rgb.dtype != np.uint8:
        raise ValueError(f"Expected uint8 RGB image, got {image_rgb.dtype}")
    target_height, target_width = target_size
    resized = np.asarray(
        Image.fromarray(image_rgb).resize(
            (target_width, target_height), resample=Image.Resampling.BILINEAR
        )
    )
    chw = np.transpose(standardize(resized), (2, 0, 1))
    return PreparedImage(chw, tuple(image_rgb.shape[:2]))


def load_tensorflow_model(model_path: Path) -> Any:
    """Construct pinned SegFormer B0 locally and load the CoastSeg weights."""
    import tensorflow as tf
    from transformers import SegformerConfig, TFSegformerForSemanticSegmentation

    from coastseg_lite.model_store import load_manifest

    tf.get_logger().setLevel("ERROR")
    manifest = load_manifest()
    config_path = files("coastseg_lite").joinpath(
        manifest["architecture"]["bundled_config"]
    )
    config = SegformerConfig.from_json_file(str(config_path))
    model = TFSegformerForSemanticSegmentation(config)
    model(tf.zeros((1, 3, 512, 512), dtype=tf.float32), training=False)
    model.load_weights(model_path)
    return model


def _forward_logits(model: Any, model_input: np.ndarray, nclasses: int = 4) -> Any:
    import tensorflow as tf

    model_object = model[0] if isinstance(model, tuple) else model
    batched = tf.expand_dims(model_input, 0)
    try:
        result = model_object(batched, training=False)
    except TypeError:
        result = model_object(batched)
    logits = result.logits if hasattr(result, "logits") else result
    logits_array = np.asarray(logits, dtype=np.float32)
    if logits_array.ndim == 4 and logits_array.shape[0] == 1:
        logits_array = logits_array[0]
    if logits_array.ndim == 3 and logits_array.shape[-1] == nclasses:
        return logits_array
    if logits_array.ndim == 3 and logits_array.shape[0] == nclasses:
        return np.transpose(logits_array, (1, 2, 0))
    raise ValueError(f"Unsupported SegFormer logits shape: {logits_array.shape}")


def infer_array(image_rgb: np.ndarray, model: Any) -> np.ndarray:
    """Return CoastSeg class IDs on the source image pixel grid."""
    import tensorflow as tf

    prepared = prepare_image(image_rgb)
    logits = _forward_logits(model, prepared.model_input)
    resized = tf.image.resize(logits, size=prepared.source_shape, method="bilinear")
    return tf.argmax(resized, axis=-1).numpy().astype(np.uint8)


def infer_image(path: Path, model: Any) -> tuple[np.ndarray, tuple[int, int, int]]:
    with Image.open(path) as image:
        image_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return infer_array(image_rgb, model), tuple(image_rgb.shape)
