from __future__ import annotations

import numpy as np
import pytest

from coastseg_lite.inference import prepare_image, standardize


def synthetic_rgb(height: int = 72, width: int = 96) -> np.ndarray:
    y, x = np.indices((height, width))
    return np.stack(
        (
            (3 * x + 2 * y) % 256,
            (5 * y + x // 2) % 256,
            ((x > width // 2) * 180 + (y % 61)) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)


def test_standardize_matches_adjusted_whole_image_contract() -> None:
    image = synthetic_rgb().astype(np.float32)
    result = standardize(image)
    assert result.dtype == np.float32
    assert float(np.mean(result)) == pytest.approx(0.0, abs=1e-6)
    assert float(np.std(result)) == pytest.approx(1.0, abs=1e-6)


def test_prepare_image_preserves_source_shape_and_emits_chw() -> None:
    image = synthetic_rgb()
    prepared = prepare_image(image)
    assert prepared.source_shape == image.shape[:2]
    assert prepared.model_input.shape == (3, 512, 512)
    assert prepared.model_input.dtype == np.float32


def test_prepare_image_rejects_non_rgb_or_non_uint8() -> None:
    with pytest.raises(ValueError, match="HxWx3"):
        prepare_image(np.zeros((10, 10), dtype=np.uint8))
    with pytest.raises(ValueError, match="uint8"):
        prepare_image(np.zeros((10, 10, 3), dtype=np.float32))
