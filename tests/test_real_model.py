from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from coastseg_lite.inference import infer_array, load_tensorflow_model
from coastseg_lite.model_store import best_model_path


def synthetic_rgb(height: int = 96, width: int = 128) -> np.ndarray:
    y, x = np.indices((height, width))
    checker = ((x // 11 + y // 7) % 2) * 80
    return np.stack(
        (
            (2 * x + y + checker) % 256,
            (x + 3 * y) % 256,
            (255 - x + 2 * y + checker) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)


MODEL_DIR = os.environ.get("COASTSEG_LITE_MODEL_DIR", "")


@pytest.mark.real_model
@pytest.mark.skipif(not MODEL_DIR, reason="COASTSEG_LITE_MODEL_DIR is not set")
def test_pinned_model_regression() -> None:
    model = load_tensorflow_model(best_model_path(Path(MODEL_DIR)))
    mask = infer_array(synthetic_rgb(), model)
    assert mask.shape == (96, 128)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)) <= {0, 1, 2, 3}
    assert (
        hashlib.sha256(mask.tobytes()).hexdigest()
        == "94c5ae7569b18ec746799b8c7ddf41a50be0a6a2ec7d7730c97d06d002aa9a14"
    )
