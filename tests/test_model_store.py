from __future__ import annotations

import hashlib

from coastseg_lite.model_store import load_manifest, sha256_file


def test_bundled_manifest_pins_one_best_rgb_model() -> None:
    manifest = load_manifest()
    assert manifest["model_id"] == "global_segformer_RGB_4class_14036903"
    assert manifest["implementation"] == "BEST"
    assert manifest["input"] == {
        "image_type": "RGB",
        "dtype": "uint8",
        "channels": 3,
        "target_size": [512, 512],
    }
    assert manifest["classes"] == {
        "0": "water",
        "1": "whitewater",
        "2": "sediment",
        "3": "other",
    }
    assert len(manifest["files"]) == 4
    assert manifest["architecture"]["bundled_config_sha256"] == (
        "d6a2b6f8e9fd12f094eb023e427afe15d2b82b551296c5f57d5c38ed2a7d1bb5"
    )


def test_sha256_file(tmp_path) -> None:
    path = tmp_path / "payload"
    path.write_bytes(b"coastseg-lite")
    assert sha256_file(path) == hashlib.sha256(b"coastseg-lite").hexdigest()
