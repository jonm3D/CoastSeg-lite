"""Command-line interface for model setup and raw class inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from coastseg_lite import CLASS_MAPPING, __version__
from coastseg_lite.inference import infer_image, load_tensorflow_model
from coastseg_lite.model_store import (
    best_model_path,
    fetch_model,
    load_manifest,
    sha256_file,
    verify_model_dir,
)


def _write_array(path: Path, array: np.ndarray) -> None:
    if path.suffix.lower() != ".npy":
        raise ValueError("Inference output must use the .npy extension")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.part")
    try:
        with temporary.open("wb") as stream:
            np.save(stream, array, allow_pickle=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _run_fetch(args: argparse.Namespace) -> int:
    verified = fetch_model(args.output, force=args.force)
    print(json.dumps({"model_dir": str(args.output.resolve()), "files": verified}))
    return 0


def _run_verify(args: argparse.Namespace) -> int:
    verified = verify_model_dir(args.model_dir)
    print(json.dumps({"model_dir": str(args.model_dir.resolve()), "files": verified}))
    return 0


def _run_infer(args: argparse.Namespace) -> int:
    manifest = load_manifest()
    model_file = best_model_path(args.model_dir)
    model = load_tensorflow_model(model_file)
    mask, input_shape = infer_image(args.input, model)
    _write_array(args.output, mask)
    metadata_path = args.metadata or args.output.with_suffix(".json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema": "coastseg-lite.inference/v1",
        "software": {"name": "coastseg-lite", "version": __version__},
        "input": {
            "path": str(args.input.resolve()),
            "sha256": sha256_file(args.input),
            "shape": list(input_shape),
            "dtype": "uint8",
            "colorspace": "RGB",
        },
        "model": {
            "id": manifest["model_id"],
            "implementation": manifest["implementation"],
            "repository": manifest["source"]["repository"],
            "revision": manifest["source"]["revision"],
            "weights": model_file.name,
            "weights_sha256": sha256_file(model_file),
            "target_size": manifest["input"]["target_size"],
        },
        "preprocessing": {
            "decode": "Pillow convert RGB",
            "resize": "Pillow bilinear to 512x512",
            "standardization": "whole-image mean and adjusted standard deviation",
            "tensor_layout": "CHW",
        },
        "postprocessing": {
            "logits_resize": "TensorFlow bilinear to source shape",
            "class_assignment": "argmax",
        },
        "output": {
            "path": str(args.output.resolve()),
            "sha256": sha256_file(args.output),
            "shape": list(mask.shape),
            "dtype": str(mask.dtype),
            "classes": {str(key): value for key, value in CLASS_MAPPING.items()},
            "nodata": None,
        },
    }
    metadata_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(metadata_path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the pinned CoastSeg global RGB segmentation model"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    fetch = commands.add_parser("fetch-model", help="download the pinned model")
    fetch.add_argument("--output", required=True, type=Path)
    fetch.add_argument("--force", action="store_true")
    fetch.set_defaults(run=_run_fetch)

    verify = commands.add_parser("verify-model", help="verify pinned model files")
    verify.add_argument("--model-dir", required=True, type=Path)
    verify.set_defaults(run=_run_verify)

    infer = commands.add_parser("infer", help="classify one rendered RGB image")
    infer.add_argument("--input", required=True, type=Path)
    infer.add_argument("--output", required=True, type=Path)
    infer.add_argument("--metadata", type=Path)
    infer.add_argument("--model-dir", required=True, type=Path)
    infer.set_defaults(run=_run_infer)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.run(args))
