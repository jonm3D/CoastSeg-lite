"""Command-line interface for inference and CoastSat-derived measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np

from coastseg_lite import CLASS_MAPPING, __version__
from coastseg_lite.shoreline import ShorelineSettings, extract_shoreline
from coastseg_lite.slope import SlopeSettings, estimate_slopes
from coastseg_lite.transects import IntersectionSettings, compute_intersection_qc
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
    from coastseg_lite.inference import infer_image, load_tensorflow_model

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


def _load_optional_mask(path: Path | None, shape: tuple[int, int]) -> np.ndarray:
    if path is None:
        return np.zeros(shape, dtype=bool)
    mask = np.load(path, allow_pickle=False)
    if mask.shape != shape:
        raise ValueError(f"Mask {path} has shape {mask.shape}, expected {shape}")
    return np.asarray(mask, dtype=bool)


def _run_shoreline(args: argparse.Namespace) -> int:
    labels = np.load(args.classes, allow_pickle=False)
    reference = np.load(args.reference, allow_pickle=False)
    settings = ShorelineSettings(
        min_beach_area_pixels=args.min_beach_area_pixels,
        max_dist_ref_m=args.max_dist_ref,
        min_length_sl_m=args.min_length,
        dist_clouds_m=args.dist_clouds,
        dist_nodata_m=args.dist_nodata,
    )
    shoreline = extract_shoreline(
        labels,
        _load_optional_mask(args.cloud_mask, labels.shape),
        _load_optional_mask(args.nodata_mask, labels.shape),
        args.georef,
        args.pixel_size,
        reference,
        settings,
    )
    _write_array(args.output, shoreline)
    print(
        json.dumps({"schema": "coastseg-lite.shoreline/v1", "points": len(shoreline)})
    )
    return 0


def _load_transects(path: Path) -> dict[str, np.ndarray]:
    archive = np.load(path, allow_pickle=False)
    names = archive["names"].astype(str)
    coordinates = archive["coordinates"]
    if coordinates.shape != (len(names), 2, 2):
        raise ValueError("Transect coordinates must have shape (N, 2, 2)")
    return {name: coordinates[index] for index, name in enumerate(names)}


def _run_intersections(args: argparse.Namespace) -> int:
    archive = np.load(args.shorelines, allow_pickle=False)
    dates = archive["dates"].astype(str)
    offsets = archive["offsets"].astype(int)
    points = archive["points"]
    if len(offsets) != len(dates) + 1 or offsets[0] != 0 or offsets[-1] != len(points):
        raise ValueError("Invalid packed shoreline offsets")
    shorelines = [points[offsets[i] : offsets[i + 1]] for i in range(len(dates))]
    transects = _load_transects(args.transects)
    settings = IntersectionSettings(
        along_dist_m=args.along_dist,
        max_std_m=args.max_std,
        max_range_m=args.max_range,
        min_points=args.min_points,
        multiple_intersection_mode=args.multiple_inter,
        auto_percentage=args.prc_multiple,
    )
    distances = compute_intersection_qc(shorelines, transects, settings)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        schema=np.array("coastseg-lite.intersections/v1"),
        dates=dates,
        names=np.asarray(list(transects)),
        cross_distance=np.column_stack([distances[name] for name in transects]),
    )
    print(args.output)
    return 0


def _run_slopes(args: argparse.Namespace) -> int:
    intersections = np.load(args.intersections, allow_pickle=False)
    tide = np.load(args.tides, allow_pickle=False)
    dates = [
        datetime.fromisoformat(value) for value in intersections["dates"].astype(str)
    ]
    tide_dates = tide["dates"].astype(str)
    if not np.array_equal(intersections["dates"].astype(str), tide_dates):
        raise ValueError("Tide and intersection dates must align exactly")
    names = intersections["names"].astype(str)
    matrix = intersections["cross_distance"]
    cross_distance = {name: matrix[:, index] for index, name in enumerate(names)}
    settings = SlopeSettings(
        slope_min=args.slope_min,
        slope_max=args.slope_max,
        delta_slope=args.delta_slope,
        min_observations=args.min_observations,
    )
    result = estimate_slopes(dates, cross_distance, tide["tide_level"], settings)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"schema": "coastseg-lite.slopes/v1", "transects": result}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(args.output)
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

    shoreline = commands.add_parser(
        "shoreline", help="extract one shoreline from class labels"
    )
    shoreline.add_argument("--classes", required=True, type=Path)
    shoreline.add_argument("--reference", required=True, type=Path)
    shoreline.add_argument("--cloud-mask", type=Path)
    shoreline.add_argument("--nodata-mask", type=Path)
    shoreline.add_argument("--georef", nargs=6, required=True, type=float)
    shoreline.add_argument("--pixel-size", required=True, type=float)
    shoreline.add_argument("--output", required=True, type=Path)
    shoreline.add_argument("--min-beach-area-pixels", type=int, default=10)
    shoreline.add_argument("--max-dist-ref", type=float, default=100.0)
    shoreline.add_argument("--min-length", type=float, default=200.0)
    shoreline.add_argument("--dist-clouds", type=float, default=300.0)
    shoreline.add_argument("--dist-nodata", type=float, default=30.0)
    shoreline.set_defaults(run=_run_shoreline)

    intersections = commands.add_parser(
        "intersections", help="intersect packed shorelines with transects"
    )
    intersections.add_argument("--shorelines", required=True, type=Path)
    intersections.add_argument("--transects", required=True, type=Path)
    intersections.add_argument("--output", required=True, type=Path)
    intersections.add_argument("--along-dist", type=float, default=25.0)
    intersections.add_argument("--max-std", type=float, default=15.0)
    intersections.add_argument("--max-range", type=float, default=30.0)
    intersections.add_argument("--min-points", type=int, default=3)
    intersections.add_argument(
        "--multiple-inter", choices=("auto", "nan", "max"), default="auto"
    )
    intersections.add_argument("--prc-multiple", type=float, default=0.1)
    intersections.set_defaults(run=_run_intersections)

    slopes = commands.add_parser("slopes", help="estimate CoastSat beach slopes")
    slopes.add_argument("--intersections", required=True, type=Path)
    slopes.add_argument("--tides", required=True, type=Path)
    slopes.add_argument("--output", required=True, type=Path)
    slopes.add_argument("--slope-min", type=float, default=0.01)
    slopes.add_argument("--slope-max", type=float, default=0.3)
    slopes.add_argument("--delta-slope", type=float, default=0.005)
    slopes.add_argument("--min-observations", type=int, default=10)
    slopes.set_defaults(run=_run_slopes)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.run(args))
