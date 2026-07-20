"""Minimal CoastSeg shoreline extraction from semantic class labels.

This module preserves the headless shoreline path implemented by CoastSeg in
``src/coastseg/extracted_shoreline.py`` and CoastSat in
``coastsat/SDS_shoreline.py``.  It deliberately accepts arrays and an affine
georeference instead of CoastSeg sessions or CoastSat imagery metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.spatial import cKDTree
from skimage import measure, morphology

COASTSEG_REVISION = "5066f2594cff55e5b5143ac2b4d244e5a11288ff"
COASTSAT_REVISION = "98103cb450f7b511bb63942fe4810b84edc33a1c"


@dataclass(frozen=True)
class ShorelineSettings:
    """Parameters retained from CoastSeg/CoastSat shoreline extraction."""

    water_classes: tuple[int, ...] = (0, 1)
    min_beach_area_m2: float = 1000.0
    max_dist_ref_m: float = 100.0
    min_length_sl_m: float = 200.0
    dist_clouds_m: float = 300.0
    dist_nodata_m: float = 30.0


def merge_classes(labels: np.ndarray, classes_to_merge: Iterable[int]) -> np.ndarray:
    """Return a binary mask for the requested CoastSeg class IDs."""

    if labels.ndim != 2:
        raise ValueError(f"Expected a 2D class array, got {labels.shape}")
    merged = np.zeros(labels.shape, dtype=bool)
    for class_id in classes_to_merge:
        merged |= labels == int(class_id)
    return merged


def remove_small_objects_and_binarize(
    merged_labels: np.ndarray, min_size: int
) -> np.ndarray:
    """Remove connected components smaller than ``min_size`` pixels."""

    if min_size < 0:
        raise ValueError("min_size must be non-negative")
    binary = np.asarray(merged_labels, dtype=bool)
    if min_size <= 1:
        return binary.copy()
    return morphology.remove_small_objects(binary, min_size=min_size, connectivity=2)


def minimum_area_pixels(area_m2: float, georef: Sequence[float]) -> int:
    """Convert a physical area threshold to source-grid pixels as in CoastSat."""

    if not np.isfinite(area_m2) or area_m2 < 0:
        raise ValueError("Minimum beach area must be finite and non-negative")
    transform = np.asarray(georef, dtype=float)
    if transform.shape != (6,):
        raise ValueError("georef must contain six GDAL affine coefficients")
    pixel_area_m2 = abs(transform[1] * transform[5] - transform[2] * transform[4])
    if not np.isfinite(pixel_area_m2) or pixel_area_m2 <= 0:
        raise ValueError("Shoreline grid must have a positive pixel area")
    return int(np.ceil(area_m2 / pixel_area_m2))


def pixel_to_world(points: np.ndarray, georef: Sequence[float]) -> np.ndarray:
    """Convert CoastSat row/column coordinates with a GDAL geotransform."""

    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("Pixel coordinates must have shape (N, 2)")
    transform = np.asarray(georef, dtype=float)
    if transform.shape != (6,):
        raise ValueError("georef must contain six GDAL affine coefficients")
    row, col = values[:, 0], values[:, 1]
    x = transform[0] + transform[1] * col + transform[2] * row
    y = transform[3] + transform[4] * col + transform[5] * row
    return np.column_stack((x, y))


def world_to_pixel(points: np.ndarray, georef: Sequence[float]) -> np.ndarray:
    """Convert world X/Y coordinates to floating row/column coordinates."""

    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("World coordinates must have shape (N, 2)")
    transform = np.asarray(georef, dtype=float)
    matrix = np.array(
        [[transform[1], transform[2]], [transform[4], transform[5]]], dtype=float
    )
    if abs(float(np.linalg.det(matrix))) < np.finfo(float).eps:
        raise ValueError("georef affine transform is singular")
    col_row = np.linalg.solve(
        matrix, (values - np.array([transform[0], transform[3]])).T
    ).T
    return np.column_stack((col_row[:, 1], col_row[:, 0]))


def _densify_line(points: np.ndarray, spacing: float) -> np.ndarray:
    """Densify a reference line so its rasterized buffer is continuous."""

    if spacing <= 0:
        raise ValueError("spacing must be positive")
    values = np.asarray(points, dtype=float)
    if len(values) < 2:
        return values.copy()
    dense: list[np.ndarray] = []
    for start, end in zip(values[:-1], values[1:]):
        length = float(np.linalg.norm(end - start))
        count = max(1, int(np.ceil(length / spacing)))
        dense.extend(start + (end - start) * (index / count) for index in range(count))
    dense.append(values[-1])
    return np.asarray(dense, dtype=float)


def create_shoreline_buffer(
    image_shape: tuple[int, int],
    georef: Sequence[float],
    reference_shoreline: np.ndarray | None,
    max_dist_ref_m: float,
    pixel_size_m: float,
) -> np.ndarray:
    """Create CoastSat's dilated raster buffer around a reference shoreline."""

    if reference_shoreline is None:
        return np.ones(image_shape, dtype=bool)
    reference = np.asarray(reference_shoreline, dtype=float)
    if reference.ndim != 2 or reference.shape[1] != 2 or len(reference) < 2:
        raise ValueError("reference_shoreline must have shape (N, 2), N >= 2")
    dense = _densify_line(reference, spacing=max(pixel_size_m / 2.0, 0.01))
    pixels = np.rint(world_to_pixel(dense, georef)).astype(int)
    inside = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < image_shape[0])
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < image_shape[1])
    )
    shoreline_pixels = np.zeros(image_shape, dtype=bool)
    shoreline_pixels[pixels[inside, 0], pixels[inside, 1]] = True
    radius = int(np.ceil(max_dist_ref_m / pixel_size_m))
    if radius <= 0:
        return shoreline_pixels
    return morphology.binary_dilation(shoreline_pixels, morphology.disk(radius))


def process_contours(contours: Iterable[np.ndarray]) -> list[np.ndarray]:
    """Remove NaN vertices from contours, matching CoastSat's behavior."""

    processed: list[np.ndarray] = []
    for contour in contours:
        values = np.asarray(contour, dtype=float)
        values = values[~np.any(np.isnan(values), axis=1)]
        if len(values) > 1:
            processed.append(values)
    return processed


def find_binary_contours(
    water_mask: np.ndarray,
    invalid_mask: np.ndarray,
    reference_buffer: np.ndarray,
) -> list[np.ndarray]:
    """Trace the 0/1 class boundary at 0.5 inside the reference buffer."""

    if not (water_mask.shape == invalid_mask.shape == reference_buffer.shape):
        raise ValueError("water, invalid, and reference masks must share a grid")
    masked = np.asarray(water_mask, dtype=float).copy()
    masked[np.asarray(invalid_mask, dtype=bool)] = np.nan
    masked[~np.asarray(reference_buffer, dtype=bool)] = np.nan
    return process_contours(measure.find_contours(masked, 0.5))


def _line_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def _remove_near_mask(
    shoreline: np.ndarray,
    mask: np.ndarray,
    georef: Sequence[float],
    distance_m: float,
) -> np.ndarray:
    if len(shoreline) == 0 or not np.any(mask) or distance_m <= 0:
        return shoreline
    rows, cols = np.where(mask)
    mask_world = pixel_to_world(np.column_stack((rows, cols)), georef)
    distances, _ = cKDTree(mask_world).query(shoreline, k=1)
    return shoreline[distances >= distance_m]


def process_shoreline(
    contours: Iterable[np.ndarray],
    cloud_mask: np.ndarray,
    nodata_mask: np.ndarray,
    georef: Sequence[float],
    settings: ShorelineSettings,
) -> np.ndarray:
    """Convert, length-filter, and mask contours as in CoastSat."""

    retained: list[np.ndarray] = []
    for contour in contours:
        world = pixel_to_world(np.asarray(contour, dtype=float), georef)
        if _line_length(world) >= settings.min_length_sl_m:
            retained.append(world)
    if not retained:
        return np.empty((0, 2), dtype=float)
    shoreline = np.concatenate(retained, axis=0)
    shoreline = _remove_near_mask(shoreline, cloud_mask, georef, settings.dist_clouds_m)
    return _remove_near_mask(shoreline, nodata_mask, georef, settings.dist_nodata_m)


def extract_shoreline(
    labels: np.ndarray,
    cloud_mask: np.ndarray,
    nodata_mask: np.ndarray,
    georef: Sequence[float],
    pixel_size_m: float,
    reference_shoreline: np.ndarray | None,
    settings: ShorelineSettings | None = None,
) -> np.ndarray:
    """Extract one shoreline from CoastSeg hard labels on their source grid."""

    resolved = settings or ShorelineSettings()
    if not (labels.shape == cloud_mask.shape == nodata_mask.shape):
        raise ValueError("labels, cloud mask, and nodata mask must share a grid")
    water = merge_classes(labels, resolved.water_classes)
    water = remove_small_objects_and_binarize(
        water, minimum_area_pixels(resolved.min_beach_area_m2, georef)
    )
    reference_buffer = create_shoreline_buffer(
        labels.shape,
        georef,
        reference_shoreline,
        resolved.max_dist_ref_m,
        pixel_size_m,
    )
    invalid = np.asarray(cloud_mask, dtype=bool) | np.asarray(nodata_mask, dtype=bool)
    contours = find_binary_contours(water, invalid, reference_buffer)
    return process_shoreline(contours, cloud_mask, nodata_mask, georef, resolved)
