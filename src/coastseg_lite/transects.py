"""CoastSat quality-controlled shoreline/transect intersections.

Derived from ``kvos/CoastSat`` ``SDS_transects.compute_intersection_QC`` at
revision ``98103cb450f7b511bb63942fe4810b84edc33a1c``.  The only method change
is that candidate points may span the complete transect, which is required by
the approximately 2 km GCTS transects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class IntersectionSettings:
    along_dist_m: float = 25.0
    min_points: int = 3
    max_std_m: float = 15.0
    max_range_m: float = 30.0
    min_chainage_m: float = -100.0
    multiple_intersection_mode: str = "auto"
    auto_percentage: float = 0.1


def _validate_transect(name: str, coordinates: np.ndarray) -> np.ndarray:
    values = np.asarray(coordinates, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or len(values) < 2:
        raise ValueError(f"Transect {name!r} must have shape (N, 2), N >= 2")
    if np.linalg.norm(values[-1] - values[0]) == 0:
        raise ValueError(f"Transect {name!r} has zero length")
    return values


def compute_intersection_qc(
    shorelines: Sequence[np.ndarray],
    transects: Mapping[str, np.ndarray],
    settings: IntersectionSettings | None = None,
) -> dict[str, np.ndarray]:
    """Return uncorrected cross-shore distances for every transect and scene."""

    resolved = settings or IntersectionSettings()
    if resolved.multiple_intersection_mode not in {"auto", "max", "nan"}:
        raise ValueError("multiple_intersection_mode must be auto, max, or nan")
    cross_distance: dict[str, np.ndarray] = {}
    shoreline_arrays = [np.asarray(value, dtype=float) for value in shorelines]

    for name, raw_transect in transects.items():
        transect = _validate_transect(name, raw_transect)
        origin = transect[0]
        end = transect[-1]
        direction = end - origin
        length = float(np.linalg.norm(direction))
        angle = np.arctan2(direction[1], direction[0])
        rotation = np.array(
            [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
        )
        medians = np.full(len(shoreline_arrays), np.nan)
        maxima = np.full(len(shoreline_arrays), np.nan)
        minima = np.full(len(shoreline_arrays), np.nan)
        standard_deviations = np.full(len(shoreline_arrays), np.nan)
        counts = np.zeros(len(shoreline_arrays), dtype=int)
        max_origin_distance = max(length, 1000.0)

        for index, shoreline in enumerate(shoreline_arrays):
            if shoreline.ndim != 2 or shoreline.shape[1] != 2 or len(shoreline) == 0:
                continue
            offsets = shoreline - origin
            distance_to_line = np.abs(np.cross(direction, offsets) / length)
            distance_to_origin = np.linalg.norm(offsets, axis=1)
            close = (distance_to_line <= resolved.along_dist_m) & (
                distance_to_origin <= max_origin_distance
            )
            if not np.any(close):
                continue
            rotated = rotation @ offsets[close].T
            chainage = rotated[0]
            chainage[chainage < resolved.min_chainage_m] = np.nan
            valid = chainage[np.isfinite(chainage)]
            if len(valid) == 0:
                continue
            standard_deviations[index] = float(np.std(valid))
            medians[index] = float(np.median(valid))
            maxima[index] = float(np.max(valid))
            minima[index] = float(np.min(valid))
            counts[index] = len(valid)

        acceptable = (
            (standard_deviations <= resolved.max_std_m)
            & ((maxima - minima) <= resolved.max_range_m)
            & (counts >= resolved.min_points)
        )
        enough_points = counts >= resolved.min_points
        if resolved.multiple_intersection_mode == "auto":
            percentage_over = float(
                np.count_nonzero(standard_deviations > resolved.max_std_m)
            ) / max(len(shoreline_arrays), 1)
            if percentage_over > resolved.auto_percentage:
                medians[~acceptable] = maxima[~acceptable]
                medians[~enough_points] = np.nan
            else:
                medians[~acceptable] = np.nan
        elif resolved.multiple_intersection_mode == "max":
            medians[~acceptable] = maxima[~acceptable]
            medians[~enough_points] = np.nan
        else:
            medians[~acceptable] = np.nan
        cross_distance[name] = medians

    return cross_distance
