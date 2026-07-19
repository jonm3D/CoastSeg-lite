"""CoastSat Lomb–Scargle beach-face slope estimation.

The numerical functions are a minimal, non-plotting extraction of
``kvos/CoastSat/coastsat/SDS_slope.py`` at revision
``98103cb450f7b511bb63942fe4810b84edc33a1c``.  Upstream Simpson integration is
retained.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Sequence

import numpy as np
from astropy.timeseries import LombScargle
from scipy import integrate, interpolate, signal


@dataclass(frozen=True)
class SlopeSettings:
    slope_min: float = 0.01
    slope_max: float = 0.3
    delta_slope: float = 0.005
    n0: int = 50
    n_days: float = 8.0
    frequency_cutoff: float = 1.0 / (30.0 * 24.0 * 3600.0)
    delta_frequency: float = 1.0 / (1000.0 * 24.0 * 3600.0)
    confidence_fraction: float = 0.05
    min_observations: int = 10


def range_slopes(min_slope: float, max_slope: float, delta_slope: float) -> np.ndarray:
    """Create the CoastSat candidate slope grid."""

    if not (0 < min_slope <= max_slope and delta_slope > 0):
        raise ValueError("Slope bounds and increment must be positive")
    values: list[float] = []
    slope = min_slope
    while slope < max_slope:
        values.append(slope)
        slope += delta_slope
    values.append(slope)
    decimals = len(str(delta_slope).split(".")[-1])
    return np.round(values, decimals)


def tide_correct(
    chainage: np.ndarray, tide_level: np.ndarray, beach_slopes: np.ndarray
) -> list[np.ndarray]:
    """Apply CoastSat's tidal correction for each candidate slope."""

    chain = np.asarray(chainage, dtype=float)
    tide = np.asarray(tide_level, dtype=float)
    if chain.shape != tide.shape:
        raise ValueError("chainage and tide_level must have the same shape")
    return [chain + tide / slope for slope in beach_slopes]


def frequency_grid(time: np.ndarray, time_step: float, n0: int) -> np.ndarray:
    duration = float(np.max(time) - np.min(time))
    if duration <= 0 or time_step <= 0 or n0 <= 0:
        raise ValueError("Time range, time step, and n0 must be positive")
    minimum = 1.0 / duration
    maximum = 1.0 / (2.0 * time_step)
    spacing = 1.0 / (n0 * duration)
    count = int(np.ceil((maximum - minimum) / spacing))
    if count < 2:
        raise ValueError("Time series is too short for the requested frequency grid")
    return minimum + spacing * np.arange(count)


def power_spectrum(
    time: np.ndarray,
    values: np.ndarray,
    frequencies: np.ndarray,
    cutoff: np.ndarray | Sequence[int] | None = None,
) -> tuple[np.ndarray, float, float]:
    model = LombScargle(
        time,
        values,
        dy=None,
        fit_mean=True,
        center_data=True,
        nterms=1,
        normalization="psd",
    )
    power = model.power(frequencies)
    energy = float(integrate.simpson(power, x=frequencies))
    selected = np.asarray(cutoff) if cutoff is not None else np.array([], dtype=int)
    if selected.size == 0:
        selected = np.ones(frequencies.size, dtype=bool)
    cutoff_energy = float(integrate.simpson(power[selected], x=frequencies[selected]))
    return power, energy, cutoff_energy


def find_tide_peak(
    dates: Sequence[datetime], tide_level: np.ndarray, settings: SlopeSettings
) -> tuple[float, float]:
    time = np.asarray([value.timestamp() for value in dates], dtype=float)
    time_step = settings.n_days * 24.0 * 3600.0
    frequencies = frequency_grid(time, time_step, settings.n0)
    tide_power, _, _ = power_spectrum(time, tide_level, frequencies)
    peak_indices, properties = signal.find_peaks(tide_power, height=0)
    if len(peak_indices) == 0:
        raise ValueError("No tidal spectral peak was found")
    ranked = peak_indices[np.argsort(properties["peak_heights"])[::-1]]
    candidates = ranked[
        (frequencies[ranked] > settings.frequency_cutoff)
        & (frequencies[ranked] < frequencies[-1] - settings.delta_frequency)
    ]
    if len(candidates) == 0:
        raise ValueError("No tidal peak falls within the configured frequency band")
    maximum = frequencies[candidates[0]]
    return maximum - settings.delta_frequency, maximum + settings.delta_frequency


def integrate_power_spectrum(
    dates: Sequence[datetime],
    corrected_series: Sequence[np.ndarray],
    settings: SlopeSettings,
    frequency_band: tuple[float, float],
) -> tuple[float, tuple[float, float]]:
    time = np.asarray([value.timestamp() for value in dates], dtype=float)
    time_step = settings.n_days * 24.0 * 3600.0
    frequencies = frequency_grid(time, time_step, settings.n0)
    slopes = range_slopes(settings.slope_min, settings.slope_max, settings.delta_slope)
    interval = (frequencies >= frequency_band[0]) & (frequencies <= frequency_band[1])
    energies = np.zeros(len(slopes), dtype=float)
    for index, values in enumerate(corrected_series):
        power, _, _ = power_spectrum(time, values, frequencies)
        energies[index] = integrate.simpson(power[interval], x=frequencies[interval])
    half_step = settings.delta_slope / 2.0
    interpolation = interpolate.interp1d(slopes, energies, kind="linear")
    fine_slopes = range_slopes(settings.slope_min, settings.slope_max, half_step)
    fine_energy = interpolation(fine_slopes)
    minimum_band = fine_slopes[
        fine_energy <= np.min(energies) * (1.0 + settings.confidence_fraction)
    ]
    best = float(slopes[np.argmin(energies)])
    if len(minimum_band) > 1:
        confidence = (float(minimum_band[0]), float(minimum_band[-1]))
    else:
        confidence = (best, best)
    return best, confidence


def estimate_slopes(
    dates: Sequence[datetime],
    cross_distance: Mapping[str, np.ndarray],
    tide_level: np.ndarray,
    settings: SlopeSettings | None = None,
) -> dict[str, dict[str, float | int | list[float]]]:
    """Estimate one slope and confidence interval per transect."""

    resolved = settings or SlopeSettings()
    if len(dates) != len(tide_level):
        raise ValueError("dates and tide_level must have the same length")
    tide = np.asarray(tide_level, dtype=float)
    common_valid = np.isfinite(tide)
    if np.count_nonzero(common_valid) < resolved.min_observations:
        return {}
    common_dates = [dates[index] for index in np.where(common_valid)[0]]
    frequency_band = find_tide_peak(common_dates, tide[common_valid], resolved)
    slopes = range_slopes(resolved.slope_min, resolved.slope_max, resolved.delta_slope)
    results: dict[str, dict[str, float | int | list[float]]] = {}
    for name, raw_chainage in cross_distance.items():
        chainage = np.asarray(raw_chainage, dtype=float)
        if len(chainage) != len(dates):
            raise ValueError(f"Transect {name!r} does not align with dates")
        valid = common_valid & np.isfinite(chainage)
        if np.count_nonzero(valid) < resolved.min_observations:
            continue
        selected_dates = [dates[index] for index in np.where(valid)[0]]
        corrected = tide_correct(chainage[valid], tide[valid], slopes)
        best, confidence = integrate_power_spectrum(
            selected_dates, corrected, resolved, frequency_band
        )
        results[name] = {
            "slope": best,
            "ci": [confidence[0], confidence[1]],
            "n_observations": int(np.count_nonzero(valid)),
        }
    return results
