from argparse import Namespace
from datetime import datetime, timedelta, timezone
import json

import numpy as np

from coastseg_lite.cli import _run_slopes
from coastseg_lite.slope import (
    SlopeSettings,
    estimate_slopes,
    range_slopes,
    tide_correct,
)


def test_range_slopes_includes_upper_bound():
    np.testing.assert_allclose(range_slopes(0.05, 0.1, 0.025), [0.05, 0.075, 0.1])


def test_tide_correct_uses_coastsat_sign_convention():
    corrected = tide_correct(np.array([10.0]), np.array([1.0]), np.array([0.1]))
    np.testing.assert_allclose(corrected[0], [20.0])


def test_estimate_slopes_recovers_synthetic_tidal_slope():
    dates = [
        datetime(2020, 1, 1, tzinfo=timezone.utc) + timedelta(days=5 * i)
        for i in range(80)
    ]
    seconds = np.array([(date - dates[0]).total_seconds() for date in dates])
    tide = np.sin(2.0 * np.pi * seconds / (14.0 * 24.0 * 3600.0))
    true_slope = 0.1
    observed = 100.0 - tide / true_slope
    settings = SlopeSettings(
        slope_min=0.05,
        slope_max=0.15,
        delta_slope=0.005,
        n_days=5.0,
        frequency_cutoff=1.0 / (30.0 * 24.0 * 3600.0),
        delta_frequency=1.0 / (1000.0 * 24.0 * 3600.0),
    )
    result = estimate_slopes(dates, {"gcts-1": observed}, tide, settings)
    assert abs(float(result["gcts-1"]["slope"]) - true_slope) <= settings.delta_slope


def test_slope_command_records_executed_settings(tmp_path):
    dates = np.asarray(["2020-01-01T00:00:00+00:00"])
    intersections = tmp_path / "intersections.npz"
    tides = tmp_path / "tides.npz"
    output = tmp_path / "slopes.json"
    np.savez_compressed(
        intersections,
        dates=dates,
        names=np.asarray(["gcts-1"]),
        cross_distance=np.asarray([[100.0]]),
    )
    np.savez_compressed(tides, dates=dates, tide_level=np.asarray([0.0]))

    _run_slopes(
        Namespace(
            intersections=intersections,
            tides=tides,
            output=output,
            slope_min=0.04,
            slope_max=0.2,
            delta_slope=0.01,
            min_observations=14,
        )
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["settings"]["slope_min"] == 0.04
    assert payload["settings"]["slope_max"] == 0.2
    assert payload["settings"]["delta_slope"] == 0.01
    assert payload["settings"]["min_observations"] == 14
