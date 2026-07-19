import numpy as np

from coastseg_lite.transects import IntersectionSettings, compute_intersection_qc

SETTINGS = IntersectionSettings(
    along_dist_m=10.0,
    min_points=3,
    max_std_m=5.0,
    max_range_m=10.0,
    min_chainage_m=-10.0,
    multiple_intersection_mode="nan",
)


def _shoreline_at(x_position: float) -> np.ndarray:
    return np.column_stack((np.full(11, x_position), np.linspace(-5.0, 5.0, 11)))


def test_qc_intersection_returns_median_chainage():
    result = compute_intersection_qc(
        [_shoreline_at(50.0)],
        {"gcts-1": np.array([[0.0, 0.0], [100.0, 0.0]])},
        SETTINGS,
    )
    np.testing.assert_allclose(result["gcts-1"], [50.0])


def test_qc_intersection_supports_two_kilometre_gcts_transect():
    result = compute_intersection_qc(
        [_shoreline_at(1500.0)],
        {"gcts-long": np.array([[0.0, 0.0], [2000.0, 0.0]])},
        SETTINGS,
    )
    np.testing.assert_allclose(result["gcts-long"], [1500.0])
