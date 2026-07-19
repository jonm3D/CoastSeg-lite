import numpy as np

from coastseg_lite.shoreline import (
    ShorelineSettings,
    create_shoreline_buffer,
    extract_shoreline,
    merge_classes,
    pixel_to_world,
    world_to_pixel,
)

GEOREF = (0.0, 1.0, 0.0, 10.0, 0.0, -1.0)


def test_pixel_world_round_trip():
    pixels = np.array([[0.0, 0.0], [2.5, 7.5], [9.0, 9.0]])
    np.testing.assert_allclose(
        world_to_pixel(pixel_to_world(pixels, GEOREF), GEOREF), pixels
    )


def test_merge_classes_uses_water_and_whitewater():
    labels = np.array([[0, 1, 2, 3]], dtype=np.uint8)
    np.testing.assert_array_equal(
        merge_classes(labels, (0, 1)), [[True, True, False, False]]
    )


def test_reference_buffer_is_continuous_for_sparse_line():
    reference = np.array([[5.0, 10.0], [5.0, 1.0]])
    buffer = create_shoreline_buffer((10, 10), GEOREF, reference, 1.0, 1.0)
    assert np.all(buffer[:, 5])


def test_extract_vertical_binary_shoreline():
    labels = np.full((10, 10), 2, dtype=np.uint8)
    labels[:, 5:] = 0
    clear = np.zeros_like(labels, dtype=bool)
    settings = ShorelineSettings(
        min_beach_area_pixels=1,
        max_dist_ref_m=2.0,
        min_length_sl_m=5.0,
        dist_clouds_m=0.0,
        dist_nodata_m=0.0,
    )
    shoreline = extract_shoreline(
        labels,
        clear,
        clear,
        GEOREF,
        1.0,
        np.array([[4.5, 10.0], [4.5, 1.0]]),
        settings,
    )
    assert len(shoreline) >= 8
    np.testing.assert_allclose(shoreline[:, 0], 4.5)
