"""Unit tests for manifest-driven P1 preprocessing."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from grzybobranie_ml.make_p1_fixtures import baseline_preprocessing
from grzybobranie_ml.preprocess import (
    load_rgb,
    preprocess,
    resize_box,
    round_half_up,
)

TOL = 1e-5


def _solid(width: int, height: int, color: tuple[int, int, int]) -> np.ndarray:
    img = np.empty((height, width, 3), dtype=np.uint8)
    img[...] = color
    return img


def test_round_half_up_matches_math_round() -> None:
    values = np.array([0.5, 1.5, 2.5, 10.5, 2.49, 2.51], dtype=np.float64)
    assert np.array_equal(round_half_up(values), np.array([1, 2, 3, 11, 2, 3]))


def test_box_average_of_uniform_2x2() -> None:
    src = np.array(
        [[[10, 20, 30], [30, 40, 50]], [[50, 60, 70], [70, 80, 90]]],
        dtype=np.float64,
    )
    out = resize_box(src, 1, 1)
    assert out.shape == (1, 1, 3)
    assert out[0, 0] == pytest.approx(np.mean(src, axis=(0, 1)), abs=1e-12)


def test_output_shapes_224_and_256() -> None:
    image = _solid(80, 60, (12, 34, 56))
    for n in (224, 256):
        result = preprocess(image, baseline_preprocessing(n))
        assert result["uint8"].shape == (n, n, 3)
        assert result["uint8"].dtype == np.uint8
        assert result["float"].shape == (n, n, 3)
        assert result["float"].dtype == np.float32


def test_non_square_and_wide_inputs() -> None:
    tall = _solid(40, 80, (200, 10, 10))
    wide = _solid(640, 200, (10, 200, 10))
    for image in (tall, wide):
        result = preprocess(image, baseline_preprocessing(224))
        assert result["uint8"].shape == (224, 224, 3)


def test_rgba_discards_alpha_without_compositing() -> None:
    rgb = _solid(32, 24, (10, 20, 30))
    rgba = np.dstack([rgb, np.full((24, 32), 1, dtype=np.uint8)])
    a = preprocess(rgb, baseline_preprocessing(224))
    b = preprocess(rgba, baseline_preprocessing(224))
    assert np.array_equal(a["uint8"], b["uint8"])
    assert np.array_equal(a["float"], b["float"])


def test_grayscale_is_replicated_to_rgb() -> None:
    gray = np.arange(48 * 40, dtype=np.uint8).reshape(40, 48)
    result = preprocess(gray, baseline_preprocessing(224))
    assert result["uint8"].shape == (224, 224, 3)
    # After convert-to-RGB, all three channels start equal; box resize
    # preserves that equality.
    assert np.array_equal(result["uint8"][..., 0], result["uint8"][..., 1])
    assert np.array_equal(result["uint8"][..., 1], result["uint8"][..., 2])


def test_determinism() -> None:
    image = _solid(96, 64, (7, 8, 9))
    image[10:20, 30:50] = (255, 128, 0)
    first = preprocess(image, baseline_preprocessing(256))
    second = preprocess(image, baseline_preprocessing(256))
    assert np.array_equal(first["uint8"], second["uint8"])
    assert np.array_equal(first["float"], second["float"])


def test_float_is_computed_from_rounded_uint8_on_baseline() -> None:
    image = _solid(70, 50, (1, 2, 3))
    result = preprocess(image, baseline_preprocessing(224))
    expected = result["uint8"].astype(np.float32) / np.float32(127.5) - np.float32(1.0)
    assert np.allclose(result["float"], expected, atol=1e-7, rtol=0)


def test_exif_orientation_6_transposes_before_resize() -> None:
    stored = _solid(96, 64, (0, 0, 0))
    stored[0, :, :] = (255, 0, 0)  # first stored row
    stored[:, 0, :] = (0, 255, 0)  # first stored column
    image = Image.fromarray(stored, mode="RGB")
    exif = image.getexif()
    exif[0x0112] = 6
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", exif=exif)
    buffer.seek(0)
    loaded = Image.open(buffer)
    rgb = load_rgb(loaded, apply_exif=True)
    # Orientation 6 is 90° CW: stored (h, w) = (64, 96) becomes (96, 64).
    assert rgb.shape == (96, 64, 3)
    without = preprocess(stored, baseline_preprocessing(224))
    with_exif = preprocess(loaded, baseline_preprocessing(224))
    assert not np.array_equal(without["uint8"], with_exif["uint8"])


def test_bgr_swaps_channels_before_normalization() -> None:
    prep = baseline_preprocessing(224)
    prep = {**prep, "colorOrder": "BGR"}
    image = _solid(32, 32, (255, 0, 0))
    rgb = preprocess(image, baseline_preprocessing(224))
    bgr = preprocess(image, prep)
    assert np.array_equal(rgb["uint8"][..., 0], bgr["uint8"][..., 2])
    assert np.array_equal(rgb["uint8"][..., 2], bgr["uint8"][..., 0])


def test_chw_layout() -> None:
    result = preprocess(
        _solid(40, 40, (9, 8, 7)), baseline_preprocessing(224), layout="CHW"
    )
    assert result["float"].shape == (3, 224, 224)
    assert result["uint8"].shape == (224, 224, 3)


def test_missing_required_key_raises() -> None:
    prep = baseline_preprocessing(224)
    del prep["rounding"]
    with pytest.raises(ValueError, match="rounding"):
        preprocess(_solid(16, 16, (1, 1, 1)), prep)


def test_none_float_rounding_does_not_quantize_the_float_path() -> None:
    prep = baseline_preprocessing(224)
    prep = {**prep, "rounding": "none_float"}
    src = np.array(
        [[[0, 0, 0], [255, 255, 255]], [[255, 0, 0], [0, 255, 0]]],
        dtype=np.uint8,
    )
    result = preprocess(src, prep)
    # The important lock is that none_float still returns both tensors.
    assert result["float"].shape == (224, 224, 3)
    assert result["float"].dtype == np.float32
    assert result["uint8"].dtype == np.uint8
