"""Manifest-driven P1 reference preprocessing.

Contract sources (authority order):

- ``schemas/model-manifest.schema.json`` (v1.0.1) ``preprocessing``
- ``docs/decisions/2026-11-16-schema-v1-clarifications.md`` §e
- ``docs/prompty.md`` (P3 baseline chain, P7 parity fixtures)
- ``docs/decisions/2026-11-18-p1-preprocess-assumptions.md``

Pipeline:

1. Apply EXIF orientation when ``exifOrientation`` is ``apply``.
2. Convert to 8-bit RGB (see the assumptions note for RGBA / grayscale).
3. Apply ``resize.steps`` in order.
4. Apply ``crop`` to reach ``inputSize``.
5. Round (``half_up_uint8`` or ``none_float``).
6. Reorder channels (``RGB`` / ``BGR``).
7. Normalize.
8. Apply layout (``NHWC`` default, or ``CHW``).

The box filter is the separable area average from the outdated prototype
(``ml/preprocess.py``), height axis first, then width. That path is what
the phone is expected to match.
"""

# ruff: noqa: RUF046  # int(math.floor/ceil) kept as in the prototype

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

_REQUIRED = (
    "inputSize",
    "exifOrientation",
    "colorOrder",
    "resize",
    "crop",
    "rounding",
    "normalization",
)


def round_half_up(values: np.ndarray) -> np.ndarray:
    """Match JavaScript ``Math.round`` for finite values: halves go toward +∞.

    ``numpy.rint`` is half-to-even; ``floor(x + 0.5)`` matches ``Math.round``
    and the prototype / schema ``half_up_uint8``.
    """
    return np.floor(np.asarray(values, dtype=np.float64) + 0.5)


def _as_rgb_uint8_from_array(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        # Drop the alpha channel; do not composite on a background.
        arr = arr[:, :, :3]
    elif arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected HxW, HxWx1, HxWx3 or HxWx4, got shape {arr.shape}")
    if arr.size == 0 or arr.shape[0] < 1 or arr.shape[1] < 1:
        raise ValueError("empty image")
    if arr.dtype != np.uint8:
        arr = np.clip(round_half_up(arr), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _apply_exif(image: Image.Image) -> Image.Image:
    oriented = ImageOps.exif_transpose(image)
    if oriented is None:
        raise ValueError("EXIF transpose produced an empty image")
    return oriented


def load_rgb(
    image: Image.Image | np.ndarray | Path | str,
    *,
    apply_exif: bool = True,
) -> np.ndarray:
    """Return an oriented HxWx3 uint8 RGB array.

    EXIF is applied only for PIL images and files (arrays have no tag).
    """
    if isinstance(image, (Path, str)):
        with Image.open(image) as opened:
            working = _apply_exif(opened) if apply_exif else opened
            rgb = working.convert("RGB")
            array = np.asarray(rgb)
        return _as_rgb_uint8_from_array(array)
    if isinstance(image, Image.Image):
        working = _apply_exif(image) if apply_exif else image
        return _as_rgb_uint8_from_array(np.asarray(working.convert("RGB")))
    if isinstance(image, np.ndarray):
        return _as_rgb_uint8_from_array(image)
    raise TypeError(f"unsupported image type: {type(image)!r}")


def _resize_axis(image: np.ndarray, new_size: int, axis: int) -> np.ndarray:
    """Antialiased box / area filter along one axis (prototype algorithm)."""
    old_size = int(image.shape[axis])
    if old_size == new_size:
        return image
    if new_size < 1:
        raise ValueError("target size must be positive")
    scale = old_size / new_size
    moved = np.moveaxis(image, axis, 0)
    output = np.empty((new_size,) + moved.shape[1:], dtype=np.float64)
    for index in range(new_size):
        start = index * scale
        end = (index + 1) * scale
        low = int(math.floor(start))
        high = int(math.ceil(end - 1e-12))
        high = min(max(high, low + 1), old_size)
        accumulator = np.zeros(moved.shape[1:], dtype=np.float64)
        covered = 0.0
        for source in range(low, high):
            overlap = min(end, source + 1.0) - max(start, float(source))
            if overlap <= 0:
                continue
            accumulator += moved[source] * overlap
            covered += overlap
        output[index] = accumulator / covered
    return np.moveaxis(output, 0, axis)


def resize_box(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Separable antialiased box resize, height first, then width."""
    src = np.asarray(image, dtype=np.float64)
    if src.ndim != 3 or src.shape[2] != 3:
        raise ValueError("RGB HxWx3 array required")
    resized = _resize_axis(src, int(height), axis=0)
    return _resize_axis(resized, int(width), axis=1)


def resize_nearest(image: np.ndarray, height: int, width: int) -> np.ndarray:
    src = np.asarray(image, dtype=np.float64)
    ys = np.clip(
        np.floor(
            (np.arange(height, dtype=np.float64) + 0.5) * src.shape[0] / height
        ).astype(np.int32),
        0,
        src.shape[0] - 1,
    )
    xs = np.clip(
        np.floor(
            (np.arange(width, dtype=np.float64) + 0.5) * src.shape[1] / width
        ).astype(np.int32),
        0,
        src.shape[1] - 1,
    )
    return src[ys[:, None], xs[None, :]]


def resize_bilinear(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Half-pixel-center bilinear (prototype upsample path)."""
    pixels = np.asarray(image, dtype=np.float64)
    src_h, src_w, _ = pixels.shape
    ys = (np.arange(height, dtype=np.float64) + 0.5) * src_h / height - 0.5
    xs = (np.arange(width, dtype=np.float64) + 0.5) * src_w / width - 0.5
    ys = np.clip(ys, 0.0, src_h - 1.0)
    xs = np.clip(xs, 0.0, src_w - 1.0)
    y0 = np.floor(ys).astype(np.int32)
    x0 = np.floor(xs).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, src_h - 1)
    x1 = np.clip(x0 + 1, 0, src_w - 1)
    ty = (ys - y0).astype(np.float64)[:, None, None]
    tx = (xs - x0).astype(np.float64)[None, :, None]
    top_left = pixels[y0[:, None], x0[None, :]]
    top_right = pixels[y0[:, None], x1[None, :]]
    bottom_left = pixels[y1[:, None], x0[None, :]]
    bottom_right = pixels[y1[:, None], x1[None, :]]
    top = top_left * (1.0 - tx) + top_right * tx
    bottom = bottom_left * (1.0 - tx) + bottom_right * tx
    return top * (1.0 - ty) + bottom * ty


def resize_bicubic(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Keys cubic (a = -0.5) with half-pixel centers, clamped to the source."""
    pixels = np.asarray(image, dtype=np.float64)
    src_h, src_w, channels = pixels.shape
    ys = (np.arange(height, dtype=np.float64) + 0.5) * src_h / height - 0.5
    xs = (np.arange(width, dtype=np.float64) + 0.5) * src_w / width - 0.5
    padded = np.pad(pixels, ((1, 2), (1, 2), (0, 0)), mode="edge")
    out = np.empty((height, width, channels), dtype=np.float64)
    a = -0.5
    for y in range(height):
        sy = ys[y]
        y0 = int(math.floor(sy))
        for x in range(width):
            sx = xs[x]
            x0 = int(math.floor(sx))
            acc = np.zeros(channels, dtype=np.float64)
            for i in range(-1, 3):
                wy_t = abs(sy - (y0 + i))
                if wy_t < 1.0:
                    wy = ((a + 2.0) * wy_t**3) - ((a + 3.0) * wy_t**2) + 1.0
                elif wy_t < 2.0:
                    wy = (
                        (a * wy_t**3)
                        - (5.0 * a * wy_t**2)
                        + (8.0 * a * wy_t)
                        - (4.0 * a)
                    )
                else:
                    wy = 0.0
                iy = int(np.clip(y0 + i + 1, 0, padded.shape[0] - 1))
                for j in range(-1, 3):
                    wx_t = abs(sx - (x0 + j))
                    if wx_t < 1.0:
                        wx = ((a + 2.0) * wx_t**3) - ((a + 3.0) * wx_t**2) + 1.0
                    elif wx_t < 2.0:
                        wx = (
                            (a * wx_t**3)
                            - (5.0 * a * wx_t**2)
                            + (8.0 * a * wx_t)
                            - (4.0 * a)
                        )
                    else:
                        wx = 0.0
                    ix = int(np.clip(x0 + j + 1, 0, padded.shape[1] - 1))
                    acc += padded[iy, ix] * wy * wx
            out[y, x] = acc
    return out


def resize_hw(
    image: np.ndarray,
    height: int,
    width: int,
    *,
    filter_name: str,
    antialias: bool,
) -> np.ndarray:
    src = np.asarray(image, dtype=np.float64)
    if src.shape[0] == height and src.shape[1] == width:
        return src
    if not antialias:
        return resize_nearest(src, height, width)
    name = str(filter_name)
    if name in {"box", "area"}:
        return resize_box(src, height, width)
    if name == "bilinear":
        return resize_bilinear(src, height, width)
    if name == "bicubic":
        return resize_bicubic(src, height, width)
    if name == "nearest":
        return resize_nearest(src, height, width)
    raise ValueError(f"unsupported resize filter: {filter_name!r}")


def _input_size(preprocessing: Mapping[str, Any]) -> tuple[int, int]:
    size = preprocessing.get("inputSize")
    if not isinstance(size, Mapping) or "width" not in size or "height" not in size:
        raise ValueError("preprocessing.inputSize.width and height are required")
    width = int(size["width"])
    height = int(size["height"])
    if width < 1 or height < 1:
        raise ValueError("inputSize must be positive")
    return height, width


def _proportional_side(source: int, src_ref: int, dst_ref: int) -> int:
    return max(1, int(math.floor(source * dst_ref / src_ref + 0.5)))


def _apply_step(
    image: np.ndarray,
    step: Mapping[str, Any],
    input_h: int,
    input_w: int,
) -> np.ndarray:
    if "op" not in step or "filter" not in step:
        raise ValueError("each resize step requires op and filter")
    op = step["op"]
    filter_name = step["filter"]
    antialias = bool(step["antialias"]) if "antialias" in step else True
    src_h, src_w = int(image.shape[0]), int(image.shape[1])
    if op == "to_size":
        return resize_hw(
            image, input_h, input_w, filter_name=filter_name, antialias=antialias
        )
    if op == "to_width":
        if "value" not in step:
            raise ValueError("resize step to_width requires value")
        target_w = int(step["value"])
        target_h = _proportional_side(src_h, src_w, target_w)
        return resize_hw(
            image, target_h, target_w, filter_name=filter_name, antialias=antialias
        )
    if op == "shorter_side":
        if "value" not in step:
            raise ValueError("resize step shorter_side requires value")
        target = int(step["value"])
        short = min(src_h, src_w)
        target_h = _proportional_side(src_h, short, target)
        target_w = _proportional_side(src_w, short, target)
        return resize_hw(
            image, target_h, target_w, filter_name=filter_name, antialias=antialias
        )
    raise ValueError(f"unsupported resize op: {op!r}")


def _center_crop(
    image: np.ndarray, height: int, width: int, filter_name: str, antialias: bool
) -> np.ndarray:
    src = np.asarray(image, dtype=np.float64)
    src_h, src_w = int(src.shape[0]), int(src.shape[1])
    if src_h < height or src_w < width:
        # Conservative: scale up so both sides cover the target, then crop.
        scale = max(height / src_h, width / src_w)
        new_h = max(height, int(math.ceil(src_h * scale - 1e-12)))
        new_w = max(width, int(math.ceil(src_w * scale - 1e-12)))
        src = resize_hw(src, new_h, new_w, filter_name=filter_name, antialias=antialias)
        src_h, src_w = int(src.shape[0]), int(src.shape[1])
    top = (src_h - height) // 2
    left = (src_w - width) // 2
    return src[top : top + height, left : left + width]


def _letterbox(
    image: np.ndarray,
    height: int,
    width: int,
    fill: Sequence[int],
    filter_name: str,
    antialias: bool,
) -> np.ndarray:
    src = np.asarray(image, dtype=np.float64)
    src_h, src_w = int(src.shape[0]), int(src.shape[1])
    scale = min(height / src_h, width / src_w)
    new_h = max(1, int(math.floor(src_h * scale + 0.5)))
    new_w = max(1, int(math.floor(src_w * scale + 0.5)))
    new_h = min(new_h, height)
    new_w = min(new_w, width)
    resized = resize_hw(src, new_h, new_w, filter_name=filter_name, antialias=antialias)
    canvas = np.empty((height, width, 3), dtype=np.float64)
    fill_arr = np.asarray(list(fill), dtype=np.float64)
    if fill_arr.shape != (3,):
        raise ValueError("letterboxFill must have 3 integers in 0..255")
    canvas[:] = fill_arr
    top = (height - new_h) // 2
    left = (width - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas


def _apply_crop(
    image: np.ndarray,
    preprocessing: Mapping[str, Any],
    height: int,
    width: int,
) -> np.ndarray:
    crop = preprocessing["crop"]
    if crop == "none_stretch":
        if int(image.shape[0]) == height and int(image.shape[1]) == width:
            return np.asarray(image, dtype=np.float64)
        # Last resize filter, or box, is used for the stretch.
        steps = (preprocessing.get("resize") or {}).get("steps") or []
        last = steps[-1] if steps else {"filter": "box", "antialias": True}
        return resize_hw(
            image,
            height,
            width,
            filter_name=last.get("filter", "box"),
            antialias=bool(last["antialias"]) if "antialias" in last else True,
        )
    if crop == "center":
        steps = (preprocessing.get("resize") or {}).get("steps") or []
        last = steps[-1] if steps else {"filter": "box", "antialias": True}
        return _center_crop(
            image,
            height,
            width,
            last.get("filter", "box"),
            bool(last["antialias"]) if "antialias" in last else True,
        )
    if crop == "letterbox":
        fill = preprocessing.get("letterboxFill", [0, 0, 0])
        steps = (preprocessing.get("resize") or {}).get("steps") or []
        last = steps[-1] if steps else {"filter": "box", "antialias": True}
        return _letterbox(
            image,
            height,
            width,
            fill,
            last.get("filter", "box"),
            bool(last["antialias"]) if "antialias" in last else True,
        )
    raise ValueError(f"unsupported crop: {crop!r}")


def _round_pixels(image: np.ndarray, rounding: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(uint8, float_src)``.

    For ``half_up_uint8`` the float path starts from the rounded uint8
    values. For ``none_float`` the float path keeps the unrounded
    resize output; uint8 is still a half-up snapshot for inspection.
    """
    floats = np.asarray(image, dtype=np.float64)
    uint8 = np.clip(round_half_up(floats), 0, 255).astype(np.uint8)
    if rounding == "half_up_uint8":
        return uint8, uint8.astype(np.float64)
    if rounding == "none_float":
        return uint8, floats
    raise ValueError(f"unsupported rounding: {rounding!r}")


def _reorder_color(array: np.ndarray, color_order: str) -> np.ndarray:
    if color_order == "RGB":
        return array
    if color_order == "BGR":
        return array[..., ::-1].copy()
    raise ValueError(f"unsupported colorOrder: {color_order!r}")


def _normalize(pixels: np.ndarray, spec: Mapping[str, Any]) -> np.ndarray:
    kind = spec.get("type")
    src = np.asarray(pixels, dtype=np.float64)
    if kind == "scale_offset":
        if "scale" not in spec or "offset" not in spec:
            raise ValueError("normalization scale_offset requires scale and offset")
        out = src * float(spec["scale"]) + float(spec["offset"])
        return out.astype(np.float32)
    if kind == "mean_std":
        if "mean" not in spec or "std" not in spec:
            raise ValueError("normalization mean_std requires mean and std")
        mean = np.asarray(spec["mean"], dtype=np.float64)
        std = np.asarray(spec["std"], dtype=np.float64)
        if mean.shape != (3,) or std.shape != (3,):
            raise ValueError("mean and std must have length 3")
        if np.any(std <= 0):
            raise ValueError("std must be > 0")
        # Missing pixelScale is treated as 1.0 (mean/std applied to raw pixels).
        scale = float(spec["pixelScale"]) if "pixelScale" in spec else 1.0
        scaled = src * scale
        out = (scaled - mean.reshape(1, 1, 3)) / std.reshape(1, 1, 3)
        return out.astype(np.float32)
    if kind == "none_uint8":
        return src.astype(np.float32)
    raise ValueError(f"unsupported normalization type: {kind!r}")


def _apply_layout(array: np.ndarray, layout: str) -> np.ndarray:
    if layout == "NHWC":
        return np.ascontiguousarray(array)
    if layout == "CHW":
        return np.ascontiguousarray(np.transpose(array, (2, 0, 1)))
    raise ValueError(f"unsupported layout: {layout!r}")


def preprocess(
    image: Image.Image | np.ndarray | Path | str,
    preprocessing: Mapping[str, Any],
    *,
    layout: str = "NHWC",
) -> dict[str, np.ndarray]:
    """Run the manifest preprocessing pipeline.

    Parameters
    ----------
    image:
        A PIL image, an RGB(A)/grayscale ``ndarray``, or a filesystem path.
    preprocessing:
        The ``preprocessing`` object from the model manifest (or a fixture
        subset with the same required keys).
    layout:
        ``NHWC`` (schema ``model.input.layout``) or ``CHW``.

    Returns
    -------
    dict
        ``{"uint8": ndarray, "float": ndarray}`` without a batch dimension.
        ``uint8`` is always HxWx3 in the manifest ``colorOrder``. ``float``
        is in ``layout`` order.
    """
    for key in _REQUIRED:
        if key not in preprocessing:
            raise ValueError(f"preprocessing.{key} is required")
    if preprocessing["exifOrientation"] != "apply":
        raise ValueError("preprocessing.exifOrientation must be 'apply'")

    rgb = load_rgb(image, apply_exif=True)
    input_h, input_w = _input_size(preprocessing)
    steps = (preprocessing.get("resize") or {}).get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("preprocessing.resize.steps must be a non-empty list")

    current: np.ndarray = rgb.astype(np.float64)
    for step in steps:
        if not isinstance(step, Mapping):
            raise TypeError("each resize step must be an object")
        current = _apply_step(current, step, input_h, input_w)

    current = _apply_crop(current, preprocessing, input_h, input_w)
    if current.shape[0] != input_h or current.shape[1] != input_w:
        raise RuntimeError(
            f"crop produced {current.shape[0]}x{current.shape[1]}, expected {input_h}x{input_w}"
        )

    uint8, float_src = _round_pixels(current, str(preprocessing["rounding"]))
    uint8 = _reorder_color(uint8, str(preprocessing["colorOrder"]))
    float_src = _reorder_color(float_src, str(preprocessing["colorOrder"]))
    float_tensor = _normalize(float_src, preprocessing["normalization"])
    return {
        "uint8": np.ascontiguousarray(uint8),
        "float": _apply_layout(float_tensor, layout),
    }


__all__ = [
    "load_rgb",
    "preprocess",
    "resize_bilinear",
    "resize_box",
    "resize_hw",
    "resize_nearest",
    "round_half_up",
]
