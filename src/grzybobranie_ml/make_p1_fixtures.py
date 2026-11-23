"""Generate synthetic P1 preprocessing parity fixtures.

Write ``fixtures/p1/parity.json``, the PNG inputs under
``fixtures/p1/images/``, and ``fixtures/p1/SHA256SUMS``. The document
format is the shared app path ``ml/fixtures/p1/parity.json`` (see the
assumptions note). All images are synthetic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from grzybobranie_ml.preprocess import preprocess

FIXTURE_SCHEMA_VERSION = "1.0.1"
GENERATOR = "grzybobranie_ml.make_p1_fixtures"
TOLERANCE = 1e-5
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "fixtures" / "p1"
FORMAT_ID = "ml/fixtures/p1/parity.json"

# ImageNet-style mean/std used only in an extra mean_std case, not the baseline.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def baseline_preprocessing(n: int) -> dict[str, Any]:
    """Prototype / schema-example baseline for a square input of size N."""
    return {
        "inputSize": {"width": n, "height": n},
        "exifOrientation": "apply",
        "colorOrder": "RGB",
        "decodeMinWidth": 448,
        "resize": {
            "steps": [
                {
                    "op": "to_width",
                    "value": 448,
                    "filter": "box",
                    "antialias": True,
                },
                {"op": "to_size", "filter": "box", "antialias": True},
            ]
        },
        "crop": "none_stretch",
        "rounding": "half_up_uint8",
        "normalization": {
            "type": "scale_offset",
            "scale": 1.0 / 127.5,
            "offset": -1.0,
        },
    }


def _pattern(width: int, height: int, seed: int = 0) -> np.ndarray:
    ys = np.arange(height, dtype=np.int32)[:, None]
    xs = np.arange(width, dtype=np.int32)[None, :]
    r = (xs * 3 + ys * 7 + seed * 17) % 256
    g = (xs * 11 + ys * 5 + 13 + seed * 19) % 256
    b = (xs * 2 + ys * 17 + 19 + seed * 23) % 256
    # A few solid blocks so box averaging is easy to inspect by eye.
    img = np.stack([r, g, b], axis=-1).astype(np.uint8)
    img[: height // 5, : width // 5] = (255, 0, 0)
    img[-height // 6 :, -width // 6 :] = (0, 0, 255)
    mid_y, mid_x = height // 2, width // 2
    img[mid_y - 2 : mid_y + 3, mid_x - 2 : mid_x + 3] = (0, 255, 0)
    return img


def _save_png(path: Path, array: np.ndarray, *, mode: str = "RGB", exif=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(array, mode=mode)
    kwargs: dict[str, Any] = {"format": "PNG", "optimize": False, "compress_level": 9}
    if exif is not None:
        kwargs["exif"] = exif
    image.save(path, **kwargs)


def _orientation_exif(orientation: int):
    exif = Image.Exif()
    exif[0x0112] = int(orientation)
    return exif


def write_images(output_dir: Path) -> dict[str, str]:
    images = output_dir / "images"
    images.mkdir(parents=True, exist_ok=True)

    _save_png(images / "pattern_64x48.png", _pattern(64, 48, seed=1))
    _save_png(images / "wide_640x320.png", _pattern(640, 320, seed=2))

    # Stored 96x64, Orientation=6 (90° CW). After exif_transpose the
    # raster is 64x96. Pixels are written in the *stored* orientation.
    stored = _pattern(96, 64, seed=3)
    _save_png(images / "exif_orient6_96x64.png", stored, exif=_orientation_exif(6))

    rgba = np.dstack([_pattern(48, 40, seed=4), np.full((40, 48), 128, dtype=np.uint8)])
    _save_png(images / "rgba_48x40.png", rgba, mode="RGBA")

    gray = _pattern(40, 48, seed=5)[:, :, 1]  # green channel as luminance
    _save_png(images / "gray_48x40.png", gray, mode="L")

    return {p.name: _sha256(p) for p in sorted(images.glob("*.png"))}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _grid_sample(array: np.ndarray, step: int) -> dict[str, Any]:
    """Channel-last sample on a regular grid, plus the four corners and centre."""
    if array.ndim == 3 and array.shape[0] in {1, 3} and array.shape[-1] not in {1, 3}:
        # CHW → sample as HWC for a stable layout in the fixture.
        view = np.transpose(array, (1, 2, 0))
    else:
        view = array
    height, width = int(view.shape[0]), int(view.shape[1])
    ys = list(range(0, height, step))
    if (height - 1) not in ys:
        ys.append(height - 1)
    xs = list(range(0, width, step))
    if (width - 1) not in xs:
        xs.append(width - 1)
    coords = [(y, x) for y in ys for x in xs]
    # Corners and centre are already in a full grid that includes 0 and max.
    values = [float(v) for y, x in coords for v in np.asarray(view[y, x]).reshape(-1)]
    return {"method": "grid", "step": step, "coords": coords, "values": values}


def _tensor_report(array: np.ndarray, *, step: int) -> dict[str, Any]:
    flat = np.asarray(array, dtype=np.float64)
    return {
        "shape": [int(s) for s in array.shape],
        "dtype": str(array.dtype),
        "sha256": _tensor_sha256(array),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat, ddof=0)),
        "sample": _grid_sample(array, step),
    }


def _case(
    id_: str,
    description: str,
    image_name: str,
    image_sha: str,
    preprocessing: dict[str, Any],
    result: dict[str, np.ndarray],
    *,
    layout: str = "NHWC",
    step: int = 32,
) -> dict[str, Any]:
    return {
        "id": id_,
        "description": description,
        "synthetic": True,
        "image": f"images/{image_name}",
        "imageSha256": image_sha,
        "layout": layout,
        "preprocessing": preprocessing,
        "expected": {
            "uint8": _tensor_report(result["uint8"], step=step),
            "float": _tensor_report(result["float"], step=step),
        },
    }


def build_cases(output_dir: Path, image_shas: dict[str, str]) -> list[dict[str, Any]]:
    images = output_dir / "images"
    cases: list[dict[str, Any]] = []

    core = [
        (
            "pattern_64x48.png",
            "Non-square input smaller than N (steps include to_width 448, then to_size).",
        ),
        (
            "wide_640x320.png",
            "Non-square input wider than 448 px; first step is box to_width 448.",
        ),
        (
            "exif_orient6_96x64.png",
            "PNG with EXIF Orientation=6; raster is transposed before resize.",
        ),
    ]
    for n in (224, 256):
        prep = baseline_preprocessing(n)
        for name, desc in core:
            result = preprocess(images / name, prep, layout="NHWC")
            cases.append(
                _case(
                    f"baseline_{n}_{Path(name).stem}",
                    f"N={n}. {desc} Baseline: EXIF, box 448, to_size {n}, half_up_uint8, /127.5-1, RGB NHWC.",
                    name,
                    image_shas[name],
                    prep,
                    result,
                    step=32,
                )
            )

    # Channel-conversion locks (one size is enough).
    prep_224 = baseline_preprocessing(224)
    for name, desc in (
        ("rgba_48x40.png", "RGBA input; alpha is discarded, not composited."),
        ("gray_48x40.png", "Grayscale input; the single channel is replicated to RGB."),
    ):
        result = preprocess(images / name, prep_224, layout="NHWC")
        cases.append(
            _case(
                f"baseline_224_{Path(name).stem}",
                f"N=224. {desc}",
                name,
                image_shas[name],
                prep_224,
                result,
                step=32,
            )
        )

    # Extra: mean_std + pixelScale + BGR, to lock non-baseline normalisation.
    mean_std = baseline_preprocessing(224)
    mean_std = {
        **mean_std,
        "colorOrder": "BGR",
        "normalization": {
            "type": "mean_std",
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
            "pixelScale": 1.0 / 255.0,
        },
    }
    result = preprocess(images / "pattern_64x48.png", mean_std, layout="NHWC")
    cases.append(
        _case(
            "mean_std_224_bgr_pattern_64x48",
            "N=224, BGR, mean_std with pixelScale=1/255 (ImageNet-style). Not the baseline chain.",
            "pattern_64x48.png",
            image_shas["pattern_64x48.png"],
            mean_std,
            result,
            step=32,
        )
    )
    return cases


def _dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2) + "\n"


def write_fixtures(output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_shas = write_images(output_dir)
    payload = {
        "schemaVersion": FIXTURE_SCHEMA_VERSION,
        "generator": GENERATOR,
        "format": FORMAT_ID,
        "tolerance": TOLERANCE,
        "synthetic": True,
        "cases": build_cases(output_dir, image_shas),
    }
    parity_path = output_dir / "parity.json"
    parity_path.write_text(_dump(payload), encoding="utf-8")

    written: dict[str, str] = {"parity.json": _sha256(parity_path)}
    for name, digest in image_shas.items():
        written[f"images/{name}"] = digest

    sums = output_dir / "SHA256SUMS"
    # Stable order: parity.json first, then images by name.
    lines = [f"{written['parity.json']}  parity.json\n"]
    for name in sorted(image_shas):
        rel = f"images/{name}"
        lines.append(f"{written[rel]}  {rel}\n")
    sums.write_text("".join(lines), encoding="utf-8")
    written["SHA256SUMS"] = _sha256(sums)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory for parity.json, images/ and SHA256SUMS",
    )
    args = parser.parse_args(argv)
    written = write_fixtures(args.output_dir.resolve())
    for name, digest in written.items():
        print(f"{digest}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
