# P1 v1 reference preprocessing — explicit assumptions

Recorded 18 November 2026 (Europe/Warsaw) with the reference implementation in `src/grzybobranie_ml/preprocess.py` and the synthetic fixtures in `fixtures/p1/`. These points fill gaps where the approved contract (`schemas/model-manifest.schema.json` v1.0.1, `docs/decisions/2026-11-16-schema-v1-clarifications.md` §e, `docs/prompty.md` P3/P7) is silent or only implied. They should be confirmed by Wojciech and the React Native architect before the app treats the fixtures as frozen.

## Baseline chain

The baseline used in the committed fixtures is the prototype / schema-example chain, for both $N \in \{224, 256\}$:

1. Apply EXIF orientation (`exifOrientation: "apply"`).
2. `resize.steps`: `to_width` 448 with filter `box` and `antialias: true`, then `to_size` (the manifest `inputSize`) with the same filter.
3. `crop: "none_stretch"` (no crop; stretch to $N \times N$ if the image is not already that size).
4. `rounding: "half_up_uint8"` (`floor(x + 0.5)`, clip to $0..255$).
5. `normalization` `scale_offset` with `scale = 1/127.5`, `offset = -1`.
6. `colorOrder: "RGB"`, layout `NHWC` (schema `model.input.layout` is `NHWC`).

`decodeMinWidth` (448 in the example) describes JPEG-decoder subsampling. This reference does **not** re-decode or subsample already-rasterized PIL / NumPy inputs. The geometric equivalent is the explicit `to_width` 448 step in the manifest.

## EXIF

The schema fixes `exifOrientation` to `"apply"`: "Obrót wg EXIF przed resize." This reference uses Pillow `ImageOps.exif_transpose`. Arrays passed in already decoded have no EXIF tag, so nothing is applied.

## Resize semantics

- `to_width`: always sets width to `value` (including upsampling when the source is narrower than `value`). Height is $\max(1, \lfloor h \cdot value / w + 0.5 \rfloor)$. This differs from the outdated prototype, which skipped the 448 step when `width <= 448`.
- `to_size`: stretches to `inputSize` width $\times$ height (may change aspect ratio).
- `shorter_side`: scales so $\min(h, w)$ becomes `value`, preserving aspect with the same half-up rule.
- Filter `box` and `area` use the prototype's separable area average, **height axis first, then width**, for both downsampling and upsampling. The outdated prototype used bilinear only when *both* sides were smaller than $N$; this reference does not switch filters implicitly. If a step names `bilinear` or `bicubic`, that filter is used instead.
- `antialias: false` (or a missing flag treated as `true` for the baseline) forces nearest-neighbour when the flag is present and false.

## Crop

- `none_stretch`: if the image is not already `inputSize`, stretch with the last step's filter (box in the baseline). No pixels are cropped.
- `center`: centre-crop to `inputSize`. If a side is smaller than the target, the image is first scaled up so both sides cover the target, then cropped. This is a conservative extension; the baseline does not use `center`.
- `letterbox`: fit inside `inputSize`, pad with `letterboxFill` (default `[0, 0, 0]` if omitted).

## Colour, alpha, layout

- RGBA: the alpha channel is **discarded**, not composited onto a background. Grayscale (`L` / HxW / HxWx1) is replicated to three identical RGB channels. Both use Pillow `convert("RGB")` for file/PIL inputs.
- `colorOrder: "BGR"` swaps channels after rounding and before normalisation. `mean` / `std` are interpreted in the tensor's colour order.
- `layout` is not inside the `preprocessing` object. The function defaults to `NHWC`. `CHW` is available as an argument.

## Normalisation and rounding

- `half_up_uint8`: the float tensor is computed from the rounded uint8 values (so `float = uint8 * scale + offset` on the baseline).
- `none_float`: the float path keeps unrounded resize output; the returned uint8 array is still a half-up snapshot for inspection and is not on the float path.
- `mean_std` without `pixelScale` uses `pixelScale = 1` (mean/std applied to raw pixel values).
- `none_uint8`: the float tensor is the (possibly reordered) pixel values as `float32`, with no scale/offset.

## Fixture format

P7 asks for fixtures "zdjęcie → tensor uint8 i float" for 224 and 256, including one EXIF-rotated image and one wider than 448 px, "format zgodny z `ml/fixtures/` z prototypu". The clarifications name the shared app path `ml/fixtures/p1/parity.json`. That file did not yet exist in the app repo, so this pack defines it (it is not a fork of a second format):

- `fixtures/p1/parity.json` — cases with a relative PNG path, image SHA-256, the `preprocessing` object, layout, and expected `uint8` / `float` reports: shape, dtype, raw-bytes SHA-256, min/max/mean/std, and a regular-grid sample.
- `fixtures/p1/images/*.png` — small deterministic synthetic PNGs only (no dataset photos). The EXIF case is a PNG with an `Orientation=6` tag.
- Numeric tolerance $1 \times 10^{-5}$, same as P2. SHA-256 of the tensor bytes is the exact-parity check with the app.

The app should consume this pack as `ml/fixtures/p1/` without changing the JSON shape.
