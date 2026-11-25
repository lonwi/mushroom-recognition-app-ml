---
name: pileus-blog-visuals
description: >-
  Generate 300 DPI Pileus-styled ("Nordic Botanical Paper") technical charts and CC0/CC-BY
  species comparison graphics for blog posts and documentation without matplotlib. Use when
  creating visualizations, calibration plots, or dataset photo comparisons for Grzybobranie AI.
---

# Pileus Visuals & Attribution Guide

Use this skill when generating charts or species comparison visuals matching the **Pileus v1.0** UI system (`mushroom-recognition-app/src/ui/tokens.ts`).

## 1. Pileus Design Tokens (Nordic Botanical Paper — Light)

- **Canvas (`paper`):** `#F6F5F1` | **Card (`surface`):** `#FFFFFF` | **Raised (`surfaceRaised`):** `#EAE6DC`
- **Borders & Sharp Shadows (zero blur):** `border: #DCDAD3`, `borderStrong: #767B74`, offset shadow `+3..4 px` in `#E2DFD5`.
- **Typography (`Plus Jakarta Sans`):** Primary `ink: #1B382B`, secondary `muted: #3E5348`. Resolve TTF weights dynamically via `PILEUS_FONT_DIR` or `Path.home() / "projekty/mushroom-recognition-app/..."` with `/usr/share/fonts/truetype/dejavu/DejaVuSans*.ttf` fallback (never hard-code user home paths).
- **Semantic Accents:**
  - In-Distribution / Edible: `#1F6B2A` (`bg: #E8F5E9`, `border: #81C784`)
  - Unknown Fungi / Warning: `#B45309`
  - Non-Fungi / Danger: `#AD1414` / `#B71C1C` (`bg: #FFEBEE`, `border: #FF8A80`)
  - Deadly Badge: `bg: #2A080C`, `text: #FF4560`, `border: #FF1744`

## 2. Python 3.14 + ImageMagick MSVG Rendering Rules

- In `.venv` (Python 3.14), `matplotlib` and `seaborn` wheels may be unavailable. Compute histograms/KDE curves directly with `numpy.histogram` and `scipy.stats.gaussian_kde`.
- ImageMagick 7's built-in `MSVG` parser drops `<polyline>` strokes and ignores `stroke-dasharray` on `<line>`:
  - Draw all KDE/line curves as closed filled `<polygon>` ribbons (`_curve_ribbon_svg`).
  - Draw all dashed lines as explicit `<polygon>` dash segments (`_dashed_line_svg`).
- Draw all text using Pillow `ImageDraw` + `ImageFont.truetype(...)` onto the rasterized vector base (`3000px` wide @ `dpi=(300, 300)`).
- **Greek `θ` glyph:** `PlusJakartaSans` lacks `θ`; render `θ` inline using `DejaVuSans-Bold.ttf` to prevent missing-glyph tofu boxes.

## 3. Dataset Photo Selection & Licensing Check

- Select photos strictly from `data/splits/val.csv` or `train.csv` — assert `image_url not in test_urls` (`data/splits/test.csv`).
- Verify `license` in `manifest_768.csv` and split CSV is open (`CC0`, `CC BY`, or `CC BY-SA`; never `All Rights Reserved`, avoid `NC`/`ND` for blog reuse).
- Include observation ID, observer/author, and license both in the card footer and in the accompanying Markdown caption.
- **Always inspect every generated PNG via `read_file`** before finalizing to catch overlapping labels or font glyph issues.
