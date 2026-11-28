"""Generate Pileus-styled 300 DPI PNG charts for dev.to from P3-P6 artifacts.

Applies the Pileus v1.0 ("Nordic Botanical Paper") visual design system:
- paper: #F6F5F1, surface: #FFFFFF, ink: #1B382B, muted: #3E5348, border: #DCDAD3
- typography: Plus Jakarta Sans (with DejaVu Sans fallback)

Reads only existing artifacts on disk without modifying any data or models.
Outputs to artifacts/blog/images/:
- ood_energy_distribution.png (300 DPI)
- accuracy_progression.png (300 DPI)
- reliability_diagram.png (300 DPI)
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import gaussian_kde

from grzybobranie_ml.calibrate_ood import compute_ece_mce, compute_energy_vector

OUTPUT_DIR = Path("artifacts/blog/images")
EXTRACTED_LOGITS_PATH = Path("artifacts/calibration/extracted_logits.pt")
CALIBRATION_RESULTS_PATH = Path("artifacts/calibration/calibration_results.json")
MULTI_PHOTO_RESULTS_PATH = Path("artifacts/multi_photo/multi_photo_results.json")
SHIP_GATES_REPORT_PATH = Path("artifacts/ship_gates/ship_gates_report.json")
RUN_256_METRICS_PATH = Path(
    "artifacts/train_runs/mobilenetv4_conv_medium.e500_r256_in1k_r256_1795514748/metrics.jsonl"
)

# Verified baseline 224x224 EMA metrics from scripts/compare_runs.py evaluation on val.csv
BASELINE_224_OVERALL_TOP1 = 70.05
BASELINE_224_PHALLOIDES_RECALL = 72.82

# Flat Minimalist Editorial Palette
PALETTE = {
    "paper": "#F7F6F2",
    "surface": "#FFFFFF",
    "ink": "#1A1F1C",
    "muted": "#5C635E",
    "grid": "#E4E2DA",
    "axis": "#8F9690",
    "border": "#DCD9CF",
    "green": "#163827",
    "red": "#A61C1C",
    "amber": "#B85A17",
    "ok_bg": "#EFF5F1",
    "reject_bg": "#FDF3F3",
    "conflict_bg": "#FDF7EE",
}


def _resolve_font(weight: str) -> str:
    """Resolve Plus Jakarta Sans from sibling app repo (or env) with DejaVu fallback."""
    font_root = Path(
        os.environ.get(
            "PILEUS_FONT_DIR",
            str(
                Path.home()
                / "projekty/mushroom-recognition-app/node_modules/.pnpm/@expo-google-fonts+plus-jakarta-sans@0.4.2/node_modules/@expo-google-fonts/plus-jakarta-sans"
            ),
        )
    )
    mapping = {
        "regular": (font_root / "400Regular/PlusJakartaSans_400Regular.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        "semibold": (font_root / "600SemiBold/PlusJakartaSans_600SemiBold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        "bold": (font_root / "700Bold/PlusJakartaSans_700Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        "italic": (font_root / "600SemiBold_Italic/PlusJakartaSans_600SemiBold_Italic.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf"),
    }
    preferred, fallback = mapping[weight]
    return str(preferred) if preferred.exists() else fallback


def _font(weight: str, size_pt: float, scale: float = 3.0) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(_resolve_font(weight), max(10, round(size_pt * scale)))


def _svg_to_pil(svg_text: str, width_px: int, height_px: int) -> Image.Image:
    """Rasterize clean vector geometry SVG via ImageMagick to RGBA Pillow Image."""
    with tempfile.NamedTemporaryFile("w", suffix=".svg", encoding="utf-8", delete=True) as tmp_svg:
        tmp_svg.write(svg_text)
        tmp_svg.flush()
        with tempfile.NamedTemporaryFile("rb", suffix=".png", delete=True) as tmp_png:
            subprocess.run(
                ["convert", "-density", "300", tmp_svg.name, "-resize", f"{width_px}x{height_px}!", tmp_png.name],
                check=True,
            )
            with Image.open(tmp_png.name) as img:
                return img.convert("RGBA")


def _draw_text_centered(
    draw: ImageDraw.ImageDraw, cx: float, cy: float, text: str, font: ImageFont.FreeTypeFont, fill: str
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text((cx - tw / 2 - bbox[0], cy - th / 2 - bbox[1]), text, font=font, fill=fill)


def _draw_text_right(
    draw: ImageDraw.ImageDraw, rx: float, cy: float, text: str, font: ImageFont.FreeTypeFont, fill: str
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text((rx - tw - bbox[0], cy - th / 2 - bbox[1]), text, font=font, fill=fill)


def _dejavu_font(weight: str, size_pt: float, scale: float = 3.0) -> ImageFont.FreeTypeFont:
    ttf = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if weight == "bold" else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    return ImageFont.truetype(ttf, max(10, round(size_pt * scale)))


def _tokenize_mixed(text: str) -> list[tuple[str, bool]]:
    """Split text into runs of (substring, use_dejavu) so Unicode math/subscripts render cleanly."""
    tokens: list[tuple[str, bool]] = []
    i = 0
    buf = []
    cur_djv = False
    djv_chars = {"θ", "ᵢ", "≥", "≤", "−"}
    while i < len(text):
        # Handle combining macron pair 'z̄' as a single DejaVu unit
        if i + 1 < len(text) and text[i + 1] == "\u0304":
            ch = text[i : i + 2]
            is_djv = True
            i += 2
        else:
            ch = text[i]
            is_djv = ch in djv_chars
            i += 1
        if buf and is_djv != cur_djv:
            tokens.append(("".join(buf), cur_djv))
            buf = [ch]
            cur_djv = is_djv
        else:
            if not buf:
                cur_djv = is_djv
            buf.append(ch)
    if buf:
        tokens.append(("".join(buf), cur_djv))
    return tokens


def _draw_mixed_left(
    draw: ImageDraw.ImageDraw,
    lx: float,
    cy: float,
    text: str,
    primary_font: ImageFont.FreeTypeFont,
    djv_font: ImageFont.FreeTypeFont,
    fill: str,
) -> None:
    bbox_ref = draw.textbbox((0, 0), "Ag", font=primary_font)
    th_ref = bbox_ref[3] - bbox_ref[1]
    base_y = cy - th_ref / 2 - bbox_ref[1]
    x = lx
    for seg, is_djv in _tokenize_mixed(text):
        f = djv_font if is_djv else primary_font
        draw.text((x, base_y), seg, font=f, fill=fill)
        x += draw.textlength(seg, font=f)


def _draw_mixed_centered(
    draw: ImageDraw.ImageDraw,
    cx: float,
    cy: float,
    text: str,
    primary_font: ImageFont.FreeTypeFont,
    djv_font: ImageFont.FreeTypeFont,
    fill: str,
) -> None:
    total_w = sum(draw.textlength(seg, font=(djv_font if is_djv else primary_font)) for seg, is_djv in _tokenize_mixed(text))
    _draw_mixed_left(draw, cx - total_w / 2.0, cy, text, primary_font, djv_font, fill)


def _draw_text_left(
    draw: ImageDraw.ImageDraw, lx: float, cy: float, text: str, font: ImageFont.FreeTypeFont, fill: str
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    th = bbox[3] - bbox[1]
    draw.text((lx - bbox[0], cy - th / 2 - bbox[1]), text, font=font, fill=fill)


def _draw_rotated_y_label(canvas: Image.Image, cx: float, cy: float, text: str, font: ImageFont.FreeTypeFont, fill: str) -> None:
    tmp = Image.new("RGBA", (800, 120), (0, 0, 0, 0))
    d = ImageDraw.Draw(tmp)
    _draw_text_centered(d, 400, 60, text, font, fill)
    rot = tmp.rotate(90, expand=True)
    canvas.alpha_composite(rot, dest=(round(cx - rot.width / 2), round(cy - rot.height / 2)))


def _dashed_line_svg(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: str,
    stroke_width: float,
    dash_len: float = 8.0,
    gap_len: float = 6.0,
) -> list[str]:
    """Emit explicit filled <polygon> dashes so dashed lines render cleanly in ImageMagick."""
    dx = x2 - x1
    dy = y2 - y1
    total_len = float(np.hypot(dx, dy))
    if total_len <= 0:
        return []
    ux, uy = dx / total_len, dy / total_len
    nx, ny = -uy * (stroke_width / 2.0), ux * (stroke_width / 2.0)
    rects: list[str] = []
    pos = 0.0
    while pos < total_len:
        end = min(pos + dash_len, total_len)
        sx1, sy1 = x1 + ux * pos, y1 + uy * pos
        sx2, sy2 = x1 + ux * end, y1 + uy * end
        pts = (
            f"{sx1 + nx:.2f},{sy1 + ny:.2f} {sx2 + nx:.2f},{sy2 + ny:.2f} "
            f"{sx2 - nx:.2f},{sy2 - ny:.2f} {sx1 - nx:.2f},{sy1 - ny:.2f}"
        )
        rects.append(f'<polygon points="{pts}" fill="{color}"/>')
        pos += dash_len + gap_len
    return rects


def _curve_ribbon_svg(xs: np.ndarray, ys: np.ndarray, color: str, width: float) -> str:
    """Emit a smooth closed <polygon> ribbon so curve color and width render cleanly in ImageMagick."""
    dx = np.gradient(xs)
    dy = np.gradient(ys)
    norms = np.hypot(dx, dy)
    norms = np.where(norms == 0, 1.0, norms)
    half = width / 2.0
    nx = -dy / norms * half
    ny = dx / norms * half
    upper = [f"{xs[i] + nx[i]:.2f},{ys[i] + ny[i]:.2f}" for i in range(len(xs))]
    lower = [f"{xs[i] - nx[i]:.2f},{ys[i] - ny[i]:.2f}" for i in range(len(xs) - 1, -1, -1)]
    pts = " ".join(upper + lower)
    return f'<polygon points="{pts}" fill="{color}"/>'


def build_ood_energy_chart(out_png: Path) -> dict[str, float]:
    """Generate ood_energy_distribution.png in Pileus Nordic Botanical Paper theme."""
    data = torch.load(EXTRACTED_LOGITS_PATH, map_location="cpu", weights_only=False)
    cal = json.loads(CALIBRATION_RESULTS_PATH.read_text(encoding="utf-8"))
    sg = json.loads(SHIP_GATES_REPORT_PATH.read_text(encoding="utf-8"))

    temp = float(cal["temperature"])
    theta = float(sg["frozen_configuration"]["energy_threshold_theta"])

    e_val = compute_energy_vector(data["val_logits"].numpy(), temp)
    e_fungi = compute_energy_vector(data["ood_fungi_logits"].numpy(), temp)
    e_nonfungi = compute_energy_vector(data["ood_nonfungi_logits"].numpy(), temp)

    w, h = 1040, 600
    scale = 3.0
    px_w, px_h = round(w * scale), round(h * scale)
    ml, mr, mt, mb = 96, 56, 134, 82
    pw, ph = w - ml - mr, h - mt - mb

    x_min, x_max = -6.6, 0.6
    y_min, y_max = 0.0, 0.86

    def sx(x: float) -> float:
        return ml + (x - x_min) / (x_max - x_min) * pw

    def sy(y: float) -> float:
        return mt + ph - (y - y_min) / (y_max - y_min) * ph

    series = [
        ("Known Mushrooms (val, n=13,065)", e_val, PALETTE["green"], 0.13, 2.4),
        ("Unknown Fungi (ood_fungi, n=2,580)", e_fungi, PALETTE["amber"], 0.13, 2.4),
        ("Non-Fungi (ood_nonfungi, n=1,880)", e_nonfungi, PALETTE["red"], 0.13, 2.4),
    ]

    bins = np.linspace(x_min, x_max, 49)
    xs_kde = np.linspace(x_min, x_max, 240)
    tx = sx(theta)

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="10.4in" height="6.0in" viewBox="0 0 {w} {h}">',
        f'<rect width="100%" height="100%" fill="{PALETTE["paper"]}"/>',
    ]

    # Subtle horizontal gridlines
    for y_tick in [0.2, 0.4, 0.6, 0.8]:
        yy = sy(y_tick)
        svg.append(f'<line x1="{ml}" y1="{yy:.2f}" x2="{ml + pw}" y2="{yy:.2f}" stroke="{PALETTE["grid"]}" stroke-width="0.7"/>')

    # Histograms & KDE curves directly on #F7F6F2
    for _, arr, color, fill_op, stroke_w in series:
        hist_vals, edges = np.histogram(arr, bins=bins, density=True)
        for idx, val in enumerate(hist_vals):
            if val <= 0:
                continue
            bx = sx(float(edges[idx]))
            bw = sx(float(edges[idx + 1])) - bx
            by = sy(min(float(val), y_max))
            bh = sy(0.0) - by
            svg.append(
                f'<rect x="{bx:.2f}" y="{by:.2f}" width="{max(0.5, bw - 0.5):.2f}" height="{bh:.2f}" fill="{color}" opacity="{fill_op}"/>'
            )

        kde = gaussian_kde(arr)
        ys_kde = np.clip(kde(xs_kde), 0.0, y_max)
        px_xs = np.array([sx(float(x)) for x in xs_kde], dtype=np.float64)
        px_ys = np.array([sy(float(y)) for y in ys_kde], dtype=np.float64)
        svg.append(_curve_ribbon_svg(px_xs, px_ys, color, stroke_w))

    # Crisp bottom & left spines only (no top/right spines)
    svg.append(f'<line x1="{ml}" y1="{mt + ph}" x2="{ml + pw}" y2="{mt + ph}" stroke="{PALETTE["axis"]}" stroke-width="1.1"/>')
    svg.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + ph}" stroke="{PALETTE["axis"]}" stroke-width="1.1"/>')
    for x_tick in [-6, -5, -4, -3, -2, -1, 0]:
        xx = sx(float(x_tick))
        svg.append(f'<line x1="{xx:.2f}" y1="{mt + ph}" x2="{xx:.2f}" y2="{mt + ph + 4}" stroke="{PALETTE["axis"]}" stroke-width="1.1"/>')

    # Vertical dashed threshold
    svg.extend(_dashed_line_svg(tx, mt + 24, tx, mt + ph, PALETTE["ink"], 1.5, dash_len=7.0, gap_len=5.0))

    # Flat legend swatches directly on #F7F6F2 (no card, no shadow)
    leg_x, leg_y = ml + 12, mt + 14
    for idx, (_, _, color, _, _) in enumerate(series):
        ly = leg_y + idx * 24
        svg.append(f'<rect x="{leg_x}" y="{ly - 4}" width="18" height="3.5" rx="1.5" fill="{color}"/>')

    svg.append("</svg>")
    canvas = _svg_to_pil("\n".join(svg), px_w, px_h)
    draw = ImageDraw.Draw(canvas)

    # Typography
    _draw_text_left(draw, ml * scale, 38 * scale, "Out-of-Distribution (OOD) Energy Score Distribution", _font("bold", 20, scale), PALETTE["ink"])
    _draw_text_left(draw, ml * scale, 64 * scale, f"Energy score E(x) = −T · logsumexp(z / T) across validation & OOD splits (T = {temp:.4f})", _font("regular", 12.5, scale), PALETTE["muted"])

    # Y ticks
    for y_tick in [0.0, 0.2, 0.4, 0.6, 0.8]:
        _draw_text_right(draw, (ml - 12) * scale, sy(y_tick) * scale, f"{y_tick:.1f}", _font("regular", 11, scale), PALETTE["muted"])

    # X ticks
    for x_tick in [-6, -5, -4, -3, -2, -1, 0]:
        _draw_text_centered(draw, sx(float(x_tick)) * scale, (mt + ph + 18) * scale, str(x_tick), _font("regular", 11, scale), PALETTE["muted"])

    # Clean bold text directly above vertical dashed line (no pill box)
    _draw_mixed_centered(
        draw,
        tx * scale,
        (mt + 10) * scale,
        f"Threshold θ = {theta:.4f}",
        _font("bold", 12, scale),
        _dejavu_font("bold", 12, scale),
        PALETTE["ink"],
    )
    _draw_text_right(draw, (tx - 14) * scale, (mt + 38) * scale, "← Accepted (In-Distribution)", _font("semibold", 11.5, scale), PALETTE["green"])
    _draw_text_left(draw, (tx + 14) * scale, (mt + 38) * scale, "Rejected (OOD) →", _font("semibold", 11.5, scale), PALETTE["red"])

    # Flat legend text
    for idx, (label, _, _, _, _) in enumerate(series):
        ly = leg_y + idx * 24 - 2
        _draw_text_left(draw, (leg_x + 26) * scale, ly * scale, label, _font("regular", 11.5, scale), PALETTE["ink"])

    # Axis labels
    _draw_text_centered(draw, (ml + pw / 2) * scale, (h - 22) * scale, "Energy Score E(x)  (Lower = More In-Distribution Confidence)", _font("semibold", 12, scale), PALETTE["ink"])
    _draw_rotated_y_label(canvas, 30 * scale, (mt + ph / 2) * scale, "Probability Density", _font("semibold", 12, scale), PALETTE["ink"])

    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(out_png, dpi=(300, 300))

    return {
        "n_val": len(e_val),
        "n_ood_fungi": len(e_fungi),
        "n_ood_nonfungi": len(e_nonfungi),
        "theta": theta,
        "val_energy_keep_rate": float((e_val <= theta).mean()),
    }


def build_accuracy_progression_chart(out_png: Path) -> dict[str, dict[str, float]]:
    """Generate accuracy_progression.png in Pileus Nordic Botanical Paper theme."""
    run256_records = [
        json.loads(line) for line in RUN_256_METRICS_PATH.read_text(encoding="utf-8").splitlines() if line
    ]
    best256 = max(run256_records, key=lambda r: float(r["macro_top1"]))
    acc_256_single = float(best256["overall_top1"]) * 100.0
    rec_256_single = float(best256["per_class"]["amanita_phalloides"]["recall"]) * 100.0

    mp = json.loads(MULTI_PHOTO_RESULTS_PATH.read_text(encoding="utf-8"))
    acc_p5_val = float(mp["multi_photo_subset"]["accuracy"]["fused_raw"]) * 100.0
    rec_p5_val = (
        float(mp["high_stakes_recall"]["amanita_phalloides"]["all_observations_pipeline"]["conditional_recall"]) * 100.0
    )

    sg = json.loads(SHIP_GATES_REPORT_PATH.read_text(encoding="utf-8"))
    acc_p5_test = float(sg["multi_photo_audit"]["conditional_accuracy"]) * 100.0
    rec_test_gated = (
        float(sg["single_frame_audit"]["high_stakes_recall"]["species"]["amanita_phalloides"]["conditional_recall"])
        * 100.0
    )

    stages = [
        {
            "title_line1": "Small 224×224",
            "title_line2": "(Baseline)",
            "acc": BASELINE_224_OVERALL_TOP1,
            "acc_sub": "",
            "rec": BASELINE_224_PHALLOIDES_RECALL,
            "rec_sub": "",
        },
        {
            "title_line1": "Medium 256×256",
            "title_line2": "(Single-Frame)",
            "acc": acc_256_single,
            "acc_sub": "",
            "rec": rec_256_single,
            "rec_sub": "",
        },
        {
            "title_line1": "Medium 256×256",
            "title_line2": "+ Multi-Photo Fusion",
            "acc": acc_p5_val,
            "acc_sub": f"({acc_p5_test:.2f}% test)",
            "rec": rec_p5_val,
            "rec_sub": f"({rec_test_gated:.1f}% test gated)",
        },
    ]

    w, h = 1040, 600
    scale = 3.0
    px_w, px_h = round(w * scale), round(h * scale)
    ml, mr, mt, mb = 96, 56, 134, 92
    pw, ph = w - ml - mr, h - mt - mb

    y_min, y_max = 0.0, 108.0

    def sy(y: float) -> float:
        return mt + ph - (y - y_min) / (y_max - y_min) * ph

    color_acc = PALETTE["green"]
    color_rec = PALETTE["red"]

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="10.4in" height="6.0in" viewBox="0 0 {w} {h}">',
        f'<rect width="100%" height="100%" fill="{PALETTE["paper"]}"/>',
    ]

    # Subtle horizontal gridlines directly on #F7F6F2
    for y_tick in [20, 40, 60, 80, 100]:
        yy = sy(float(y_tick))
        svg.append(f'<line x1="{ml}" y1="{yy:.2f}" x2="{ml + pw}" y2="{yy:.2f}" stroke="{PALETTE["grid"]}" stroke-width="0.7"/>')

    # Dashed 85% high-stakes safety target
    y_target = sy(85.0)
    svg.extend(_dashed_line_svg(ml, y_target, ml + pw, y_target, PALETTE["axis"], 1.3, dash_len=6.0, gap_len=5.0))

    n_groups = len(stages)
    group_width = pw / n_groups
    bar_w = 52
    gap = 26

    for i, st in enumerate(stages):
        cx = ml + (i + 0.5) * group_width
        x1 = cx - gap / 2 - bar_w
        x2 = cx + gap / 2

        y1 = sy(st["acc"])
        h1 = sy(0.0) - y1
        svg.append(f'<rect x="{x1:.2f}" y="{y1:.2f}" width="{bar_w}" height="{h1:.2f}" rx="3" fill="{color_acc}"/>')
        if not st["acc_sub"]:
            # Clean #F7F6F2 backdrop behind 78.85% so 85% dashed line breaks cleanly
            svg.append(f'<rect x="{x1 + bar_w / 2 - 30:.2f}" y="{y1 - 24:.2f}" width="60" height="17" fill="{PALETTE["paper"]}"/>')

        y2 = sy(st["rec"])
        h2 = sy(0.0) - y2
        svg.append(f'<rect x="{x2:.2f}" y="{y2:.2f}" width="{bar_w}" height="{h2:.2f}" rx="3" fill="{color_rec}"/>')

    # Crisp bottom & left spines only
    svg.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + ph}" stroke="{PALETTE["axis"]}" stroke-width="1.1"/>')
    svg.append(f'<line x1="{ml}" y1="{mt + ph}" x2="{ml + pw}" y2="{mt + ph}" stroke="{PALETTE["axis"]}" stroke-width="1.1"/>')

    # Flat legend swatches
    leg_x, leg_y = ml, 88
    svg.append(f'<rect x="{leg_x}" y="{leg_y}" width="13" height="13" rx="2" fill="{color_acc}"/>')
    svg.append(f'<rect x="{leg_x + 178}" y="{leg_y}" width="13" height="13" rx="2" fill="{color_rec}"/>')

    svg.append("</svg>")
    canvas = _svg_to_pil("\n".join(svg), px_w, px_h)
    draw = ImageDraw.Draw(canvas)

    # Typography
    _draw_text_left(draw, ml * scale, 38 * scale, "Classification Accuracy & Death Cap Recall Progression", _font("bold", 20, scale), PALETTE["ink"])
    _draw_text_left(draw, ml * scale, 64 * scale, "Impact of input resolution (224 → 256 px), model capacity, and Multi-Photo Logit Fusion", _font("regular", 12.5, scale), PALETTE["muted"])

    _draw_text_left(draw, (leg_x + 20) * scale, (leg_y + 6.5) * scale, "Overall Accuracy", _font("semibold", 12, scale), PALETTE["ink"])
    _draw_text_left(draw, (leg_x + 198) * scale, (leg_y + 6.5) * scale, "Death Cap (Amanita phalloides) Recall", _font("semibold", 12, scale), PALETTE["ink"])

    for y_tick in [0, 20, 40, 60, 80, 100]:
        _draw_text_right(draw, (ml - 12) * scale, sy(float(y_tick)) * scale, f"{y_tick}%", _font("regular", 11, scale), PALETTE["muted"])

    _draw_text_left(draw, (ml + 12) * scale, (y_target - 12) * scale, "High-Stakes Safety Target (85%)", _font("semibold", 10.5, scale), PALETTE["muted"])

    for i, st in enumerate(stages):
        cx = ml + (i + 0.5) * group_width
        x1 = cx - gap / 2 - bar_w
        x2 = cx + gap / 2

        y1 = sy(st["acc"])
        if st["acc_sub"]:
            _draw_text_centered(draw, (x1 + bar_w / 2) * scale, (y1 - 26) * scale, f"{st['acc']:.2f}%", _font("bold", 12.5, scale), PALETTE["green"])
            _draw_text_centered(draw, (x1 + bar_w / 2) * scale, (y1 - 11) * scale, st["acc_sub"], _font("regular", 9.5, scale), PALETTE["muted"])
        else:
            _draw_text_centered(draw, (x1 + bar_w / 2) * scale, (y1 - 14) * scale, f"{st['acc']:.2f}%", _font("bold", 12.5, scale), PALETTE["green"])

        y2 = sy(st["rec"])
        if st["rec_sub"]:
            _draw_text_centered(draw, (x2 + bar_w / 2) * scale, (y2 - 26) * scale, f"{st['rec']:.1f}%", _font("bold", 12.5, scale), PALETTE["red"])
            _draw_text_centered(draw, (x2 + bar_w / 2) * scale, (y2 - 11) * scale, st["rec_sub"], _font("regular", 9.5, scale), PALETTE["muted"])
        else:
            _draw_text_centered(draw, (x2 + bar_w / 2) * scale, (y2 - 14) * scale, f"{st['rec']:.1f}%", _font("bold", 12.5, scale), PALETTE["red"])

        _draw_text_centered(draw, cx * scale, (sy(0.0) + 22) * scale, st["title_line1"], _font("bold", 12, scale), PALETTE["ink"])
        _draw_text_centered(draw, cx * scale, (sy(0.0) + 41) * scale, st["title_line2"], _font("regular", 11.5, scale), PALETTE["muted"])

    _draw_rotated_y_label(canvas, 30 * scale, (mt + ph / 2) * scale, "Score (%)", _font("semibold", 12, scale), PALETTE["ink"])

    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(out_png, dpi=(300, 300))

    return {
        "stage_1_224": {"acc": BASELINE_224_OVERALL_TOP1, "death_cap_recall": BASELINE_224_PHALLOIDES_RECALL},
        "stage_2_256": {"acc": acc_256_single, "death_cap_recall": rec_256_single},
        "stage_3_p5": {
            "acc_val": acc_p5_val,
            "acc_test": acc_p5_test,
            "death_cap_recall_val": rec_p5_val,
            "death_cap_recall_test_gated": rec_test_gated,
        },
    }


def build_reliability_diagram_chart(out_png: Path) -> dict[str, float]:
    """Generate reliability_diagram.png in Pileus Nordic Botanical Paper theme."""
    data = torch.load(EXTRACTED_LOGITS_PATH, map_location="cpu", weights_only=False)
    cal = json.loads(CALIBRATION_RESULTS_PATH.read_text(encoding="utf-8"))
    temp = float(cal["temperature"])

    logits = data["val_logits"].numpy()
    targets = np.asarray(data["val_targets"], dtype=np.int64)
    ece_raw, mce_raw, bins_raw = compute_ece_mce(logits, targets, temperature=1.0, n_bins=15)
    ece_cal, mce_cal, bins_cal = compute_ece_mce(logits, targets, temperature=temp, n_bins=15)

    w, h = 1040, 590
    scale = 3.0
    px_w, px_h = round(w * scale), round(h * scale)

    ml, mr, mt, mb = 88, 52, 126, 102
    gutter = 64
    pw = (w - ml - mr - gutter) / 2.0
    ph = h - mt - mb

    panels = [
        {
            "x0": ml,
            "title": "Uncalibrated (T = 1.0000)",
            "badge": f"ECE: {ece_raw * 100:.2f}%  ·  MCE: {mce_raw * 100:.2f}%",
            "badge_text": PALETTE["amber"],
            "bar_color": "#7A827C",
            "bins": bins_raw,
        },
        {
            "x0": ml + pw + gutter,
            "title": f"Calibrated (T = {temp:.4f})",
            "badge": f"ECE: {ece_cal * 100:.2f}%  ·  MCE: {mce_cal * 100:.2f}%",
            "badge_text": PALETTE["green"],
            "bar_color": PALETTE["green"],
            "bins": bins_cal,
        },
    ]

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="10.4in" height="5.9in" viewBox="0 0 {w} {h}">',
        f'<rect width="100%" height="100%" fill="{PALETTE["paper"]}"/>',
    ]

    for p in panels:
        x0 = float(p["x0"])
        y0 = float(mt)

        # Subtle gridlines directly on #F7F6F2
        for t_val in [0.2, 0.4, 0.6, 0.8]:
            yy = y0 + ph * (1.0 - t_val)
            xx = x0 + pw * t_val
            svg.append(f'<line x1="{x0:.2f}" y1="{yy:.2f}" x2="{x0 + pw:.2f}" y2="{yy:.2f}" stroke="{PALETTE["grid"]}" stroke-width="0.7"/>')
            svg.append(f'<line x1="{xx:.2f}" y1="{y0:.2f}" x2="{xx:.2f}" y2="{y0 + ph:.2f}" stroke="{PALETTE["grid"]}" stroke-width="0.7"/>')

        # Diagonal perfect calibration dashed line
        svg.extend(_dashed_line_svg(x0, y0 + ph, x0 + pw, y0, PALETTE["axis"], 1.3, dash_len=6.0, gap_len=5.0))

        # 15 bins: slightly slimmer bars with wider gap, alpha 0.85, subtler 2.0 px red confidence tick
        for b in p["bins"]:
            bx0 = x0 + float(b["low"]) * pw + 2.4
            bx1 = x0 + float(b["high"]) * pw - 2.4
            bw = max(2.0, bx1 - bx0)
            if b["count"] > 0:
                acc = float(b["acc"])
                conf = float(b["conf"])
                by = y0 + ph * (1.0 - acc)
                bh = (y0 + ph) - by
                svg.append(
                    f'<rect x="{bx0:.2f}" y="{by:.2f}" width="{bw:.2f}" height="{bh:.2f}" rx="2" '
                    f'fill="{p["bar_color"]}" opacity="0.85"/>'
                )
                cy_conf = y0 + ph * (1.0 - conf)
                svg.append(
                    f'<rect x="{bx0 - 0.5:.2f}" y="{cy_conf - 1.0:.2f}" width="{bw + 1.0:.2f}" height="2.0" rx="1" fill="{PALETTE["red"]}"/>'
                )

        # Left & bottom spines only
        svg.append(f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x0:.2f}" y2="{y0 + ph:.2f}" stroke="{PALETTE["axis"]}" stroke-width="1.1"/>')
        svg.append(f'<line x1="{x0:.2f}" y1="{y0 + ph:.2f}" x2="{x0 + pw:.2f}" y2="{y0 + ph:.2f}" stroke="{PALETTE["axis"]}" stroke-width="1.1"/>')

    # Bottom flat legend swatches
    leg_y = h - 22
    svg.append(f'<rect x="{ml + 178}" y="{leg_y - 5}" width="14" height="10" rx="2" fill="{PALETTE["green"]}" opacity="0.85"/>')
    svg.append(f'<rect x="{ml + 394}" y="{leg_y - 1}" width="18" height="2" rx="1" fill="{PALETTE["red"]}"/>')

    svg.append("</svg>")
    canvas = _svg_to_pil("\n".join(svg), px_w, px_h)
    draw = ImageDraw.Draw(canvas)

    # Main Title & Subtitle
    _draw_text_left(draw, ml * scale, 34 * scale, "Reliability Diagram (15 Bins) — Validation Set", _font("bold", 20, scale), PALETTE["ink"])
    _draw_text_left(draw, ml * scale, 58 * scale, "Temperature scaling removes label-smoothing underconfidence (NLL 0.9267 → 0.8527, ECE 3.55× lower)", _font("regular", 12.5, scale), PALETTE["muted"])

    for idx, p in enumerate(panels):
        x0 = float(p["x0"])
        y0 = float(mt)
        _draw_text_left(draw, x0 * scale, (y0 - 20) * scale, str(p["title"]), _font("bold", 13.5, scale), PALETTE["ink"])
        # Lightweight text label instead of colored box
        _draw_text_left(draw, (x0 + 10) * scale, (y0 + 14) * scale, str(p["badge"]), _font("semibold", 11, scale), str(p["badge_text"]))

        for t_val in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            _draw_text_centered(draw, (x0 + pw * t_val) * scale, (y0 + ph + 16) * scale, f"{t_val:.1f}", _font("regular", 10.5, scale), PALETTE["muted"])
            if idx == 0:
                _draw_text_right(draw, (x0 - 10) * scale, (y0 + ph * (1.0 - t_val)) * scale, f"{t_val:.1f}", _font("regular", 10.5, scale), PALETTE["muted"])

        _draw_text_centered(draw, (x0 + pw / 2) * scale, (y0 + ph + 38) * scale, "Predicted Confidence", _font("semibold", 11.5, scale), PALETTE["ink"])

    _draw_rotated_y_label(canvas, 28 * scale, (mt + ph / 2) * scale, "Empirical Accuracy", _font("semibold", 12, scale), PALETTE["ink"])
    _draw_text_left(draw, (ml + 200) * scale, leg_y * scale, "Bin Accuracy Bar", _font("regular", 11.5, scale), PALETTE["ink"])
    _draw_text_left(draw, (ml + 420) * scale, leg_y * scale, "Mean Bin Confidence (Gap = Calibration Error)", _font("regular", 11.5, scale), PALETTE["ink"])

    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(out_png, dpi=(300, 300))

    return {"ece_raw": ece_raw, "ece_cal": ece_cal, "temp": temp}


def build_fusion_pipeline_chart(out_png: Path) -> dict[str, str]:
    """Generate flat architectural fusion_pipeline.png on #F7F6F2 (~1640 px wide, 300 DPI)."""
    w, h = 820, 740
    scale = 2.0
    px_w, px_h = round(w * scale), round(h * scale)

    left_x, left_w = 48, 396
    right_x, right_w = 544, 232
    cx_left = left_x + left_w / 2.0

    boxes = [
        {"id": "in", "y": 106, "h": 58, "bg": PALETTE["paper"], "border": PALETTE["green"]},
        {"id": "s1", "y": 200, "h": 74, "bg": PALETTE["surface"], "border": PALETTE["green"]},
        {"id": "s2", "y": 312, "h": 74, "bg": PALETTE["surface"], "border": PALETTE["green"]},
        {"id": "s3", "y": 424, "h": 88, "bg": PALETTE["surface"], "border": PALETTE["green"]},
        {"id": "s4", "y": 550, "h": 74, "bg": PALETTE["surface"], "border": PALETTE["green"]},
        {"id": "ok", "y": 656, "h": 54, "bg": PALETTE["ok_bg"], "border": PALETTE["green"]},
    ]
    exits = [
        {"id": "e1", "y": 205, "h": 64, "bg": PALETTE["reject_bg"], "border": PALETTE["red"]},
        {"id": "e2", "y": 317, "h": 64, "bg": PALETTE["conflict_bg"], "border": PALETTE["amber"]},
        {"id": "e3", "y": 436, "h": 64, "bg": PALETTE["conflict_bg"], "border": PALETTE["amber"]},
        {"id": "e4", "y": 555, "h": 64, "bg": PALETTE["reject_bg"], "border": PALETTE["red"]},
    ]

    def _arrow_down(svg_list: list[str], x: float, y1: float, y2: float, color: str) -> None:
        svg_list.append(f'<line x1="{x:.1f}" y1="{y1:.1f}" x2="{x:.1f}" y2="{y2 - 6:.1f}" stroke="{color}" stroke-width="1.2"/>')
        svg_list.append(f'<polygon points="{x:.1f},{y2:.1f} {x - 4.5:.1f},{y2 - 7:.1f} {x + 4.5:.1f},{y2 - 7:.1f}" fill="{color}"/>')

    def _arrow_right(svg_list: list[str], x1: float, x2: float, y: float, color: str) -> None:
        svg_list.append(f'<line x1="{x1:.1f}" y1="{y:.1f}" x2="{x2 - 6:.1f}" y2="{y:.1f}" stroke="{color}" stroke-width="1.2"/>')
        svg_list.append(f'<polygon points="{x2:.1f},{y:.1f} {x2 - 7:.1f},{y - 4.5:.1f} {x2 - 7:.1f},{y + 4.5:.1f}" fill="{color}"/>')

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="8.2in" height="7.4in" viewBox="0 0 {w} {h}">',
        f'<rect width="100%" height="100%" fill="{PALETTE["paper"]}"/>',
    ]

    for idx in range(len(boxes) - 1):
        b_top, b_bot = boxes[idx], boxes[idx + 1]
        _arrow_down(svg, cx_left, b_top["y"] + b_top["h"], b_bot["y"], PALETTE["green"])

    for ex in exits:
        y_mid = ex["y"] + ex["h"] / 2.0
        _arrow_right(svg, left_x + left_w, right_x, y_mid, str(ex["border"]))

    svg.append("</svg>")
    canvas = _svg_to_pil("\n".join(svg), px_w, px_h)
    draw = ImageDraw.Draw(canvas)

    # Draw crisp flat architectural rectangles with 1.2 px borders via Pillow
    stroke_px = max(2, round(1.2 * scale))
    radius_px = round(6 * scale)
    for b in boxes:
        draw.rounded_rectangle(
            (left_x * scale, b["y"] * scale, (left_x + left_w) * scale, (b["y"] + b["h"]) * scale),
            radius=radius_px,
            fill=str(b["bg"]),
            outline=str(b["border"]),
            width=stroke_px,
        )
    for ex in exits:
        draw.rounded_rectangle(
            (right_x * scale, ex["y"] * scale, (right_x + right_w) * scale, (ex["y"] + ex["h"]) * scale),
            radius=radius_px,
            fill=str(ex["bg"]),
            outline=str(ex["border"]),
            width=stroke_px,
        )

    # Header
    _draw_text_left(draw, left_x * scale, 34 * scale, "4-Step Fail-Closed Multi-Photo Logit Fusion", _font("bold", 20, scale), PALETTE["ink"])
    _draw_text_left(draw, left_x * scale, 60 * scale, "Sequential OOD filtering, conflict detection, logit re-calibration, and High-Stakes Veto", _font("regular", 12, scale), PALETTE["muted"])

    f_title_card = _font("bold", 12.5, scale)
    f_sub_card = _font("regular", 11, scale)
    f_djv_sub = _dejavu_font("regular", 11, scale)

    # Main Spine Texts with clean Unicode
    _draw_mixed_left(draw, (left_x + 16) * scale, (boxes[0]["y"] + 19) * scale, "Input: 2–3 Specimen Photos (Cap, Gills, Stem)", f_title_card, _dejavu_font("bold", 12.5, scale), PALETTE["ink"])
    _draw_mixed_left(draw, (left_x + 16) * scale, (boxes[0]["y"] + 40) * scale, "Raw frame logits zᵢ & softmax probabilities per angle", f_sub_card, f_djv_sub, PALETTE["muted"])

    _draw_text_left(draw, (left_x + 16) * scale, (boxes[1]["y"] + 21) * scale, "1. Frame-Level OOD Filter", f_title_card, PALETTE["ink"])
    _draw_mixed_left(draw, (left_x + 16) * scale, (boxes[1]["y"] + 44) * scale, "Reject frames where E(xᵢ) > θ (θ = -2.2082, T_single = 0.7848)", f_sub_card, f_djv_sub, PALETTE["muted"])

    _draw_text_left(draw, (left_x + 16) * scale, (boxes[2]["y"] + 21) * scale, "2. Disagreement Check", f_title_card, PALETTE["ink"])
    _draw_mixed_left(draw, (left_x + 16) * scale, (boxes[2]["y"] + 44) * scale, "≥ 2 accepted frames with p_top1 ≥ 0.70 predict different species?", f_sub_card, f_djv_sub, PALETTE["muted"])

    _draw_text_left(draw, (left_x + 16) * scale, (boxes[3]["y"] + 20) * scale, "3. Logit Averaging & Re-Calibration", f_title_card, PALETTE["ink"])
    _draw_mixed_left(draw, (left_x + 16) * scale, (boxes[3]["y"] + 43) * scale, "z̄ = mean(z_accepted), re-scaled with T_multi = 0.5146", f_sub_card, f_djv_sub, PALETTE["muted"])
    _draw_mixed_left(draw, (left_x + 16) * scale, (boxes[3]["y"] + 63) * scale, "Require E(z̄) ≤ -2.2082 and p_top1 ≥ 0.50", f_sub_card, f_djv_sub, PALETTE["muted"])

    _draw_text_left(draw, (left_x + 16) * scale, (boxes[4]["y"] + 21) * scale, "4. High-Stakes Safety Veto", f_title_card, PALETTE["ink"])
    _draw_mixed_left(draw, (left_x + 16) * scale, (boxes[4]["y"] + 44) * scale, "Top-1 edible AND any raw frame has p_toxic ≥ 0.20?", f_sub_card, f_djv_sub, PALETTE["muted"])

    _draw_text_left(draw, (left_x + 16) * scale, (boxes[5]["y"] + 18) * scale, "OK  —  Accepted Prediction", f_title_card, PALETTE["green"])
    _draw_text_left(draw, (left_x + 16) * scale, (boxes[5]["y"] + 38) * scale, "Taxonomic Match + Calibrated Confidence", _font("semibold", 11, scale), PALETTE["ink"])

    # Downward pass labels
    pass_labels = [
        (boxes[1], boxes[2], "  ≥ 1 accepted"),
        (boxes[2], boxes[3], "  Consistent"),
        (boxes[3], boxes[4], "  Gates pass"),
        (boxes[4], boxes[5], "  No toxic risk"),
    ]
    f_lbl = _font("semibold", 10, scale)
    f_djv_lbl = _dejavu_font("semibold", 10, scale)
    for b_top, b_bot, lbl in pass_labels:
        ym = (b_top["y"] + b_top["h"] + b_bot["y"]) / 2.0
        _draw_mixed_left(draw, (cx_left + 8) * scale, ym * scale, lbl, f_lbl, f_djv_lbl, PALETTE["muted"])

    # Branch arrow labels + Exit Card Texts
    branch_specs = [
        (exits[0], "All rejected", "ALL_FRAMES_OOD", "No valid mushroom frame", PALETTE["red"]),
        (exits[1], "Conflict", "CONFLICT_DETECTED", "Confident angle mismatch", PALETTE["amber"]),
        (exits[2], "Gate fail", "OOD / UNCLEAR", "E(z̄) > θ or p < 0.50", PALETTE["amber"]),
        (exits[3], "Yes (risk)", "HIGH_STAKES_VETO", "Toxic look-alike flagged", PALETTE["red"]),
    ]
    mid_x = (left_x + left_w + right_x) / 2.0
    for ex, br_lbl, title, sub, col in branch_specs:
        y_mid = ex["y"] + ex["h"] / 2.0
        _draw_text_centered(draw, mid_x * scale, (y_mid - 11) * scale, br_lbl, f_lbl, col)
        _draw_text_left(draw, (right_x + 16) * scale, (ex["y"] + 21) * scale, title, _font("bold", 12, scale), col)
        _draw_mixed_left(draw, (right_x + 16) * scale, (ex["y"] + 43) * scale, sub, _font("regular", 10.5, scale), _dejavu_font("regular", 10.5, scale), PALETTE["muted"])

    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(out_png, dpi=(300, 300))
    return {"path": str(out_png), "size_px": f"{px_w}x{px_h}"}


def main() -> None:
    ood_png = OUTPUT_DIR / "ood_energy_distribution.png"
    acc_png = OUTPUT_DIR / "accuracy_progression.png"
    rel_png = OUTPUT_DIR / "reliability_diagram.png"
    fus_png = OUTPUT_DIR / "fusion_pipeline.png"

    ood_stats = build_ood_energy_chart(ood_png)
    acc_stats = build_accuracy_progression_chart(acc_png)
    rel_stats = build_reliability_diagram_chart(rel_png)
    fus_stats = build_fusion_pipeline_chart(fus_png)

    print("Generated Pileus-styled charts:")
    print(f" - {ood_png}: {ood_stats}")
    print(f" - {acc_png}: {acc_stats}")
    print(f" - {rel_png}: {rel_stats}")
    print(f" - {fus_png}: {fus_stats}")


if __name__ == "__main__":
    main()
