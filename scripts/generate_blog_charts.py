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

# Pileus v1.0 Nordic Botanical Paper palette
PILEUS = {
    "paper": "#F6F5F1",
    "surface": "#FFFFFF",
    "surface_raised": "#EAE6DC",
    "ink": "#1B382B",
    "muted": "#3E5348",
    "border": "#DCDAD3",
    "border_strong": "#767B74",
    "shadow": "#E2DFD5",
    "edible_text": "#1F6B2A",
    "edible_bg": "#E8F5E9",
    "edible_border": "#81C784",
    "ochre": "#B45309",
    "danger": "#AD1414",
    "poisonous_text": "#B71C1C",
    "poisonous_bg": "#FFEBEE",
    "deadly_bg": "#2A080C",
    "deadly_text": "#FF4560",
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

    w, h = 1000, 580
    scale = 3.0
    px_w, px_h = round(w * scale), round(h * scale)
    ml, mr, mt, mb = 86, 44, 114, 74
    pw, ph = w - ml - mr, h - mt - mb

    x_min, x_max = -6.6, 0.6
    y_min, y_max = 0.0, 0.86

    def sx(x: float) -> float:
        return ml + (x - x_min) / (x_max - x_min) * pw

    def sy(y: float) -> float:
        return mt + ph - (y - y_min) / (y_max - y_min) * ph

    series = [
        ("Known Mushrooms (val, n=13,065)", e_val, PILEUS["edible_text"], 0.16, 2.8),
        ("Unknown Fungi (ood_fungi, n=2,580)", e_fungi, PILEUS["ochre"], 0.17, 2.8),
        ("Non-Fungi (ood_nonfungi, n=1,880)", e_nonfungi, PILEUS["danger"], 0.16, 2.8),
    ]

    bins = np.linspace(x_min, x_max, 49)
    xs_kde = np.linspace(x_min, x_max, 240)
    tx = sx(theta)

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="10in" height="5.8in" viewBox="0 0 {w} {h}">',
        f'<rect width="100%" height="100%" fill="{PILEUS["paper"]}"/>',
        # Crisp Pileus sharp paper shadow + card surface
        f'<rect x="{ml + 3}" y="{mt + 3}" width="{pw}" height="{ph}" rx="8" fill="{PILEUS["shadow"]}"/>',
        f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" rx="8" fill="{PILEUS["surface"]}" stroke="{PILEUS["border"]}" stroke-width="1.4"/>',
        # Subtle tinted halves inside card
        f'<rect x="{ml + 1}" y="{mt + 1}" width="{tx - ml - 1:.2f}" height="{ph - 2}" rx="6" fill="#E8F5E9" opacity="0.35"/>',
        f'<rect x="{tx:.2f}" y="{mt + 1}" width="{ml + pw - tx - 1:.2f}" height="{ph - 2}" rx="6" fill="#FFEBEE" opacity="0.30"/>',
    ]

    # Horizontal gridlines
    for y_tick in [0.2, 0.4, 0.6, 0.8]:
        yy = sy(y_tick)
        svg.append(f'<line x1="{ml}" y1="{yy:.2f}" x2="{ml + pw}" y2="{yy:.2f}" stroke="{PILEUS["border"]}" stroke-width="1"/>')

    # Histograms & KDE curves
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

    # Crisp bottom baseline & ticks
    svg.append(f'<line x1="{ml}" y1="{mt + ph}" x2="{ml + pw}" y2="{mt + ph}" stroke="{PILEUS["border_strong"]}" stroke-width="1.2"/>')
    svg.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + ph}" stroke="{PILEUS["border_strong"]}" stroke-width="1.2"/>')
    for x_tick in [-6, -5, -4, -3, -2, -1, 0]:
        xx = sx(float(x_tick))
        svg.append(f'<line x1="{xx:.2f}" y1="{mt + ph}" x2="{xx:.2f}" y2="{mt + ph + 5}" stroke="{PILEUS["border_strong"]}" stroke-width="1.2"/>')

    # Vertical dashed threshold
    svg.extend(_dashed_line_svg(tx, mt + 30, tx, mt + ph, PILEUS["ink"], 2.4, dash_len=9.0, gap_len=6.0))
    # Dark ink pill for threshold label
    svg.append(f'<rect x="{tx - 98:.2f}" y="{mt + 8}" width="196" height="26" rx="6" fill="{PILEUS["ink"]}"/>')

    # Legend card
    leg_x, leg_y = ml + 14, mt + 14
    svg.append(f'<rect x="{leg_x + 2}" y="{leg_y + 2}" width="318" height="84" rx="6" fill="{PILEUS["shadow"]}"/>')
    svg.append(f'<rect x="{leg_x}" y="{leg_y}" width="318" height="84" rx="6" fill="{PILEUS["surface"]}" stroke="{PILEUS["border"]}" stroke-width="1.2"/>')
    for idx, (_, _, color, _, _) in enumerate(series):
        ly = leg_y + 22 + idx * 23
        svg.append(f'<rect x="{leg_x + 14}" y="{ly - 9}" width="20" height="10" rx="2" fill="{color}"/>')

    svg.append("</svg>")
    canvas = _svg_to_pil("\n".join(svg), px_w, px_h)
    draw = ImageDraw.Draw(canvas)

    # Crisp Plus Jakarta Sans Typography
    _draw_text_left(draw, ml * scale, 34 * scale, "Out-of-Distribution (OOD) Energy Score Distribution", _font("bold", 21, scale), PILEUS["ink"])
    _draw_text_left(draw, ml * scale, 60 * scale, f"Energy score E(x) = −T · logsumexp(z / T) across validation & OOD splits (T = {temp:.4f})", _font("regular", 13, scale), PILEUS["muted"])

    # Y ticks
    for y_tick in [0.0, 0.2, 0.4, 0.6, 0.8]:
        _draw_text_right(draw, (ml - 10) * scale, sy(y_tick) * scale, f"{y_tick:.1f}", _font("regular", 11.5, scale), PILEUS["muted"])

    # X ticks
    for x_tick in [-6, -5, -4, -3, -2, -1, 0]:
        _draw_text_centered(draw, sx(float(x_tick)) * scale, (mt + ph + 18) * scale, str(x_tick), _font("regular", 11.5, scale), PILEUS["muted"])

    # Threshold pill text (render Greek theta with DejaVu Sans Bold so glyph never tofus) & directional badges
    f_pill = _font("bold", 12, scale)
    f_theta = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", max(10, round(12 * scale)))
    prefix_txt = "Threshold "
    theta_txt = "θ"
    suffix_txt = f" = {theta:.4f}"
    w1 = draw.textlength(prefix_txt, font=f_pill)
    w2 = draw.textlength(theta_txt, font=f_theta)
    w3 = draw.textlength(suffix_txt, font=f_pill)
    total_w = w1 + w2 + w3
    pill_cy = (mt + 21) * scale
    start_x = tx * scale - total_w / 2.0
    _draw_text_left(draw, start_x, pill_cy, prefix_txt, f_pill, "#FFFFFF")
    _draw_text_left(draw, start_x + w1, pill_cy, theta_txt, f_theta, "#FFFFFF")
    _draw_text_left(draw, start_x + w1 + w2, pill_cy, suffix_txt, f_pill, "#FFFFFF")
    _draw_text_right(draw, (tx - 14) * scale, (mt + 48) * scale, "← Accepted (In-Distribution)", _font("bold", 12, scale), PILEUS["edible_text"])
    _draw_text_left(draw, (tx + 14) * scale, (mt + 48) * scale, "Rejected (OOD) →", _font("bold", 12, scale), PILEUS["danger"])

    # Legend text
    for idx, (label, _, _, _, _) in enumerate(series):
        ly = leg_y + 22 + idx * 23 - 4
        _draw_text_left(draw, (leg_x + 42) * scale, ly * scale, label, _font("semibold", 11.5, scale), PILEUS["ink"])

    # Axis labels
    _draw_text_centered(draw, (ml + pw / 2) * scale, (h - 20) * scale, "Energy Score E(x)  (Lower = More In-Distribution Confidence)", _font("semibold", 12.5, scale), PILEUS["ink"])
    _draw_rotated_y_label(canvas, 28 * scale, (mt + ph / 2) * scale, "Probability Density", _font("semibold", 12.5, scale), PILEUS["ink"])

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
            "title_line2": "+ Multi-Photo P5",
            "acc": acc_p5_val,
            "acc_sub": f"({acc_p5_test:.2f}% test)",
            "rec": rec_p5_val,
            "rec_sub": f"({rec_test_gated:.1f}% test gated)",
        },
    ]

    w, h = 1000, 580
    scale = 3.0
    px_w, px_h = round(w * scale), round(h * scale)
    ml, mr, mt, mb = 84, 44, 120, 86
    pw, ph = w - ml - mr, h - mt - mb

    y_min, y_max = 0.0, 108.0

    def sy(y: float) -> float:
        return mt + ph - (y - y_min) / (y_max - y_min) * ph

    color_acc = PILEUS["ink"]     # Botanical forest green-black (#1B382B)
    color_rec = PILEUS["danger"]  # Pileus danger crimson (#AD1414)

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="10in" height="5.8in" viewBox="0 0 {w} {h}">',
        f'<rect width="100%" height="100%" fill="{PILEUS["paper"]}"/>',
        # Crisp sharp paper card
        f'<rect x="{ml + 3}" y="{mt + 3}" width="{pw}" height="{ph}" rx="8" fill="{PILEUS["shadow"]}"/>',
        f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" rx="8" fill="{PILEUS["surface"]}" stroke="{PILEUS["border"]}" stroke-width="1.4"/>',
    ]

    # Gridlines (20..100%)
    for y_tick in [20, 40, 60, 80, 100]:
        yy = sy(float(y_tick))
        svg.append(f'<line x1="{ml}" y1="{yy:.2f}" x2="{ml + pw}" y2="{yy:.2f}" stroke="{PILEUS["border"]}" stroke-width="1"/>')

    # Dashed 85% high-stakes safety target
    y_target = sy(85.0)
    svg.extend(_dashed_line_svg(ml, y_target, ml + pw, y_target, PILEUS["border_strong"], 1.7, dash_len=7.0, gap_len=5.0))

    n_groups = len(stages)
    group_width = pw / n_groups
    bar_w = 78
    gap = 18

    for i, st in enumerate(stages):
        cx = ml + (i + 0.5) * group_width
        x1 = cx - gap / 2 - bar_w
        x2 = cx + gap / 2

        y1 = sy(st["acc"])
        h1 = sy(0.0) - y1
        svg.append(f'<rect x="{x1:.2f}" y="{y1:.2f}" width="{bar_w}" height="{h1:.2f}" rx="4" fill="{color_acc}"/>')
        if not st["acc_sub"]:
            # Clean white backdrop behind 78.85% so 85% dashed line breaks cleanly
            svg.append(f'<rect x="{x1 + bar_w / 2 - 32:.2f}" y="{y1 - 24:.2f}" width="64" height="18" rx="3" fill="{PILEUS["surface"]}"/>')

        y2 = sy(st["rec"])
        h2 = sy(0.0) - y2
        svg.append(f'<rect x="{x2:.2f}" y="{y2:.2f}" width="{bar_w}" height="{h2:.2f}" rx="4" fill="{color_rec}"/>')

    # Crisp spines
    svg.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + ph}" stroke="{PILEUS["border_strong"]}" stroke-width="1.2"/>')
    svg.append(f'<line x1="{ml}" y1="{mt + ph}" x2="{ml + pw}" y2="{mt + ph}" stroke="{PILEUS["border_strong"]}" stroke-width="1.2"/>')

    # Legend swatches
    leg_x, leg_y = ml, 80
    svg.append(f'<rect x="{leg_x}" y="{leg_y}" width="15" height="15" rx="3" fill="{color_acc}"/>')
    svg.append(f'<rect x="{leg_x + 178}" y="{leg_y}" width="15" height="15" rx="3" fill="{color_rec}"/>')

    svg.append("</svg>")
    canvas = _svg_to_pil("\n".join(svg), px_w, px_h)
    draw = ImageDraw.Draw(canvas)

    # Typography
    _draw_text_left(draw, ml * scale, 34 * scale, "Classification Accuracy & Death Cap Recall Progression", _font("bold", 21, scale), PILEUS["ink"])
    _draw_text_left(draw, ml * scale, 60 * scale, "Impact of input resolution (224 → 256 px), model capacity, and Multi-Photo Logit Fusion (P5)", _font("regular", 13, scale), PILEUS["muted"])

    _draw_text_left(draw, (leg_x + 22) * scale, (leg_y + 7.5) * scale, "Overall Accuracy", _font("semibold", 12.5, scale), PILEUS["ink"])
    _draw_text_left(draw, (leg_x + 200) * scale, (leg_y + 7.5) * scale, "Death Cap (Amanita phalloides) Recall", _font("semibold", 12.5, scale), PILEUS["ink"])

    for y_tick in [0, 20, 40, 60, 80, 100]:
        _draw_text_right(draw, (ml - 10) * scale, sy(float(y_tick)) * scale, f"{y_tick}%", _font("regular", 11.5, scale), PILEUS["muted"])

    _draw_text_left(draw, (ml + 14) * scale, (y_target - 12) * scale, "High-Stakes Safety Target (85%)", _font("semibold", 11, scale), PILEUS["muted"])

    for i, st in enumerate(stages):
        cx = ml + (i + 0.5) * group_width
        x1 = cx - gap / 2 - bar_w
        x2 = cx + gap / 2

        y1 = sy(st["acc"])
        if st["acc_sub"]:
            _draw_text_centered(draw, (x1 + bar_w / 2) * scale, (y1 - 26) * scale, f"{st['acc']:.2f}%", _font("bold", 13.5, scale), PILEUS["ink"])
            _draw_text_centered(draw, (x1 + bar_w / 2) * scale, (y1 - 11) * scale, st["acc_sub"], _font("semibold", 10.2, scale), PILEUS["muted"])
        else:
            _draw_text_centered(draw, (x1 + bar_w / 2) * scale, (y1 - 14) * scale, f"{st['acc']:.2f}%", _font("bold", 13.5, scale), PILEUS["ink"])

        y2 = sy(st["rec"])
        if st["rec_sub"]:
            _draw_text_centered(draw, (x2 + bar_w / 2) * scale, (y2 - 26) * scale, f"{st['rec']:.1f}%", _font("bold", 13.5, scale), PILEUS["danger"])
            _draw_text_centered(draw, (x2 + bar_w / 2) * scale, (y2 - 11) * scale, st["rec_sub"], _font("semibold", 10.2, scale), PILEUS["muted"])
        else:
            _draw_text_centered(draw, (x2 + bar_w / 2) * scale, (y2 - 14) * scale, f"{st['rec']:.1f}%", _font("bold", 13.5, scale), PILEUS["danger"])

        _draw_text_centered(draw, cx * scale, (sy(0.0) + 21) * scale, st["title_line1"], _font("bold", 12.5, scale), PILEUS["ink"])
        _draw_text_centered(draw, cx * scale, (sy(0.0) + 40) * scale, st["title_line2"], _font("regular", 12, scale), PILEUS["muted"])

    _draw_rotated_y_label(canvas, 28 * scale, (mt + ph / 2) * scale, "Score (%)", _font("semibold", 12.5, scale), PILEUS["ink"])

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

    w, h = 1000, 566
    scale = 3.0
    px_w, px_h = round(w * scale), round(h * scale)

    ml, mr, mt, mb = 78, 42, 108, 98
    gutter = 54
    pw = (w - ml - mr - gutter) / 2.0
    ph = h - mt - mb

    panels = [
        {
            "x0": ml,
            "title": "Uncalibrated (T = 1.0000)",
            "badge": f"ECE: {ece_raw * 100:.2f}%  ·  MCE: {mce_raw * 100:.2f}%",
            "badge_bg": "#FFF3E0",
            "badge_text": "#9A3412",
            "badge_border": "#FFB74D",
            "bar_color": "#767B74",
            "bins": bins_raw,
        },
        {
            "x0": ml + pw + gutter,
            "title": f"Calibrated (T = {temp:.4f})",
            "badge": f"ECE: {ece_cal * 100:.2f}%  ·  MCE: {mce_cal * 100:.2f}%",
            "badge_bg": PILEUS["edible_bg"],
            "badge_text": PILEUS["edible_text"],
            "badge_border": PILEUS["edible_border"],
            "bar_color": PILEUS["edible_text"],
            "bins": bins_cal,
        },
    ]

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="10in" height="5.66in" viewBox="0 0 {w} {h}">',
        f'<rect width="100%" height="100%" fill="{PILEUS["paper"]}"/>',
    ]

    for p in panels:
        x0 = float(p["x0"])
        y0 = float(mt)
        # Paper card shadow & surface
        svg.append(f'<rect x="{x0 + 3:.2f}" y="{y0 + 3:.2f}" width="{pw:.2f}" height="{ph:.2f}" rx="8" fill="{PILEUS["shadow"]}"/>')
        svg.append(f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{pw:.2f}" height="{ph:.2f}" rx="8" fill="{PILEUS["surface"]}" stroke="{PILEUS["border"]}" stroke-width="1.4"/>')

        # Gridlines 0.2..0.8
        for t_val in [0.2, 0.4, 0.6, 0.8]:
            yy = y0 + ph * (1.0 - t_val)
            xx = x0 + pw * t_val
            svg.append(f'<line x1="{x0:.2f}" y1="{yy:.2f}" x2="{x0 + pw:.2f}" y2="{yy:.2f}" stroke="{PILEUS["border"]}" stroke-width="1"/>')
            svg.append(f'<line x1="{xx:.2f}" y1="{y0:.2f}" x2="{xx:.2f}" y2="{y0 + ph:.2f}" stroke="{PILEUS["border"]}" stroke-width="1"/>')

        # Diagonal perfect calibration dashed line
        svg.extend(_dashed_line_svg(x0, y0 + ph, x0 + pw, y0, PILEUS["border_strong"], 1.8, dash_len=7.0, gap_len=5.0))

        # 15 bins
        for b in p["bins"]:
            bx0 = x0 + float(b["low"]) * pw + 1.5
            bx1 = x0 + float(b["high"]) * pw - 1.5
            bw = max(2.0, bx1 - bx0)
            if b["count"] > 0:
                acc = float(b["acc"])
                conf = float(b["conf"])
                by = y0 + ph * (1.0 - acc)
                bh = (y0 + ph) - by
                svg.append(
                    f'<rect x="{bx0:.2f}" y="{by:.2f}" width="{bw:.2f}" height="{bh:.2f}" rx="2" '
                    f'fill="{p["bar_color"]}" opacity="0.86" stroke="{PILEUS["surface"]}" stroke-width="0.8"/>'
                )
                cy_conf = y0 + ph * (1.0 - conf)
                svg.append(
                    f'<rect x="{bx0 - 1:.2f}" y="{cy_conf - 1.5:.2f}" width="{bw + 2:.2f}" height="3.0" rx="1" fill="{PILEUS["danger"]}"/>'
                )

        # Panel badge
        svg.append(
            f'<rect x="{x0 + 12:.2f}" y="{y0 + 12:.2f}" width="212" height="24" rx="5" '
            f'fill="{p["badge_bg"]}" stroke="{p["badge_border"]}" stroke-width="1"/>'
        )

        # Spines
        svg.append(f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x0:.2f}" y2="{y0 + ph:.2f}" stroke="{PILEUS["border_strong"]}" stroke-width="1.2"/>')
        svg.append(f'<line x1="{x0:.2f}" y1="{y0 + ph:.2f}" x2="{x0 + pw:.2f}" y2="{y0 + ph:.2f}" stroke="{PILEUS["border_strong"]}" stroke-width="1.2"/>')

    # Bottom legend swatches
    leg_y = h - 22
    svg.append(f'<rect x="{ml + 170}" y="{leg_y - 6}" width="16" height="12" rx="2" fill="{PILEUS["edible_text"]}"/>')
    svg.append(f'<rect x="{ml + 390}" y="{leg_y - 2}" width="20" height="4" rx="1" fill="{PILEUS["danger"]}"/>')

    svg.append("</svg>")
    canvas = _svg_to_pil("\n".join(svg), px_w, px_h)
    draw = ImageDraw.Draw(canvas)

    # Main Title & Subtitle
    _draw_text_left(draw, ml * scale, 30 * scale, "Reliability Diagram (15 Bins) — Validation Set", _font("bold", 21, scale), PILEUS["ink"])
    _draw_text_left(draw, ml * scale, 54 * scale, "Temperature scaling removes label-smoothing underconfidence (NLL 0.9267 → 0.8527, ECE 3.55× lower)", _font("regular", 13, scale), PILEUS["muted"])

    for idx, p in enumerate(panels):
        x0 = float(p["x0"])
        y0 = float(mt)
        _draw_text_left(draw, x0 * scale, (y0 - 14) * scale, str(p["title"]), _font("bold", 14, scale), PILEUS["ink"])
        _draw_text_centered(draw, (x0 + 118) * scale, (y0 + 24) * scale, str(p["badge"]), _font("bold", 11, scale), str(p["badge_text"]))

        for t_val in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            _draw_text_centered(draw, (x0 + pw * t_val) * scale, (y0 + ph + 16) * scale, f"{t_val:.1f}", _font("regular", 10.5, scale), PILEUS["muted"])
            if idx == 0:
                _draw_text_right(draw, (x0 - 8) * scale, (y0 + ph * (1.0 - t_val)) * scale, f"{t_val:.1f}", _font("regular", 10.5, scale), PILEUS["muted"])

        _draw_text_centered(draw, (x0 + pw / 2) * scale, (y0 + ph + 36) * scale, "Predicted Confidence", _font("semibold", 11.5, scale), PILEUS["ink"])

    _draw_rotated_y_label(canvas, 24 * scale, (mt + ph / 2) * scale, "Empirical Accuracy", _font("semibold", 12, scale), PILEUS["ink"])
    _draw_text_left(draw, (ml + 194) * scale, leg_y * scale, "Bin Accuracy Bar", _font("semibold", 11.5, scale), PILEUS["ink"])
    _draw_text_left(draw, (ml + 418) * scale, leg_y * scale, "Mean Bin Confidence (Gap = Calibration Error)", _font("semibold", 11.5, scale), PILEUS["ink"])

    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(out_png, dpi=(300, 300))

    return {"ece_raw": ece_raw, "ece_cal": ece_cal, "temp": temp}


def main() -> None:
    ood_png = OUTPUT_DIR / "ood_energy_distribution.png"
    acc_png = OUTPUT_DIR / "accuracy_progression.png"
    rel_png = OUTPUT_DIR / "reliability_diagram.png"

    ood_stats = build_ood_energy_chart(ood_png)
    acc_stats = build_accuracy_progression_chart(acc_png)
    rel_stats = build_reliability_diagram_chart(rel_png)

    print("Generated Pileus-styled charts:")
    print(f" - {ood_png}: {ood_stats}")
    print(f" - {acc_png}: {acc_stats}")
    print(f" - {rel_png}: {rel_stats}")


if __name__ == "__main__":
    main()
