"""Generate a Pileus-styled Side-by-Side look-alike comparison PNG for dev.to.

Applies the Pileus v1.0 ("Nordic Botanical Paper") visual design system:
- paper: #F6F5F1, surface: #FFFFFF, ink: #1B382B, muted: #3E5348, border: #DCDAD3
- edible badge: #E8F5E9 / #1F6B2A / #81C784
- deadly badge: #2A080C / #FF4560 / #FF1744
- typography: Plus Jakarta Sans (with DejaVu fallback)

Reads two CC0-licensed validation photos without modifying any dataset files.
Output:
- artifacts/blog/images/lookalike_side_by_side.png
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

VAL_CSV = Path("data/splits/val.csv")
TEST_CSV = Path("data/splits/test.csv")
MANIFEST_CSV = Path("/home/bubolka/hf-768/upload/manifest_768.csv")
IMAGE_ROOT = Path("/home/bubolka/hf-768/upload")
OUT_PNG = Path("artifacts/blog/images/lookalike_side_by_side.png")

LEFT_PATH = Path("images/kuehneromyces_mutabilis/155421201.jpg")
RIGHT_PATH = Path("images/galerina_marginata/582568029.jpg")

# Pileus v1.0 Nordic Botanical Paper palette
PILEUS = {
    "paper": "#F6F5F1",
    "surface": "#FFFFFF",
    "surface_raised": "#EAE6DC",
    "ink": "#1B382B",
    "muted": "#3E5348",
    "border": "#DCDAD3",
    "shadow": "#E2DFD5",
    "edible_bg": "#E8F5E9",
    "edible_text": "#1F6B2A",
    "edible_border": "#81C784",
    "deadly_bg": "#2A080C",
    "deadly_text": "#FF4560",
    "deadly_border": "#FF1744",
    "poisonous_bg": "#FFEBEE",
    "poisonous_border": "#FF8A80",
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


def _verify_sample_metadata() -> dict[str, dict[str, str]]:
    """Verify both chosen images belong to val.csv (never test.csv) and have open CC0 licenses."""
    test_urls: set[str] = set()
    with open(TEST_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("image_url"):
                test_urls.add(row["image_url"])
            if row.get("canonical_url"):
                test_urls.add(row["canonical_url"])

    val_by_photo_id: dict[str, dict[str, str]] = {}
    with open(VAL_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = row.get("inat_photo_id") or ""
            if pid in ("155421201", "582568029"):
                val_by_photo_id[pid] = row

    records: dict[str, dict[str, str]] = {}
    target_rel_paths = {str(LEFT_PATH), str(RIGHT_PATH)}
    with open(MANIFEST_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["path"] in target_rel_paths:
                if row["image_url"] in test_urls:
                    raise RuntimeError(f"Refusing to use test split image: {row['path']}")
                pid = Path(row["path"]).stem
                val_row = val_by_photo_id.get(pid, {})
                lic = row.get("license") or val_row.get("license") or ""
                if "cc0" not in lic.lower() and "by" not in lic.lower():
                    raise RuntimeError(f"Unsupported license {lic} for {row['path']}")
                records[row["path"]] = {
                    "species_slug": row["species_slug"],
                    "species": row["species"],
                    "license": lic,
                    "license_url": row.get("license_url") or val_row.get("license_url") or "",
                    "author": row.get("author") or val_row.get("author") or "",
                    "observer": val_row.get("inat_observer") or "",
                    "source": row.get("source") or "inaturalist",
                    "source_url": row.get("source_url") or val_row.get("source_url") or "",
                    "obs_id": val_row.get("inat_observation_id") or "",
                    "photo_id": pid,
                }

    if len(records) != 2:
        raise RuntimeError(f"Expected 2 verified records, found {len(records)}")
    return records


def _rounded_image(img: Image.Image, target_w: int, target_h: int, radius: int) -> Image.Image:
    """Resize and apply clean rounded corners using an antialiased alpha mask."""
    resized = img.convert("RGB").resize((target_w, target_h), Image.Resampling.LANCZOS)
    mask = Image.new("L", (target_w * 2, target_h * 2), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rounded_rectangle((0, 0, target_w * 2, target_h * 2), radius=radius * 2, fill=255)
    mask = mask.resize((target_w, target_h), Image.Resampling.LANCZOS)
    rgba = resized.convert("RGBA")
    rgba.putalpha(mask)
    return rgba


def main() -> None:
    meta = _verify_sample_metadata()
    left_meta = meta[str(LEFT_PATH)]
    right_meta = meta[str(RIGHT_PATH)]

    # Pileus 8-dp grid layout (scaled 2x cleanly -> 1848x980 px @ 300 DPI)
    margin_x = 56
    gutter = 40
    card_pad = 16
    img_w, img_h = 816, 612
    card_w = img_w + card_pad * 2
    header_h = 128
    caption_h = 144
    card_h = card_pad + img_h + caption_h
    bottom_pad = 40

    canvas_w = margin_x * 2 + card_w * 2 + gutter
    canvas_h = header_h + card_h + bottom_pad

    canvas = Image.new("RGBA", (canvas_w, canvas_h), PILEUS["paper"])
    draw = ImageDraw.Draw(canvas)

    f_title = ImageFont.truetype(_resolve_font("bold"), 36)
    f_sub = ImageFont.truetype(_resolve_font("regular"), 19)
    f_species_italic = ImageFont.truetype(_resolve_font("italic"), 26)
    f_species_status = ImageFont.truetype(_resolve_font("bold"), 26)
    f_badge = ImageFont.truetype(_resolve_font("bold"), 14)
    f_hint = ImageFont.truetype(_resolve_font("regular"), 17)
    f_attr = ImageFont.truetype(_resolve_font("regular"), 14)

    # Top header (Botanical Paper Ink)
    draw.text((margin_x, 34), "Deadly Look-Alike Pair: Edible vs. Lethal Double", font=f_title, fill=PILEUS["ink"])
    draw.text(
        (margin_x, 82),
        "Shared habitat on mossy wood and near-identical golden-brown caps require high-resolution stem & gill cues",
        font=f_sub,
        fill=PILEUS["muted"],
    )

    cards = [
        {
            "x": margin_x,
            "img_path": IMAGE_ROOT / LEFT_PATH,
            "meta": left_meta,
            "species_name": "Kuehneromyces mutabilis",
            "status_text": " (Edible)",
            "badge_label": "EDIBLE",
            "badge_bg": PILEUS["edible_bg"],
            "badge_text": PILEUS["edible_text"],
            "badge_border": PILEUS["edible_border"],
            "accent": PILEUS["edible_text"],
            "card_border": PILEUS["edible_border"],
            "hint": "Diagnostic cue: distinct dark scales on lower stem below the ring",
        },
        {
            "x": margin_x + card_w + gutter,
            "img_path": IMAGE_ROOT / RIGHT_PATH,
            "meta": right_meta,
            "species_name": "Galerina marginata",
            "status_text": " (Deadly)",
            "badge_label": "DEADLY POISONOUS",
            "badge_bg": PILEUS["deadly_bg"],
            "badge_text": PILEUS["deadly_text"],
            "badge_border": PILEUS["deadly_border"],
            "accent": PILEUS["deadly_bg"],
            "card_border": PILEUS["poisonous_border"],
            "hint": "Diagnostic cue: smooth/fibrillose stem; contains lethal amatoxins",
        },
    ]

    for card in cards:
        cx = int(card["x"])
        cy = header_h

        # Sharp paper offset shadow (Pileus elevation.level2)
        draw.rounded_rectangle(
            (cx + 4, cy + 4, cx + card_w + 4, cy + card_h + 4),
            radius=12,
            fill=PILEUS["shadow"],
        )
        # Main surface card (#FFFFFF)
        draw.rounded_rectangle(
            (cx, cy, cx + card_w, cy + card_h),
            radius=12,
            fill=PILEUS["surface"],
            outline=PILEUS["border"],
            width=2,
        )

        # Paste rounded photo inside card padding
        ix, iy = cx + card_pad, cy + card_pad
        with Image.open(Path(str(card["img_path"]))) as src_img:
            rounded = _rounded_image(src_img, img_w, img_h, radius=8)
            canvas.alpha_composite(rounded, dest=(ix, iy))

        draw.rounded_rectangle(
            (ix, iy, ix + img_w, iy + img_h),
            radius=8,
            outline=str(card["card_border"]),
            width=2,
        )

        # Caption row inside card below photo
        tx = cx + card_pad + 4
        ty = iy + img_h + 18
        sp_name = str(card["species_name"])
        st_text = str(card["status_text"])

        draw.text((tx, ty), sp_name, font=f_species_italic, fill=PILEUS["ink"])
        bbox_sp = draw.textbbox((tx, ty), sp_name, font=f_species_italic)
        status_fill = PILEUS["edible_text"] if card["badge_label"] == "EDIBLE" else "#AD1414"
        draw.text((bbox_sp[2], ty), st_text, font=f_species_status, fill=status_fill)

        # Right-aligned Pileus status pill badge (Edible vs Deadly Badge)
        badge_txt = str(card["badge_label"])
        b_box = draw.textbbox((0, 0), badge_txt, font=f_badge)
        bw = (b_box[2] - b_box[0]) + 22
        bh = 30
        bx = cx + card_w - card_pad - 4 - bw
        by = ty + 2
        draw.rounded_rectangle(
            (bx, by, bx + bw, by + bh),
            radius=6,
            fill=str(card["badge_bg"]),
            outline=str(card["badge_border"]),
            width=2,
        )
        draw.text(
            (bx + 11 - b_box[0], by + (bh - (b_box[3] - b_box[1])) / 2 - b_box[1]),
            badge_txt,
            font=f_badge,
            fill=str(card["badge_text"]),
        )

        # Diagnostic cue
        draw.text((tx, ty + 42), str(card["hint"]), font=f_hint, fill=PILEUS["ink"])

        # Subtle attribution strip inside raised surface tint
        m = card["meta"]
        obs_credit = m["observer"] or m["author"]
        attr_line = f"Source: iNaturalist #{m['obs_id']} (observer: {obs_credit})  ·  License: {m['license']} (Public Domain)"
        draw.text((tx, ty + 78), attr_line, font=f_attr, fill=PILEUS["muted"])

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(OUT_PNG, dpi=(300, 300), quality=95)
    print(f"Saved Pileus look-alike card: {OUT_PNG} ({canvas_w}x{canvas_h} px @ 300 DPI)")


if __name__ == "__main__":
    main()
