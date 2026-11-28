"""Generate a flat editorial Side-by-Side look-alike comparison JPG for dev.to.

Applies a minimalist flat editorial style on #F7F6F2:
- No white cards, no drop shadows
- 6 px rounded photo corners with thin 1 px #DCD9CF border
- Lightweight outline pills (EDIBLE in #163827, DEADLY in #A61C1C)
- Typography: Plus Jakarta Sans (with DejaVu fallback)

Reads two CC0-licensed validation photos without modifying any dataset files.
Output:
- artifacts/blog/images/lookalike_side_by_side.jpg (<= 1600 px, JPEG quality 85)
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
OUT_JPG = Path("artifacts/blog/images/lookalike_side_by_side.jpg")

LEFT_PATH = Path("images/kuehneromyces_mutabilis/155421201.jpg")
RIGHT_PATH = Path("images/galerina_marginata/582568029.jpg")

PALETTE = {
    "paper": "#F7F6F2",
    "ink": "#1A1F1C",
    "muted": "#5C635E",
    "border": "#DCD9CF",
    "green": "#163827",
    "red": "#A61C1C",
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

    # Flat editorial layout (1560 px wide <= 1600 px, generous whitespace)
    margin_x = 56
    gutter = 48
    img_w, img_h = 700, 525
    header_h = 128
    caption_h = 134
    bottom_pad = 36

    canvas_w = margin_x * 2 + img_w * 2 + gutter
    canvas_h = header_h + img_h + caption_h + bottom_pad

    canvas = Image.new("RGBA", (canvas_w, canvas_h), PALETTE["paper"])
    draw = ImageDraw.Draw(canvas)

    f_title = ImageFont.truetype(_resolve_font("bold"), 32)
    f_sub = ImageFont.truetype(_resolve_font("regular"), 17)
    f_species_italic = ImageFont.truetype(_resolve_font("italic"), 23)
    f_species_status = ImageFont.truetype(_resolve_font("semibold"), 22)
    f_badge = ImageFont.truetype(_resolve_font("semibold"), 12)
    f_hint = ImageFont.truetype(_resolve_font("regular"), 15)
    f_attr = ImageFont.truetype(_resolve_font("regular"), 13)

    # Top header directly on #F7F6F2
    draw.text((margin_x, 34), "Deadly Look-Alike Pair: Edible vs. Lethal Double", font=f_title, fill=PALETTE["ink"])
    draw.text(
        (margin_x, 78),
        "Shared habitat on mossy wood and near-identical golden-brown caps require high-resolution stem & gill cues",
        font=f_sub,
        fill=PALETTE["muted"],
    )

    items = [
        {
            "x": margin_x,
            "img_path": IMAGE_ROOT / LEFT_PATH,
            "meta": left_meta,
            "species_name": "Kuehneromyces mutabilis",
            "status_text": " (Edible)",
            "badge_label": "EDIBLE",
            "accent": PALETTE["green"],
            "hint": "Diagnostic cue: distinct dark scales on lower stem below the ring",
        },
        {
            "x": margin_x + img_w + gutter,
            "img_path": IMAGE_ROOT / RIGHT_PATH,
            "meta": right_meta,
            "species_name": "Galerina marginata",
            "status_text": " (Deadly)",
            "badge_label": "DEADLY",
            "accent": PALETTE["red"],
            "hint": "Diagnostic cue: smooth/fibrillose stem; contains lethal amatoxins",
        },
    ]

    for item in items:
        ix = int(item["x"])
        iy = header_h

        # Photo directly on #F7F6F2 with 6 px rounded corners and 1 px #DCD9CF border
        with Image.open(Path(str(item["img_path"]))) as src_img:
            rounded = _rounded_image(src_img, img_w, img_h, radius=6)
            canvas.alpha_composite(rounded, dest=(ix, iy))

        draw.rounded_rectangle(
            (ix, iy, ix + img_w, iy + img_h),
            radius=6,
            outline=PALETTE["border"],
            width=1,
        )

        # Clean flat typography directly below photo
        tx = ix
        ty = iy + img_h + 22
        sp_name = str(item["species_name"])
        st_text = str(item["status_text"])
        accent = str(item["accent"])

        draw.text((tx, ty), sp_name, font=f_species_italic, fill=PALETTE["ink"])
        bbox_sp = draw.textbbox((tx, ty), sp_name, font=f_species_italic)
        draw.text((bbox_sp[2] + 2, ty + 1), st_text, font=f_species_status, fill=accent)

        # Lightweight outline pill (transparent bg, 1 px border)
        badge_txt = str(item["badge_label"])
        b_box = draw.textbbox((0, 0), badge_txt, font=f_badge)
        bw = (b_box[2] - b_box[0]) + 20
        bh = 24
        bx = ix + img_w - bw
        by = ty + 2
        draw.rounded_rectangle(
            (bx, by, bx + bw, by + bh),
            radius=5,
            fill=None,
            outline=accent,
            width=1,
        )
        draw.text(
            (bx + 10 - b_box[0], by + (bh - (b_box[3] - b_box[1])) / 2 - b_box[1]),
            badge_txt,
            font=f_badge,
            fill=accent,
        )

        # Diagnostic cue & attribution
        draw.text((tx, ty + 38), str(item["hint"]), font=f_hint, fill=PALETTE["ink"])

        m = item["meta"]
        obs_credit = m["observer"] or m["author"]
        attr_line = f"Source: iNaturalist #{m['obs_id']} (observer: {obs_credit})  ·  License: {m['license']} (Public Domain)"
        draw.text((tx, ty + 68), attr_line, font=f_attr, fill=PALETTE["muted"])

    OUT_JPG.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(OUT_JPG, format="JPEG", quality=85, optimize=True, progressive=True, dpi=(300, 300))
    print(f"Saved flat editorial look-alike JPG: {OUT_JPG} ({canvas_w}x{canvas_h} px, {OUT_JPG.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
