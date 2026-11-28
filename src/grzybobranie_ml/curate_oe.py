"""Curate an external, strictly CC0 Outlier Exposure dataset of forest & natural negatives.

Module: grzybobranie_ml.curate_oe
CLI: python -m grzybobranie_ml.curate_oe [options]

Invariants:
- Saved strictly outside data/ood-v1/ (default: data/oe_cc0_v2/).
- Every row strictly CC0 (`cc0` / `cc0-1.0`).
- 100% disjoint from data/ood-v1/ (both ood_calib and ood_test), test.csv, val.csv, and train.csv
  across photo/observation IDs, canonical URLs, and image SHA-256 digests.
- Focuses on hard forest/nature negatives: moss, bark, leaf litter, pine cones, ferns,
  non-mushroom lichens, soil, stones, twigs, forest insects/snails, and outdoor surfaces.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

from grzybobranie_ml.split import canonical_url
from grzybobranie_ml.train import (
    _is_strict_cc0_license,
    validate_oe_manifest_disjointness,
)

logger = logging.getLogger("grzybobranie_ml.curate_oe")

DEFAULT_OUTPUT_DIR = Path("data/oe_cc0_v2")
DEFAULT_OOD_IDS = Path("data/ood-v1/ood_ids.txt")
DEFAULT_EXCLUDED_SPLITS = (
    Path("data/splits/test.csv"),
    Path("data/splits/val.csv"),
    Path("data/splits/train.csv"),
)
USER_AGENT = "GrzybobranieML-OECurator/0.1 (CC0 forest background negatives)"

# Curated non-mushroom forest/nature taxa on iNaturalist (strictly CC0 photos)
INAT_CC0_QUERIES: tuple[dict[str, Any], ...] = (
    {"category": "moss", "taxon_id": 311249, "max_items": 350},        # Bryophyta (mosses)
    {"category": "fern", "taxon_id": 121943, "max_items": 220},        # Polypodiopsida (ferns)
    {"category": "conifer", "taxon_id": 136329, "max_items": 260},     # Pinophyta (conifers, needles, cones)
    {"category": "deciduous", "taxon_id": 47851, "max_items": 200},    # Fagaceae (oak, beech leaves/bark)
    {"category": "birch", "taxon_id": 49154, "max_items": 120},        # Betula (bark, leaves)
    {"category": "understory_oxalis", "taxon_id": 47758, "max_items": 90},   # Oxalis
    {"category": "understory_vaccinium", "taxon_id": 48353, "max_items": 90}, # Vaccinium
    {"category": "understory_hedera", "taxon_id": 55857, "max_items": 90},    # Hedera
    {"category": "lichen_cladonia", "taxon_id": 54743, "max_items": 100, "allow_lichen": True},
    {"category": "lichen_parmelia", "taxon_id": 55495, "max_items": 70, "allow_lichen": True},
    {"category": "lichen_xanthoria", "taxon_id": 55517, "max_items": 60, "allow_lichen": True},
    {"category": "forest_beetle", "taxon_id": 47208, "max_items": 120},  # Coleoptera
    {"category": "forest_snail", "taxon_id": 47114, "max_items": 100},   # Gastropoda
)

COMMONS_CC0_QUERIES: tuple[dict[str, Any], ...] = (
    {"category": "forest_litter", "query": "forest floor leaf litter", "max_items": 90},
    {"category": "bark_texture", "query": "tree bark close up", "max_items": 90},
    {"category": "pine_cones", "query": "pine cone forest floor", "max_items": 80},
    {"category": "moss_stone", "query": "moss rock stone forest", "max_items": 80},
    {"category": "soil_gravel", "query": "soil gravel pebbles ground", "max_items": 80},
    {"category": "autumn_leaves", "query": "fallen autumn leaves ground", "max_items": 80},
    {"category": "wood_twigs", "query": "wood log twigs forest", "max_items": 80},
)


def load_blocked_identifiers(
    ood_ids_path: str | Path = DEFAULT_OOD_IDS,
    excluded_split_csvs: Sequence[str | Path] = DEFAULT_EXCLUDED_SPLITS,
) -> tuple[set[str], set[str], set[str]]:
    """Load blocked IDs, canonical URLs, and SHA-256 digests from OOD-v1 and train/val/test."""
    blocked_ids: set[str] = set()
    blocked_urls: set[str] = set()
    blocked_hashes: set[str] = set()

    id_keys = (
        "photo_id",
        "inat_photo_id",
        "observation_id",
        "inat_observation_id",
        "mo_observation_id",
        "mo_image_id",
        "gbif_occurrence_id",
        "commons_file",
    )
    url_keys = ("canonical_url", "image_url", "source_url")
    hash_keys = ("original_sha256", "sha256")

    ood_p = Path(ood_ids_path)
    extra_csvs: list[Path] = []
    if ood_p.exists():
        with open(ood_p, encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#") or "\t" not in s:
                    continue
                kind, val = s.split("\t", 1)
                val = val.strip()
                if kind in url_keys:
                    blocked_urls.add(canonical_url(val))
                else:
                    blocked_ids.add(val)
        for sib_name in ("ood_fungi.csv", "ood_nonfungi.csv"):
            sib = ood_p.parent / sib_name
            if sib.exists():
                extra_csvs.append(sib)

    for sp in [*excluded_split_csvs, *extra_csvs]:
        p = Path(sp)
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                for k in id_keys:
                    v = (row.get(k) or "").strip()
                    if v:
                        blocked_ids.add(v)
                for uk in url_keys:
                    u = (row.get(uk) or "").strip()
                    if u:
                        blocked_urls.add(canonical_url(u))
                for hk in hash_keys:
                    hv = (row.get(hk) or "").strip().lower()
                    if hv:
                        blocked_hashes.add(hv)

    return blocked_ids, blocked_urls, blocked_hashes


def _fetch_json(url: str, timeout: float = 15.0) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_inat_cc0_candidates(
    queries: tuple[dict[str, Any], ...] = INAT_CC0_QUERIES,
    blocked_ids: set[str] | None = None,
    blocked_urls: set[str] | None = None,
    per_page: int = 100,
    max_pages_per_query: int = 4,
) -> list[dict[str, str]]:
    """Fetch CC0 forest/nature observation photo candidates from iNaturalist API v1."""
    b_ids = blocked_ids or set()
    b_urls = blocked_urls or set()
    candidates: list[dict[str, str]] = []
    seen_photo_ids: set[str] = set()
    user_counts: dict[str, int] = {}

    for spec in queries:
        taxon_id = int(spec["taxon_id"])
        category = str(spec["category"])
        max_items = int(spec["max_items"])
        allow_lichen = bool(spec.get("allow_lichen", False))
        collected = 0

        for page in range(1, max_pages_per_query + 1):
            if collected >= max_items:
                break
            params = urllib.parse.urlencode({
                "taxon_id": taxon_id,
                "photo_license": "cc0",
                "quality_grade": "research",
                "captive": "false",
                "order_by": "votes",
                "per_page": per_page,
                "page": page,
            })
            url = f"https://api.inaturalist.org/v1/observations?{params}"
            try:
                data = _fetch_json(url)
            except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
                logger.warning("iNat fetch failed for %s page %d: %s", category, page, exc)
                break

            results = data.get("results") or []
            if not results:
                break

            for obs in results:
                if collected >= max_items:
                    break
                obs_id = str(obs.get("id") or "")
                if not obs_id or obs_id in b_ids:
                    continue
                taxon = obs.get("taxon") or {}
                iconic = str(taxon.get("iconic_taxon_name") or "")
                if iconic == "Fungi" and not allow_lichen:
                    continue
                user = (obs.get("user") or {}).get("login") or "unknown"
                if user_counts.get(user, 0) >= 20:
                    continue

                obs_taken = 0
                for ph in obs.get("photos") or []:
                    if collected >= max_items or obs_taken >= 2:
                        break
                    if str(ph.get("license_code") or "").lower() != "cc0":
                        continue
                    pid = str(ph.get("id") or "")
                    if not pid or pid in b_ids or pid in seen_photo_ids:
                        continue
                    raw_url = str(ph.get("url") or "")
                    if not raw_url:
                        continue
                    med_url = raw_url.replace("/square.", "/medium.")
                    if med_url == raw_url:
                        med_url = raw_url.replace("square", "medium")
                    can = canonical_url(med_url)
                    if can in b_urls or canonical_url(raw_url) in b_urls:
                        continue

                    seen_photo_ids.add(pid)
                    b_urls.add(can)
                    user_counts[user] = user_counts.get(user, 0) + 1
                    obs_taken += 1
                    collected += 1
                    candidates.append({
                        "source": "inat",
                        "category": category,
                        "photo_id": f"inat_{pid}",
                        "inat_photo_id": pid,
                        "observation_id": f"inat_obs_{obs_id}",
                        "inat_observation_id": obs_id,
                        "commons_file": "",
                        "taxon_name": str(taxon.get("name") or category),
                        "license": "CC0",
                        "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
                        "image_url": med_url,
                        "source_url": str(obs.get("uri") or f"https://www.inaturalist.org/observations/{obs_id}"),
                        "author": user,
                    })
            time.sleep(0.2)

        logger.info("iNat CC0 [%s]: collected %d candidates", category, collected)
    return candidates


def fetch_commons_cc0_candidates(
    queries: tuple[dict[str, Any], ...] = COMMONS_CC0_QUERIES,
    blocked_ids: set[str] | None = None,
    blocked_urls: set[str] | None = None,
) -> list[dict[str, str]]:
    """Fetch strictly CC0 forest floor / stone / bark / surface photos from Wikimedia Commons."""
    b_ids = blocked_ids or set()
    b_urls = blocked_urls or set()
    candidates: list[dict[str, str]] = []
    seen_titles: set[str] = set()

    for spec in queries:
        category = str(spec["category"])
        query = str(spec["query"])
        max_items = int(spec["max_items"])
        collected = 0

        params = urllib.parse.urlencode({
            "action": "query",
            "generator": "search",
            "gsrsearch": f'filetype:bitmap {query} "CC0"',
            "gsrnamespace": 6,
            "gsrlimit": 50,
            "prop": "imageinfo",
            "iiprop": "url|extmetadata|mime|sha1|size",
            "iiurlwidth": 768,
            "format": "json",
        })
        url = f"https://commons.wikimedia.org/w/api.php?{params}"
        try:
            data = _fetch_json(url)
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            logger.warning("Commons fetch failed for %r: %s", query, exc)
            continue

        pages = (data.get("query") or {}).get("pages") or {}
        for page in pages.values():
            if collected >= max_items:
                break
            title = str(page.get("title") or "")
            if not title or title in seen_titles or title in b_ids:
                continue
            lower_title = title.lower()
            if any(bad in lower_title for bad in ("mushroom", "fungus", "fungi", "amanita", "boletus", "diagram", "illustration")):
                continue
            ii_list = page.get("imageinfo") or []
            if not ii_list:
                continue
            ii = ii_list[0]
            mime = str(ii.get("mime") or "")
            if mime not in {"image/jpeg", "image/png"}:
                continue
            ext = ii.get("extmetadata") or {}
            short_lic = str((ext.get("LicenseShortName") or {}).get("value") or "")
            lic_url = str((ext.get("LicenseUrl") or {}).get("value") or "")
            if not _is_strict_cc0_license(short_lic, lic_url):
                continue

            thumb_url = str(ii.get("thumburl") or ii.get("url") or "")
            if not thumb_url:
                continue
            can = canonical_url(thumb_url)
            if can in b_urls:
                continue

            seen_titles.add(title)
            b_urls.add(can)
            collected += 1
            page_id = str(page.get("pageid") or hashlib.sha256(title.encode()).hexdigest()[:10])
            candidates.append({
                "source": "commons",
                "category": category,
                "photo_id": f"commons_{page_id}",
                "inat_photo_id": "",
                "observation_id": f"commons_obs_{page_id}",
                "inat_observation_id": "",
                "commons_file": title,
                "taxon_name": category,
                "license": "CC0",
                "license_url": lic_url or "https://creativecommons.org/publicdomain/zero/1.0/",
                "image_url": thumb_url,
                "source_url": str(ii.get("descriptionurl") or thumb_url),
                "author": "commons_cc0",
            })
        logger.info("Commons CC0 [%s]: collected %d candidates", category, collected)
    return candidates


def _download_and_normalize_image(
    candidate: dict[str, str],
    images_dir: Path,
    max_side: int = 768,
) -> dict[str, str] | None:
    url = candidate["image_url"]
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=15.0) as resp:
            raw_bytes = resp.read()
        orig_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        with Image.open(io.BytesIO(raw_bytes)) as img:
            oriented = ImageOps.exif_transpose(img)
            rgb = (oriented if oriented is not None else img).convert("RGB")
            if min(rgb.size) < 128:
                return None
            if max(rgb.size) > max_side:
                rgb.thumbnail((max_side, max_side), Image.Resampling.BILINEAR)
            rel_path = f"images/{candidate['category']}/{candidate['photo_id']}.jpg"
            out_path = images_dir / candidate["category"] / f"{candidate['photo_id']}.jpg"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            rgb.save(out_path, format="JPEG", quality=92)
    except (urllib.error.URLError, TimeoutError, OSError, UnidentifiedImageError, ValueError):
        return None

    row = dict(candidate)
    row["original_sha256"] = orig_sha256
    row["sha256"] = hashlib.sha256(out_path.read_bytes()).hexdigest()
    row["local_path"] = rel_path
    return row


def curate_oe_dataset(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    ood_ids_path: str | Path = DEFAULT_OOD_IDS,
    excluded_split_csvs: tuple[str | Path, ...] = DEFAULT_EXCLUDED_SPLITS,
    max_workers: int = 12,
    include_commons: bool = True,
) -> dict[str, Any]:
    """Download, normalize, deduplicate, and validate external CC0 forest OE dataset."""
    out_dir = Path(output_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    blocked_ids, blocked_urls, blocked_hashes = load_blocked_identifiers(
        ood_ids_path=ood_ids_path,
        excluded_split_csvs=excluded_split_csvs,
    )

    candidates = fetch_inat_cc0_candidates(blocked_ids=blocked_ids, blocked_urls=blocked_urls)
    if include_commons:
        candidates.extend(
            fetch_commons_cc0_candidates(blocked_ids=blocked_ids, blocked_urls=blocked_urls)
        )

    logger.info("Fetched %d CC0 OE candidates; downloading & verifying...", len(candidates))
    saved_rows: list[dict[str, str]] = []
    seen_hashes: set[str] = set(blocked_hashes)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(_download_and_normalize_image, cand, images_dir) for cand in candidates
        ]
        for fut in as_completed(futures):
            row = fut.result()
            if row is None:
                continue
            if row["original_sha256"] in seen_hashes or row["sha256"] in seen_hashes:
                continue
            seen_hashes.add(row["original_sha256"])
            seen_hashes.add(row["sha256"])
            saved_rows.append(row)

    saved_rows.sort(key=lambda r: (r["category"], r["photo_id"]))
    manifest_csv = out_dir / "oe_manifest.csv"
    fieldnames = [
        "photo_id",
        "inat_photo_id",
        "observation_id",
        "inat_observation_id",
        "commons_file",
        "source",
        "category",
        "taxon_name",
        "license",
        "license_url",
        "image_url",
        "source_url",
        "author",
        "original_sha256",
        "sha256",
        "local_path",
    ]
    with open(manifest_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(saved_rows)

    validated_count = validate_oe_manifest_disjointness(
        manifest_csv,
        ood_ids_path=ood_ids_path,
        excluded_split_csvs=excluded_split_csvs,
    )
    by_cat: dict[str, int] = {}
    for r in saved_rows:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1

    summary = {
        "manifest_csv": str(manifest_csv),
        "total_images": validated_count,
        "categories": dict(sorted(by_cat.items())),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logger.info("Curated %d strictly CC0 disjoint OE images at %s", validated_count, manifest_csv)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Curate disjoint CC0 forest negatives for OE training.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--ood-ids", type=str, default=str(DEFAULT_OOD_IDS))
    parser.add_argument("--max-workers", type=int, default=12)
    parser.add_argument("--no-commons", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    curate_oe_dataset(
        output_dir=args.output_dir,
        ood_ids_path=args.ood_ids,
        max_workers=args.max_workers,
        include_commons=not args.no_commons,
    )


if __name__ == "__main__":
    main()
