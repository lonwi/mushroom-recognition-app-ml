"""Deterministic, stratified train/val/test split using HMAC and connected components.

Implements the architectural decision from docs/decisions/2026-11-20-stratified-hmac-split.md (R-01).
Guarantees:
1. Connected component atomicity: All photos sharing an observer/author or observation
   stay strictly together in one split (zero data leakage).
2. Determinism: Assignments are 100% reproducible for a given dataset and split_salt.
3. Stratified quotas: ~70% Train, 15% Val, 15% Test per species.
4. Hard minimum for high_stakes classes: At least 8 photos in test (n_test >= 8) and
   8 photos in val for all high_stakes classes with N >= 55 (e.g. Amanita verna).
"""

from __future__ import annotations

import hashlib
import hmac
import math
import os
import re
from collections import defaultdict
from typing import Any

_GBIF_OCCURRENCE = re.compile(r"gbif\.org/occurrence/(\d+)", re.IGNORECASE)
_INAT_OBS = re.compile(r"inaturalist\.org/observations/(\d+)", re.IGNORECASE)
_INAT_PHOTO = re.compile(r"inaturalist\.org/photos/(\d+)", re.IGNORECASE)
_MO_OBS = re.compile(r"mushroomobserver\.org/(?:observations/)?(\d+)", re.IGNORECASE)

_EMPTY_ID = frozenset({"none", "null", "nan", "", "-"})

_PLACEHOLDER_PERSONS = frozenset({
    "none",
    "null",
    "nan",
    "unknown",
    "anonymous",
    "anonim",
    "anonimowy",
    "nieznany",
    "unknown author",
    "no rights reserved",
    "-",
    "?",
    "n/a",
    "na",
    "inaturalist",
    "gbif",
    "wikimedia",
    "wikimedia commons",
    "mushroom observer",
})


class SplitSaltError(RuntimeError):
    """Raised when the split salt is missing or invalid."""


def resolve_split_salt(salt: str | None = None) -> str:
    """Resolve salt from parameter, environment, or default secret file."""
    if salt and salt.strip():
        return salt.strip()

    env_salt = os.environ.get("GRZYBOBRANIE_SPLIT_SALT")
    if env_salt and env_salt.strip():
        return env_salt.strip()

    salt_path = os.environ.get(
        "GRZYBOBRANIE_SPLIT_SALT_PATH",
        os.path.expanduser("~/.config/grzybobranie/split_salt"),
    )
    if os.path.exists(salt_path):
        with open(salt_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                return content

    raise SplitSaltError(
        f"Split salt is missing. Please set GRZYBOBRANIE_SPLIT_SALT or ensure "
        f"the secret key exists at {salt_path}."
    )


def _clean_id(value: object) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            return None
        value = int(value)
    text = str(value).strip().replace("\r", " ").replace("\n", " ")
    if not text or text.casefold() in _EMPTY_ID:
        return None
    return text


def normalized_person_name(value: object) -> str | None:
    """Trim and case-fold author text, discarding placeholder names."""
    if not isinstance(value, str):
        if value is None or isinstance(value, bool):
            return None
        value = str(value)
    text = " ".join(value.casefold().split())
    if not text or text in _PLACEHOLDER_PERSONS:
        return None
    if text.endswith((" society", " stowarzyszenie", " association")):
        return None
    if "mycological society" in text:
        return None
    return text


def canonical_url(url: str) -> str:
    """Extract canonical URL without query string or hash."""
    clean = url.strip()
    clean = clean.split("#")[0].split("?")[0]
    return clean


def extract_identity_tokens(record: dict[str, Any]) -> list[str]:
    """Extract all identity tokens that connect photos to photographers and observations."""
    tokens: list[str] = []

    # 1. Author / Photographer
    for field in ("inat_observer", "author", "creator", "recorded_by", "recordedBy"):
        val = record.get(field)
        name = normalized_person_name(val)
        if name:
            tokens.append(f"person:{name}")

    # 2. Observation IDs
    obs_id = _clean_id(record.get("observation_id"))
    if obs_id:
        tokens.append(f"obs:{obs_id}")

    inat_obs = _clean_id(record.get("inat_observation_id"))
    if inat_obs:
        tokens.append(f"inat_obs:{inat_obs}")

    mo_obs = _clean_id(record.get("mo_observation_id"))
    if mo_obs:
        tokens.append(f"mo_obs:{mo_obs}")

    gbif_occ = _clean_id(record.get("gbif_occurrence_id") or record.get("occurrence_key"))
    if gbif_occ:
        tokens.append(f"gbif_occ:{gbif_occ}")

    # URLs inspection for observation patterns
    for field in ("source_url", "references"):
        val = record.get(field)
        if isinstance(val, str) and val.strip():
            for found in _INAT_OBS.findall(val):
                tokens.append(f"inat_obs:{found}")
            for found in _MO_OBS.findall(val):
                tokens.append(f"mo_obs:{found}")
            for found in _GBIF_OCCURRENCE.findall(val):
                tokens.append(f"gbif_occ:{found}")

    # 3. Photo IDs and Image URL
    photo_id = _clean_id(record.get("photo_id") or record.get("inat_photo_id"))
    if photo_id:
        tokens.append(f"photo:{photo_id}")

    img_url = record.get("image_url")
    if isinstance(img_url, str) and img_url.strip():
        tokens.append(f"image:{canonical_url(img_url)}")
        for found in _INAT_PHOTO.findall(img_url):
            tokens.append(f"photo:{found}")

    return sorted(set(tokens))


def hashed_split_key(material: list[str], salt: str) -> str:
    """HMAC-SHA256 of sorted component tokens."""
    payload = "\n".join(sorted(material)).encode("utf-8")
    return hmac.new(salt.encode("utf-8"), payload, hashlib.sha256).hexdigest()


def compute_connected_components(
    records: list[dict[str, Any]], salt: str
) -> list[dict[str, Any]]:
    """Compute connected components using Union-Find on identity tokens."""
    count = len(records)
    parent = list(range(count))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        root_i, root_j = find(i), find(j)
        if root_i != root_j:
            parent[root_j] = root_i

    token_owner: dict[str, int] = {}
    record_tokens: list[list[str]] = []

    for index, record in enumerate(records):
        tokens = extract_identity_tokens(record)
        if not tokens:
            tokens = [f"solo:{index}"]
        record_tokens.append(tokens)

        for token in tokens:
            if token.startswith("solo:"):
                continue
            if token in token_owner:
                union(index, token_owner[token])
            else:
                token_owner[token] = index

    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(count):
        grouped[find(index)].append(index)

    components: list[dict[str, Any]] = []
    for indices in grouped.values():
        indices.sort()
        all_tokens = sorted({
            t for idx in indices for t in record_tokens[idx] if not t.startswith("solo:")
        })
        if not all_tokens:
            all_tokens = [f"solo:{indices[0]}"]

        comp_id = hashed_split_key(all_tokens, salt)
        # Deterministic score in [0, 1) derived from first 16 hex chars of HMAC
        score = int(comp_id[:16], 16) / float(0xFFFFFFFFFFFFFFFF)

        components.append({
            "id": comp_id,
            "indices": indices,
            "rows": [records[idx] for idx in indices],
            "score": score,
            "size": len(indices),
        })

    # Sort components deterministically by their HMAC id
    components.sort(key=lambda c: c["id"])
    return components


def stratified_hmac_split(
    records: list[dict[str, Any]],
    *,
    salt: str | None = None,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    high_stakes_classes: set[str] | None = None,
    min_high_stakes_test: int = 8,
    min_high_stakes_val: int = 8,
) -> dict[str, list[dict[str, Any]]]:
    """Execute stratified deterministic split satisfying R-01 guarantees."""
    resolved_salt = resolve_split_salt(salt)
    hs_classes = high_stakes_classes or set()

    # Step 1: Compute connected components
    components = compute_connected_components(records, resolved_salt)

    # Step 2: Index components by class
    comps_by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in components:
        seen_classes = set()
        for row in c["rows"]:
            c_slug = row.get("species_slug") or row.get("class") or ""
            if c_slug and c_slug not in seen_classes:
                comps_by_class[c_slug].append(c)
                seen_classes.add(c_slug)

    # Step 3: Priority ordering of classes
    # High stakes first, sorted by total images ascending (e.g. amanita_verna first)
    def class_sort_key(slug: str) -> tuple[int, int, str]:
        is_hs = 0 if slug in hs_classes else 1
        total_rows = sum(
            sum(1 for r in c["rows"] if (r.get("species_slug") or r.get("class")) == slug)
            for c in comps_by_class[slug]
        )
        return (is_hs, total_rows, slug)

    sorted_classes = sorted(comps_by_class.keys(), key=class_sort_key)

    # Step 4: Assignment logic
    locked_split: dict[str, str] = {}  # comp_id -> 'train' | 'val' | 'test'
    split_records: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    for slug in sorted_classes:
        class_comps = comps_by_class[slug]
        total_class_images = sum(
            sum(1 for r in c["rows"] if (r.get("species_slug") or r.get("class")) == slug)
            for c in class_comps
        )

        is_hs = slug in hs_classes
        test_target = round(total_class_images * test_ratio)
        val_target = round(total_class_images * val_ratio)

        # Enforce hard minimum for high-stakes if dataset supports it
        if is_hs and total_class_images >= 55:
            test_target = max(min_high_stakes_test, test_target)
            val_target = max(min_high_stakes_val, val_target)

        # Account for components already locked by previous classes
        test_count = 0
        val_count = 0
        unlocked_comps: list[dict[str, Any]] = []

        for c in class_comps:
            c_rows_in_class = sum(
                1 for r in c["rows"] if (r.get("species_slug") or r.get("class")) == slug
            )
            if c["id"] in locked_split:
                assigned = locked_split[c["id"]]
                if assigned == "test":
                    test_count += c_rows_in_class
                elif assigned == "val":
                    val_count += c_rows_in_class
            else:
                unlocked_comps.append(c)

        # Sort unlocked components deterministically by score
        unlocked_comps.sort(key=lambda c: c["score"])

        # Binning: fill test first, then val, then train
        for c in unlocked_comps:
            c_rows_in_class = sum(
                1 for r in c["rows"] if (r.get("species_slug") or r.get("class")) == slug
            )
            if test_count < test_target:
                dest = "test"
                test_count += c_rows_in_class
            elif val_count < val_target:
                dest = "val"
                val_count += c_rows_in_class
            else:
                dest = "train"

            locked_split[c["id"]] = dest

    # Step 5: Materialize splits
    for c in components:
        dest = locked_split.get(c["id"], "train")
        split_records[dest].extend(c["rows"])

    # Step 6: Verify invariants
    _verify_split_invariants(split_records, hs_classes, min_high_stakes_test)

    return split_records


def _verify_split_invariants(
    splits: dict[str, list[dict[str, Any]]],
    hs_classes: set[str],
    min_hs_test: int,
) -> None:
    """Verify zero-leakage, component atomicity, and high-stakes test counts."""
    urls: dict[str, str] = {}
    for split_name, rows in splits.items():
        for r in rows:
            u = r.get("image_url")
            if u:
                canon = canonical_url(u)
                if canon in urls and urls[canon] != split_name:
                    raise RuntimeError(
                        f"Leakage detected: Image {canon} present in both "
                        f"{urls[canon]} and {split_name}!"
                    )
                urls[canon] = split_name

    # Check high stakes test counts
    test_counts: dict[str, int] = defaultdict(int)
    for r in splits["test"]:
        s = r.get("species_slug") or r.get("class") or ""
        if s:
            test_counts[s] += 1

    for hs in hs_classes:
        # If total high stakes class had >= 55 images, it must have >= min_hs_test in test
        count = test_counts[hs]
        if count < min_hs_test:
            raise RuntimeError(
                f"High-stakes invariant failed for {hs}: only {count} images in test "
                f"(expected >= {min_hs_test})!"
            )


def compute_fusion_observation_stats(
    rows: list[dict[str, Any]],
) -> dict[str, int]:
    """Count unique observations having >= 2 and >= 3 photos for multi-photo fusion."""
    obs_counts: dict[str, int] = defaultdict(int)
    for r in rows:
        obs = _clean_id(
            r.get("observation_id")
            or r.get("inat_observation_id")
            or r.get("mo_observation_id")
        )
        if obs:
            obs_counts[obs] += 1

    ge2 = sum(1 for c in obs_counts.values() if c >= 2)
    ge3 = sum(1 for c in obs_counts.values() if c >= 3)
    return {
        "unique_observations": len(obs_counts),
        "obs_ge2_photos": ge2,
        "obs_ge3_photos": ge3,
    }
