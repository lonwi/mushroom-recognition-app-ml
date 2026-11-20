"""Tests for deterministic stratified HMAC splitting and leakage prevention."""

from __future__ import annotations

from grzybobranie_ml.split import (
    compute_connected_components,
    compute_fusion_observation_stats,
    extract_identity_tokens,
    hashed_split_key,
    normalized_person_name,
    stratified_hmac_split,
)

FAKE_SALT = "test_salt_grzybobranie_12345"


def test_normalized_person_name() -> None:
    assert normalized_person_name("  Jan   Kowalski ") == "jan kowalski"
    assert normalized_person_name("Unknown") is None
    assert normalized_person_name("anonymous") is None
    assert normalized_person_name("No Rights Reserved") is None
    assert normalized_person_name("None") is None
    assert normalized_person_name("Polish Mycological Society") is None


def test_extract_identity_tokens() -> None:
    row = {
        "author": "Alice Smith",
        "inat_observation_id": "123456",
        "inat_photo_id": "9999",
        "image_url": "https://example.com/photo.jpg?size=large",
    }
    tokens = extract_identity_tokens(row)
    assert "person:alice smith" in tokens
    assert "inat_obs:123456" in tokens
    assert "photo:9999" in tokens
    assert "image:https://example.com/photo.jpg" in tokens


def test_connected_components_grouping() -> None:
    records = [
        # Group 1: Linked by author "Alice"
        {"id": "1", "author": "Alice", "observation_id": "obs_1"},
        {"id": "2", "author": "Alice", "observation_id": "obs_2"},
        # Group 2: Linked by observation "obs_3" (different author field)
        {"id": "3", "author": "Bob", "observation_id": "obs_3"},
        {"id": "4", "author": "Charlie", "observation_id": "obs_3"},
        # Group 3: Isolated
        {"id": "5", "author": "Dave", "observation_id": "obs_4"},
    ]

    components = compute_connected_components(records, FAKE_SALT)
    assert len(components) == 3

    sizes = sorted(c["size"] for c in components)
    assert sizes == [1, 2, 2]


def test_hmac_determinism() -> None:
    tokens = ["inat_obs:123", "person:alice"]
    h1 = hashed_split_key(tokens, FAKE_SALT)
    h2 = hashed_split_key(tokens, FAKE_SALT)
    assert h1 == h2
    assert len(h1) == 64

    # Order of input tokens should not change hash
    h3 = hashed_split_key(list(reversed(tokens)), FAKE_SALT)
    assert h1 == h3


def test_stratified_split_with_high_stakes_minimum() -> None:
    # Create 60 records for a high_stakes species (obs size 1 to 2)
    records = []
    for i in range(60):
        records.append({
            "species_slug": "toxic_species",
            "author": f"Author_{i // 2}",
            "observation_id": f"obs_{i // 2}",
            "image_url": f"https://example.com/toxic_{i}.jpg",
        })

    splits = stratified_hmac_split(
        records,
        salt=FAKE_SALT,
        high_stakes_classes={"toxic_species"},
        min_high_stakes_test=8,
        min_high_stakes_val=8,
    )

    assert len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == 60
    assert len(splits["test"]) >= 8
    assert len(splits["val"]) >= 8

    # Verify no URL overlap
    train_urls = {r["image_url"] for r in splits["train"]}
    val_urls = {r["image_url"] for r in splits["val"]}
    test_urls = {r["image_url"] for r in splits["test"]}

    assert len(train_urls.intersection(val_urls)) == 0
    assert len(train_urls.intersection(test_urls)) == 0
    assert len(val_urls.intersection(test_urls)) == 0


def test_fusion_observation_stats() -> None:
    rows = [
        {"observation_id": "obs_1"},
        {"observation_id": "obs_1"},
        {"observation_id": "obs_1"},  # obs_1 has 3 photos
        {"observation_id": "obs_2"},
        {"observation_id": "obs_2"},  # obs_2 has 2 photos
        {"observation_id": "obs_3"},  # obs_3 has 1 photo
    ]
    stats = compute_fusion_observation_stats(rows)
    assert stats["unique_observations"] == 3
    assert stats["obs_ge2_photos"] == 2  # obs_1, obs_2
    assert stats["obs_ge3_photos"] == 1  # obs_1
