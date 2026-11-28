"""Unit tests for CC0 Outlier Exposure curation helpers (grzybobranie_ml.curate_oe)."""

from __future__ import annotations

from pathlib import Path

from grzybobranie_ml.curate_oe import load_blocked_identifiers


def test_load_blocked_identifiers_collects_ood_and_splits(tmp_path: Path) -> None:
    ood_dir = tmp_path / "ood-v1"
    ood_dir.mkdir()
    (ood_dir / "ood_ids.txt").write_text(
        "# header\n"
        "photo_id\tph_ood_1\n"
        "image_url\thttps://static.inaturalist.org/photos/123/medium.jpg\n",
        encoding="utf-8",
    )
    (ood_dir / "ood_nonfungi.csv").write_text(
        "photo_id,inat_observation_id,image_url\n"
        "ph_nf,obs_nf,https://example.com/nf.jpg\n",
        encoding="utf-8",
    )
    test_csv = tmp_path / "test.csv"
    test_csv.write_text(
        "photo_id,sha256,canonical_url\n"
        "ph_test,abc123def,https://example.com/test.jpg\n",
        encoding="utf-8",
    )

    b_ids, b_urls, b_hashes = load_blocked_identifiers(
        ood_ids_path=ood_dir / "ood_ids.txt",
        excluded_split_csvs=(test_csv,),
    )
    assert "ph_ood_1" in b_ids
    assert "ph_nf" in b_ids
    assert "obs_nf" in b_ids
    assert "ph_test" in b_ids
    assert "abc123def" in b_hashes
    assert any("123/medium.jpg" in u for u in b_urls)
    assert any("example.com/nf.jpg" in u for u in b_urls)
    assert any("example.com/test.jpg" in u for u in b_urls)
