"""Regenerate P1 fixtures and recompute expected values within tolerance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from grzybobranie_ml.make_p1_fixtures import (
    GENERATOR,
    TOLERANCE,
    write_fixtures,
)
from grzybobranie_ml.preprocess import preprocess

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "fixtures" / "p1"
PARITY_PATH = FIXTURE_DIR / "parity.json"
SUMS_PATH = FIXTURE_DIR / "SHA256SUMS"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_close_data(got: object, expected: object, path: str) -> None:
    if isinstance(expected, bool) or isinstance(got, bool):
        assert got == expected, f"{path}: {got!r} != {expected!r}"
        return
    if isinstance(expected, (int, float)) and isinstance(got, (int, float)):
        assert abs(float(got) - float(expected)) <= TOLERANCE, (
            f"{path}: {got} vs {expected} (tol={TOLERANCE})"
        )
        return
    if expected is None or got is None:
        assert got is expected, f"{path}: {got!r} vs {expected!r}"
        return
    if isinstance(expected, dict):
        assert isinstance(got, dict), f"{path}: expected object, got {type(got)}"
        assert set(got) == set(expected), (
            f"{path}: keys {sorted(got)} != {sorted(expected)}"
        )
        for key in expected:
            _assert_close_data(got[key], expected[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        assert isinstance(got, list), f"{path}: expected list, got {type(got)}"
        assert len(got) == len(expected), (
            f"{path}: length {len(got)} != {len(expected)}"
        )
        for i, (item_got, item_expected) in enumerate(zip(got, expected)):
            _assert_close_data(item_got, item_expected, f"{path}[{i}]")
        return
    assert got == expected, f"{path}: {got!r} vs {expected!r}"


def test_committed_fixtures_match_fresh_generation(tmp_path: Path) -> None:
    write_fixtures(tmp_path)
    committed = FIXTURE_DIR / "parity.json"
    generated = tmp_path / "parity.json"
    assert committed.is_file(), "missing committed fixtures/p1/parity.json"
    _assert_close_data(_load(generated), _load(committed), "parity.json")
    # Images must be byte-identical: they are the shared visual inputs.
    for name in sorted(p.name for p in (FIXTURE_DIR / "images").glob("*.png")):
        left = (FIXTURE_DIR / "images" / name).read_bytes()
        right = (tmp_path / "images" / name).read_bytes()
        assert left == right, f"images/{name} is stale; re-run make_p1_fixtures"


def test_sha256sums_lists_parity_and_images() -> None:
    listed: dict[str, str] = {}
    for line in SUMS_PATH.read_text(encoding="utf-8").splitlines():
        digest, name = line.split()
        listed[name] = digest
    assert "parity.json" in listed
    assert listed["parity.json"] == _sha256(PARITY_PATH)
    images = sorted(p.name for p in (FIXTURE_DIR / "images").glob("*.png"))
    assert images, "no committed fixture images"
    for name in images:
        rel = f"images/{name}"
        assert listed[rel] == _sha256(FIXTURE_DIR / rel)


def test_fixture_metadata() -> None:
    payload = _load(PARITY_PATH)
    assert payload["schemaVersion"] == "1.0.1"
    assert payload["generator"] == GENERATOR
    assert payload["format"] == "ml/fixtures/p1/parity.json"
    assert payload["tolerance"] == TOLERANCE
    assert payload["synthetic"] is True
    assert payload["cases"]


def test_recompute_cases_within_tolerance() -> None:
    payload = _load(PARITY_PATH)
    for case in payload["cases"]:
        path = FIXTURE_DIR / case["image"]
        assert _sha256(path) == case["imageSha256"], case["id"]
        got = preprocess(path, case["preprocessing"], layout=case["layout"])
        for kind in ("uint8", "float"):
            arr = got[kind]
            exp = case["expected"][kind]
            assert list(arr.shape) == exp["shape"], f"{case['id']} {kind} shape"
            assert str(arr.dtype) == exp["dtype"], f"{case['id']} {kind} dtype"
            assert (
                hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()
                == exp["sha256"]
            ), (
                f"{case['id']} {kind} sha256 changed; regenerate fixtures if the math change is intended"
            )
            flat = np.asarray(arr, dtype=np.float64)
            assert abs(float(np.min(flat)) - exp["min"]) <= TOLERANCE, case["id"]
            assert abs(float(np.max(flat)) - exp["max"]) <= TOLERANCE, case["id"]
            assert abs(float(np.mean(flat)) - exp["mean"]) <= TOLERANCE, case["id"]
            assert abs(float(np.std(flat, ddof=0)) - exp["std"]) <= TOLERANCE, case[
                "id"
            ]


def test_documented_coverage() -> None:
    payload = _load(PARITY_PATH)
    ids = {c["id"] for c in payload["cases"]}
    assert any(i.startswith("baseline_224_") for i in ids)
    assert any(i.startswith("baseline_256_") for i in ids)
    assert any("wide_640x320" in i for i in ids)
    assert any("exif_orient6" in i for i in ids)
    assert any("rgba" in i for i in ids)
    assert any("gray" in i for i in ids)
    # At least three baseline images exist at both sizes.
    for n in (224, 256):
        assert len([i for i in ids if i.startswith(f"baseline_{n}_")]) >= 3
