"""Regenerate P2 fixtures and recompute expected values within tolerance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from grzybobranie_ml.make_p2_fixtures import (
    GENERATOR,
    TOLERANCE,
    write_fixtures,
)
from grzybobranie_ml.recognition_math import decide_single, fuse

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "fixtures" / "p2"
DECISION_PATH = FIXTURE_DIR / "decision_cases.json"
FUSION_PATH = FIXTURE_DIR / "fusion_cases.json"
SUMS_PATH = FIXTURE_DIR / "SHA256SUMS"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_close_data(got: object, expected: object, path: str) -> None:
    """Compare fixture JSON semantically, with TOLERANCE on numbers."""
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
    for name in ("decision_cases.json", "fusion_cases.json"):
        committed = FIXTURE_DIR / name
        generated = tmp_path / name
        assert committed.is_file(), f"missing committed fixture {committed}"
        _assert_close_data(_load(generated), _load(committed), name)


def test_sha256sums_lists_both_json_files() -> None:
    lines = SUMS_PATH.read_text(encoding="utf-8").splitlines()
    listed = {}
    for line in lines:
        digest, name = line.split()
        listed[name] = digest
    assert set(listed) == {"decision_cases.json", "fusion_cases.json"}
    assert listed["decision_cases.json"] == _sha256(DECISION_PATH)
    assert listed["fusion_cases.json"] == _sha256(FUSION_PATH)


def _assert_close(
    got: float | None, expected: float | None, case_id: str, field: str
) -> None:
    if expected is None:
        assert got is None, f"{case_id}: expected {field}=null, got {got!r}"
        return
    assert got is not None, f"{case_id}: expected {field}={expected!r}, got null"
    assert abs(got - expected) <= TOLERANCE, (
        f"{case_id}: {field} expected {expected} got {got} (tol={TOLERANCE})"
    )


def _assert_candidates(got: list, expected: list, case_id: str) -> None:
    assert len(got) == len(expected), (
        f"{case_id}: candidate count {len(got)} != {len(expected)}"
    )
    for a, b in zip(got, expected):
        assert a["index"] == b["index"], case_id
        assert a["classId"] == b["classId"], case_id
        assert a["rank"] == b["rank"], case_id
        assert abs(a["probability"] - b["probability"]) <= TOLERANCE, case_id


@pytest.mark.parametrize(
    "path", [DECISION_PATH, FUSION_PATH], ids=["decision", "fusion"]
)
def test_fixture_metadata(path: Path) -> None:
    payload = _load(path)
    assert payload["schemaVersion"] == "1.0.1"
    assert payload["generator"] == GENERATOR
    assert payload["tolerance"] == TOLERANCE
    assert payload["synthetic"] is True
    assert payload["cases"], f"{path.name} has no cases"
    for case in payload["cases"]:
        assert case["synthetic"] is True
        assert "manifest" in case and "logits" in case and "expected" in case


def test_recompute_decision_cases_within_tolerance() -> None:
    payload = _load(DECISION_PATH)
    for case in payload["cases"]:
        got = decide_single(case["logits"], case["manifest"])
        exp = case["expected"]
        assert got["status"] == exp["status"], case["id"]
        _assert_close(got["energy"], exp["energy"], case["id"], "energy")
        _assert_close(got["p_top1"], exp["p_top1"], case["id"], "p_top1")
        _assert_candidates(got["candidates"], exp["candidates"], case["id"])


def test_recompute_fusion_cases_within_tolerance() -> None:
    payload = _load(FUSION_PATH)
    for case in payload["cases"]:
        got = fuse(case["logits"], case["manifest"])
        exp = case["expected"]
        assert got["status"] == exp["status"], case["id"]
        assert got["m"] == exp["m"], case["id"]
        _assert_close(got["energy"], exp["energy"], case["id"], "energy")
        _assert_close(got["p_top1"], exp["p_top1"], case["id"], "p_top1")
        _assert_candidates(got["candidates"], exp["candidates"], case["id"])


def test_documented_boundary_and_coverage_ids() -> None:
    decision_ids = {c["id"] for c in _load(DECISION_PATH)["cases"]}
    fusion_ids = {c["id"] for c in _load(FUSION_PATH)["cases"]}
    assert "boundary_energy_equal" in decision_ids
    assert "boundary_top1_equal" in decision_ids
    assert "temperatures_both_differing" in decision_ids
    assert "numerical_stability_large_logits" in decision_ids
    assert {
        "fusion_m3_all_accepted",
        "fusion_m2_one_rejected_before",
        "fusion_m1_single_accepted",
        "fusion_m0_all_rejected",
    } <= fusion_ids
    # Equality cases really are accepted (strict inequalities).
    by_id = {c["id"]: c for c in _load(DECISION_PATH)["cases"]}
    energy_case = by_id["boundary_energy_equal"]
    top1_case = by_id["boundary_top1_equal"]
    assert energy_case["expected"]["status"] == "ok"
    assert top1_case["expected"]["status"] == "ok"
    assert (
        energy_case["expected"]["energy"]
        == energy_case["manifest"]["ood"]["energyThreshold"]
    )
    assert abs(energy_case["expected"]["energy"] - (-6.0)) <= TOLERANCE
    assert abs(top1_case["expected"]["p_top1"] - 0.40) <= TOLERANCE
