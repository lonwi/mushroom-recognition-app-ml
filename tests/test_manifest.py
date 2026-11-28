"""Unit and integration tests for Stage P7/P9 manifest & golden images pack."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path

import jsonschema
import pytest

from grzybobranie_ml.manifest import (
    PROBLEMATIC_8_EDIBLE,
    build_test_manifest,
    compute_file_sha256,
    select_golden_records,
    validate_manifest_against_schema,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "schemas" / "model-manifest.schema.json"
COVERAGE_PATH = REPO_ROOT / "configs" / "pokrycie-67.csv"
TFLITE_PATH = REPO_ROOT / "artifacts" / "models" / "mobilenetv4_medium_r256_fp16.tflite"
RELEASE_MANIFEST_PATH = REPO_ROOT / "artifacts" / "release_test" / "model-manifest.json"
GOLDEN_IMAGES_PATH = REPO_ROOT / "artifacts" / "p7_parity" / "golden_images.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.skipif(not TFLITE_PATH.exists(), reason="FP16 TFLite artifact not present")
def test_build_test_manifest_validates_against_schema() -> None:
    manifest = build_test_manifest(model_version="0.1.0-rc1-test", channel="test")
    validate_manifest_against_schema(manifest, schema_path=SCHEMA_PATH)

    assert manifest["schemaVersion"] == "1.0.1"
    assert manifest["channel"] == "test"
    assert manifest["shipGates"]["passed"] is False
    assert "rejectClasses" not in manifest

    class_ids = [c["classId"] for c in manifest["classes"]]
    uncovered_ids = [u["speciesId"] for u in manifest["uncoveredSpecies"]]
    assert len(class_ids) == 66
    assert uncovered_ids == ["amanita_verna"]
    assert "amanita_verna" not in class_ids
    assert len(class_ids) + len(uncovered_ids) == 67
    assert all(c["in_model"] is True for c in manifest["classes"])

    assert manifest["preprocessing"]["crop"] == "none_stretch"
    assert manifest["preprocessing"]["inputSize"] == {"width": 256, "height": 256}
    assert manifest["calibration"]["calibrated"] is True
    assert manifest["calibration"]["temperature"] == pytest.approx(0.7848)
    assert "2" in manifest["ood"]["fusion"]
    assert "3" in manifest["ood"]["fusion"]


@pytest.mark.skipif(not TFLITE_PATH.exists(), reason="FP16 TFLite artifact not present")
def test_manifest_guardrails_reject_invalid_configs() -> None:
    with pytest.raises(ValueError, match="channel='test'"):
        build_test_manifest(channel="production")

    base = build_test_manifest(channel="test")

    with_reject = copy.deepcopy(base)
    with_reject["rejectClasses"] = [{"classId": "not_a_mushroom", "reason": "not_a_mushroom", "acceptThreshold": 0.5}]
    with pytest.raises(ValueError, match="rejectClasses"):
        validate_manifest_against_schema(with_reject, schema_path=SCHEMA_PATH)

    with_inactive = copy.deepcopy(base)
    with_inactive["classes"][0]["in_model"] = False
    with pytest.raises(ValueError, match="in_model=true"):
        validate_manifest_against_schema(with_inactive, schema_path=SCHEMA_PATH)

    with_prod_fail = copy.deepcopy(base)
    with_prod_fail["channel"] = "production"
    with pytest.raises(jsonschema.ValidationError):
        validate_manifest_against_schema(with_prod_fail, schema_path=SCHEMA_PATH)

    with_bad_index = copy.deepcopy(base)
    with_bad_index["classes"][1]["index"] = 999
    with pytest.raises(ValueError, match="sequential"):
        validate_manifest_against_schema(with_bad_index, schema_path=SCHEMA_PATH)

    with_species_overlap = copy.deepcopy(base)
    with_species_overlap["classes"][0]["speciesIds"].append("amanita_verna")
    with pytest.raises(ValueError, match="Overlap"):
        validate_manifest_against_schema(with_species_overlap, schema_path=SCHEMA_PATH)


def test_select_golden_records_coverage() -> None:
    with open(COVERAGE_PATH, encoding="utf-8") as f:
        cov_rows = list(csv.DictReader(f))
    all_species = [r["class"] for r in cov_rows]
    hs_slugs = {r["class"] for r in cov_rows if str(r["high_stakes"]).strip().lower() == "true"}
    active = [s for s in all_species if s != "amanita_verna"]

    # Build synthetic val rows with 6 photos per species
    val_rows = []
    for s in all_species:
        for i in range(6):
            val_rows.append({
                "species_slug": s,
                "photo_id": f"{s}_{i}",
                "image_url": f"https://example.com/{s}_{i}.jpg",
            })

    selected = select_golden_records(val_rows, active_classes=active, high_stakes_slugs=hs_slugs, target_total=240)
    assert len(selected) == 240

    selected_species = {r["species_slug"] for r in selected}
    assert "amanita_verna" in selected_species
    assert sum(1 for r in selected if r["species_slug"] == "amanita_verna") == 5
    for hs in hs_slugs:
        if hs != "amanita_verna":
            assert sum(1 for r in selected if r["species_slug"] == hs) >= 4
    for ed in PROBLEMATIC_8_EDIBLE:
        assert sum(1 for r in selected if r["species_slug"] == ed) >= 4
    assert len(selected_species) == 67


@pytest.mark.skipif(
    not RELEASE_MANIFEST_PATH.exists() or not GOLDEN_IMAGES_PATH.exists(),
    reason="Generated test release artifacts not present yet",
)
def test_generated_release_artifacts_on_disk() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(RELEASE_MANIFEST_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker()).validate(manifest)

    assert manifest["channel"] == "test"
    assert manifest["shipGates"]["passed"] is False
    assert "rejectClasses" not in manifest
    assert manifest["model"]["sha256"] == compute_file_sha256(TFLITE_PATH)

    # Verify release SHA256SUMS (relative paths resolvable from release dir)
    release_dir = RELEASE_MANIFEST_PATH.parent
    sums_lines = (release_dir / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    sums_map = {line.split()[1]: line.split()[0] for line in sums_lines}
    assert sums_map["model-manifest.json"] == _sha256(RELEASE_MANIFEST_PATH)
    tflite_rel_entries = [k for k in sums_map if k.endswith(manifest["model"]["file"])]
    assert len(tflite_rel_entries) == 1
    tflite_rel = tflite_rel_entries[0]
    assert (release_dir / tflite_rel).resolve() == TFLITE_PATH.resolve()
    assert sums_map[tflite_rel] == _sha256(TFLITE_PATH)

    # Verify Golden Images Pack
    golden = json.loads(GOLDEN_IMAGES_PATH.read_text(encoding="utf-8"))
    assert golden["split"] == "val"
    assert golden["count"] >= 200
    assert len(golden["images"]) == golden["count"]
    assert golden["summary"]["activeClassesCovered"] == 66
    assert golden["summary"]["activeHighStakesCovered"] == 26
    assert golden["summary"]["uncoveredAmanitaVernaCount"] == 5

    sample = golden["images"][0]
    assert len(sample["pytorchLogits"]) == 66
    assert len(sample["tfliteFp16Logits"]) == 66
    assert len(sample["imageSha256"]) == 64
    assert not Path(sample["relativePath"]).is_absolute()
    assert sample["expectedDecision"]["status"] in {"ok", "unclear", "out_of_distribution"}


def test_upload_release_to_hf_blocks_public_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    import huggingface_hub

    from grzybobranie_ml.manifest import upload_release_to_hf

    tflite_f = tmp_path / "m.tflite"
    tflite_f.write_bytes(b"dummy")
    rel_dir = tmp_path / "release"
    rel_dir.mkdir()
    manifest = copy.deepcopy(json.loads(RELEASE_MANIFEST_PATH.read_text(encoding="utf-8"))) if RELEASE_MANIFEST_PATH.exists() else {}
    if not manifest:
        pytest.skip("Release manifest not generated")
    manifest["model"]["file"] = "m.tflite"
    manifest["model"]["sha256"] = _sha256(tflite_f)
    (rel_dir / "model-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    class FakeApi:
        def __init__(self, token: str | None = None) -> None:
            pass

        def repo_info(self, repo_id: str, repo_type: str = "model") -> SimpleNamespace:
            return SimpleNamespace(private=False)

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    with pytest.raises(ValueError, match="non-private HF repo"):
        upload_release_to_hf(
            repo_id="owner/public-model",
            token="dummy",
            tflite_path=tflite_f,
            release_dir=rel_dir,
            golden_dir=tmp_path / "none",
            p1_dir=tmp_path / "none_p1",
            p2_dir=tmp_path / "none_p2",
        )


def test_build_test_manifest_handles_calib_only_report(tmp_path: Path) -> None:
    v2_tflite = REPO_ROOT / "artifacts" / "models" / "mobilenetv4_hybrid_medium_r256_fp16.tflite"
    v2_ship = REPO_ROOT / "artifacts" / "v2" / "ship_gates" / "ship_gates_report.json"
    v2_quant = REPO_ROOT / "artifacts" / "v2" / "quantization" / "quantization_drift_val.json"
    v2_calib = REPO_ROOT / "artifacts" / "v2" / "calibration" / "calibration_results.json"
    v2_export = REPO_ROOT / "artifacts" / "v2" / "quantization" / "export_manifest.json"
    if not all(p.exists() for p in (v2_tflite, v2_ship, v2_quant, v2_calib, v2_export)):
        pytest.skip("v2 release artifacts not present")

    manifest = build_test_manifest(
        tflite_path=v2_tflite,
        ship_gates_report_path=v2_ship,
        quant_drift_path=v2_quant,
        calibration_report_path=v2_calib,
        export_manifest_path=v2_export,
        model_version="0.2.0-rc1-test",
        channel="test",
    )
    assert manifest["modelVersion"] == "0.2.0-rc1-test"
    assert manifest["calibration"]["temperature"] == pytest.approx(1.133)
    assert manifest["ood"]["energyThreshold"] == pytest.approx(-2.9074)
    assert manifest["ood"]["minTop1Prob"] == pytest.approx(0.70)
    assert "auroc" not in manifest["ood"]["metricsTest"]
