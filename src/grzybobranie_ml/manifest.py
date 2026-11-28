"""Stage P7 & P9: Generate Golden Photos parity pack and schema-validated model-manifest.json.

Module: grzybobranie_ml.manifest
CLI: python -m grzybobranie_ml.manifest [options]

Generates:
1. Golden Photos Parity Pack (`artifacts/p7_parity/golden_images.json` + `SHA256SUMS`):
   >= 200 representative photos strictly from `data/splits/val.csv` covering all 26 active
   `high_stakes` species, all 8 hardest look-alike pairs, remaining active head classes,
   and 5 uncovered `Amanita verna` cases, with P1 preprocessed PyTorch & TFLite FP16 logits,
   energy E(x), and expected P2 decisions.
2. Test Release Manifest (`artifacts/release_test/model-manifest.json` + `SHA256SUMS`):
   Validates strictly against `schemas/model-manifest.schema.json` (Draft 2020-12) with
   `channel: "test"`, `shipGates.passed: false`, `uncoveredSpecies: ["amanita_verna"]`,
   and no `rejectClasses`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import ai_edge_litert
import ai_edge_litert.interpreter as tflite
import jsonschema
import litert_torch
import numpy as np
import torch

from grzybobranie_ml.export_tflite import TFLiteRunner, load_and_prepare_model
from grzybobranie_ml.preprocess import baseline_preprocessing, preprocess
from grzybobranie_ml.recognition_math import decide_single, energy, softmax

logger = logging.getLogger("grzybobranie_ml.manifest")

DEFAULT_CHECKPOINT = Path(
    "artifacts/train_runs/mobilenetv4_conv_medium.e500_r256_in1k_r256_1795514748/best_val_checkpoint.pt"
)
DEFAULT_TFLITE = Path("artifacts/models/mobilenetv4_medium_r256_fp16.tflite")
DEFAULT_VAL_CSV = Path("data/splits/val.csv")
DEFAULT_IMAGE_DIR = Path("/home/bubolka/hf-768/upload")
DEFAULT_COVERAGE_CSV = Path("configs/pokrycie-67.csv")
DEFAULT_KARTY_PATH = Path("configs/karty_sobowtory.json")
DEFAULT_SCHEMA_PATH = Path("schemas/model-manifest.schema.json")
DEFAULT_SHIP_GATES_REPORT = Path("artifacts/ship_gates/ship_gates_report.json")
DEFAULT_QUANT_DRIFT_REPORT = Path("artifacts/quantization/quantization_drift_val.json")
DEFAULT_CALIBRATION_REPORT = Path("artifacts/calibration/calibration_results.json")
DEFAULT_EXPORT_MANIFEST = Path("artifacts/quantization/export_manifest.json")
DEFAULT_LEAK_REPORT = Path("artifacts/leak_report.json")
DEFAULT_OOD_IDS = Path("data/ood-v1/ood_ids.txt")
DEFAULT_P1_PARITY = Path("fixtures/p1/parity.json")
DEFAULT_P2_DECISION = Path("fixtures/p2/decision_cases.json")
DEFAULT_P2_FUSION = Path("fixtures/p2/fusion_cases.json")
DEFAULT_GOLDEN_OUTPUT_DIR = Path("artifacts/p7_parity")
DEFAULT_RELEASE_OUTPUT_DIR = Path("artifacts/release_test")

PROBLEMATIC_8_EDIBLE = (
    "macrolepiota_procera",
    "agaricus_campestris",
    "agaricus_sylvaticus",
    "kuehneromyces_mutabilis",
    "lactarius_deliciosus",
    "russula_vesca",
    "hypholoma_capnoides",
    "neoboletus_luridiformis",
)

REQUIRED_DANGEROUS_GENERA = (
    "Amanita",
    "Cortinarius",
    "Galerina",
    "Gyromitra",
    "Inocybe",
    "Inosperma",
    "Lepiota",
)


def compute_file_sha256(path: str | Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def _current_git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        return out or "unknown"
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def _load_coverage_table(coverage_csv: Path) -> list[dict[str, Any]]:
    with open(coverage_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows


def _load_manifest_768_index(image_dir: Path) -> dict[str, tuple[Path, str]]:
    """Map photo identifiers/URLs to ``(absolute_path, relative_path)`` under ``image_dir``."""
    lookup: dict[str, tuple[Path, str]] = {}
    mf = image_dir / "manifest_768.csv"
    if not mf.exists():
        return lookup
    with open(mf, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rel = (row.get("path") or "").strip()
            if not rel:
                continue
            full_p = image_dir / rel
            if not full_p.exists():
                continue
            for key in ("image_url", "photo_id", "source_url"):
                val = (row.get(key) or "").strip()
                if val:
                    lookup[val] = (full_p, rel)
    return lookup


def _resolve_val_image(
    row: dict[str, str],
    image_dir: Path,
    lookup: dict[str, tuple[Path, str]],
) -> tuple[Path, str] | None:
    for key in ("image_url", "canonical_url", "photo_id", "inat_photo_id", "mo_image_id", "source_url"):
        val = (row.get(key) or "").strip()
        if val and val in lookup:
            return lookup[val]

    slug = (row.get("species_slug") or "").strip()
    pid = (row.get("photo_id") or row.get("inat_photo_id") or row.get("mo_image_id") or "").strip()
    file_name = (row.get("file_name") or "").strip()
    candidates = [
        file_name,
        f"{slug}/{pid}.jpg" if slug and pid else "",
        f"{pid}.jpg" if pid else "",
        f"images/{slug}/{pid}.jpg" if slug and pid else "",
    ]
    for rel in candidates:
        if not rel:
            continue
        p = image_dir / rel
        if p.exists():
            return p, rel
    return None


def validate_manifest_against_schema(
    manifest: dict[str, Any],
    schema_path: str | Path = DEFAULT_SCHEMA_PATH,
) -> None:
    """Validate manifest with JSON Schema Draft 2020-12 and project safety invariants."""
    with open(schema_path, encoding="utf-8") as f:
        schema = json.load(f)
    validator = jsonschema.Draft202012Validator(
        schema, format_checker=jsonschema.FormatChecker()
    )
    validator.validate(manifest)

    # Project invariants beyond structural schema
    if "rejectClasses" in manifest:
        raise ValueError("rejectClasses must not be included in Grzybobranie AI manifests.")
    if any(not c.get("in_model", False) for c in manifest["classes"]):
        raise ValueError("All classes in model head must have in_model=true.")

    indices = [int(c["index"]) for c in manifest["classes"]]
    if indices != list(range(len(manifest["classes"]))):
        raise ValueError("Class indices in manifest['classes'] must be sequential 0..K-1.")

    head_ids = {c["classId"] for c in manifest["classes"]}
    head_species_ids = {
        sid for c in manifest["classes"] for sid in c.get("speciesIds", [])
    }
    uncovered_ids = {u["speciesId"] for u in manifest.get("uncoveredSpecies", [])}
    if (head_ids | head_species_ids) & uncovered_ids:
        raise ValueError(
            f"Overlap between classes and uncoveredSpecies: {(head_ids | head_species_ids) & uncovered_ids}"
        )
    if "amanita_verna" in head_ids or "amanita_verna" in head_species_ids:
        raise ValueError("Uncovered species amanita_verna must not appear in classes.")

    ship_passed = bool(manifest["shipGates"]["passed"])
    if not ship_passed and manifest["channel"] != "test":
        raise ValueError("Manifest with shipGates.passed == false must use channel='test'.")


def build_test_manifest(
    *,
    tflite_path: str | Path = DEFAULT_TFLITE,
    coverage_csv: str | Path = DEFAULT_COVERAGE_CSV,
    ship_gates_report_path: str | Path = DEFAULT_SHIP_GATES_REPORT,
    quant_drift_path: str | Path = DEFAULT_QUANT_DRIFT_REPORT,
    calibration_report_path: str | Path = DEFAULT_CALIBRATION_REPORT,
    export_manifest_path: str | Path = DEFAULT_EXPORT_MANIFEST,
    leak_report_path: str | Path = DEFAULT_LEAK_REPORT,
    ood_ids_path: str | Path = DEFAULT_OOD_IDS,
    p1_parity_path: str | Path = DEFAULT_P1_PARITY,
    schema_path: str | Path = DEFAULT_SCHEMA_PATH,
    model_version: str = "0.1.0-rc1-test",
    channel: str = "test",
) -> dict[str, Any]:
    """Assemble and validate the P7/P9 test release model-manifest.json."""
    if channel != "test":
        raise ValueError(
            f"Refusing to build test RC manifest with channel={channel!r}; "
            "only channel='test' is permitted when shipGates.passed=false."
        )

    tflite_p = Path(tflite_path)
    if not tflite_p.exists():
        raise FileNotFoundError(f"TFLite model not found: {tflite_p}")

    ship_report = json.loads(Path(ship_gates_report_path).read_text(encoding="utf-8"))
    quant_drift = json.loads(Path(quant_drift_path).read_text(encoding="utf-8"))
    calib_report = json.loads(Path(calibration_report_path).read_text(encoding="utf-8"))
    export_meta = json.loads(Path(export_manifest_path).read_text(encoding="utf-8"))
    leak_report = json.loads(Path(leak_report_path).read_text(encoding="utf-8"))

    ship_passed = bool(ship_report.get("overall_ship_gates_pass", False))
    if ship_passed:
        raise ValueError("Unexpected shipGates.passed=True in test manifest builder.")

    # Inspect actual TFLite tensor details
    interp = tflite.Interpreter(model_path=str(tflite_p))
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]

    input_size = int(in_det["shape"][1])
    active_classes: list[str] = list(export_meta["active_classes"])

    coverage_rows = _load_coverage_table(Path(coverage_csv))
    coverage_by_slug = {r["class"]: r for r in coverage_rows}

    classes_entries = []
    for idx, slug in enumerate(active_classes):
        cov = coverage_by_slug[slug]
        genus = cov["nazwa_lacinska"].split()[0]
        is_hs = str(cov["high_stakes"]).strip().lower() == "true"
        classes_entries.append({
            "index": idx,
            "classId": slug,
            "speciesIds": [slug],
            "genus": genus,
            "in_model": True,
            "highStakes": is_hs,
        })

    uncovered_entries = []
    for slug in sorted(coverage_by_slug):
        if slug in active_classes:
            continue
        cov = coverage_by_slug[slug]
        hs_flag = str(cov.get("high_stakes", "")).strip().lower() == "true"
        prog = cov.get("prog") or ("55" if hs_flag else "40")
        tier = "high_stakes" if hs_flag else "pozostałych"
        uncovered_entries.append({
            "speciesId": slug,
            "reason": (
                f"Niewystarczająca liczba zdjęć treningowych (poniżej progu {prog} w train dla {tier}); "
                f"wykluczono z głowicy K={len(active_classes)}."
            ),
        })

    frozen_cfg = ship_report["frozen_configuration"]
    t_single = float(frozen_cfg["temperature_single"])
    t_multi = float(frozen_cfg["temperature_multi"])
    energy_th = float(frozen_cfg["energy_threshold_theta"])
    min_top1 = float(frozen_cfg["min_top1_prob"])
    disagreement_th = float(frozen_cfg.get("disagreement_threshold", 0.7))
    twin_veto = float(frozen_cfg.get("twin_pair_veto_threshold", 0.1))
    twin_ratio = float(frozen_cfg.get("twin_pair_min_ratio", 5.0))
    crop_mode = str(frozen_cfg.get("crop_mode", "none_stretch"))

    single_audit = ship_report["single_frame_audit"]
    ood_audit = ship_report["ood_test_audit"]
    multi_audit = ship_report["multi_photo_audit"]
    ood_splits = calib_report.get("ood", {}).get("splits", {})
    ood_test_calib = ood_splits.get("ood_test")

    max_pair_err = max(
        (float(p["max_confusion"]) for p in single_audit["dangerous_pairs"]["pairs"].values()),
        default=0.0,
    )
    failing_pair_cnt = int(single_audit["dangerous_pairs"]["failing_pairs_count"])
    total_pairs_cnt = int(single_audit["dangerous_pairs"]["total_active_pairs"])
    min_hs_rec = float(single_audit["high_stakes_recall"]["min_conditional_recall"])
    failing_hs_cnt = int(single_audit["high_stakes_recall"]["failing_species_count"])
    total_hs_cnt = int(single_audit["high_stakes_recall"]["total_evaluated"])

    in_dtype = np.dtype(in_det["dtype"])
    quant_hint = str(export_meta.get("quantization", tflite_p.stem)).lower()
    if in_dtype == np.int8 or "int8" in quant_hint:
        weight_quant = "int8_full"
    elif "dynamic" in quant_hint:
        weight_quant = "dynamic_range"
    elif "fp32" in quant_hint:
        weight_quant = "none"
    else:
        weight_quant = "fp16"

    ood_calib_split = ood_splits["ood_calib"]
    n_ood_calib = int(ood_calib_split["n_samples"])

    preprocessing_spec = baseline_preprocessing(input_size)
    p1_path = Path(p1_parity_path)
    if p1_path.exists():
        preprocessing_spec["parityFixturesSha256"] = compute_file_sha256(p1_path)

    ood_sha = (
        compute_file_sha256(ood_ids_path)
        if Path(ood_ids_path).exists()
        else "0" * 64
    )

    metrics_test: dict[str, Any] = {
        "knownKeepRate": float(single_audit["keep_rate_tflite"]),
        "confidentEdibleRate": float(ood_audit["confident_edible_rate"]),
    }
    if isinstance(ood_test_calib, dict):
        if "auroc" in ood_test_calib:
            metrics_test["auroc"] = float(ood_test_calib["auroc"])
        if "fpr_at_95tpr" in ood_test_calib:
            metrics_test["fprAt95Tpr"] = float(ood_test_calib["fpr_at_95tpr"])

    nf_test = ood_audit.get("nonfungi_test")
    f_test = ood_audit.get("fungi_test")
    if isinstance(nf_test, dict) and isinstance(f_test, dict):
        ood_note = (
            f"Wilson 95% UB: {ood_audit['wilson_upper_bound_95']*100:.2f}% (wymóg <= 5%); "
            f"ood_nonfungi: {nf_test['confident_edible_rate']*100:.2f}% ({nf_test['confident_edible_count']}/{nf_test['n_samples']}), "
            f"ood_fungi: {f_test['confident_edible_rate']*100:.2f}% ({f_test['confident_edible_count']}/{f_test['n_samples']})."
        )
    else:
        ood_note = (
            f"Wilson 95% UB: {ood_audit['wilson_upper_bound_95']*100:.2f}% (wymóg <= 5%)."
        )

    ckpt_short = str(ship_report.get("checkpoint_sha256", ""))[:8]
    comment = (
        f"WYDANIE TESTOWE {model_version} (channel: 'test', shipGates.passed: false). "
        f"Checkpoint {ckpt_short}, crop='{crop_mode}', K={len(active_classes)}, "
        f"progi: T_single={t_single:.4f}, T_multi={t_multi:.4f}, theta={energy_th:.4f}, "
        f"minTop1Prob={min_top1:.2f}, disagreement={disagreement_th:.2f}, "
        f"asymetryczne weto P5: p_t<{twin_veto:.2f} i p_e>={twin_ratio:.1f}*p_t."
    )

    manifest: dict[str, Any] = {
        "$comment": comment,
        "schemaVersion": "1.0.1",
        "modelVersion": model_version,
        "channel": "test",
        "createdAt": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "model": {
            "file": tflite_p.name,
            "sha256": compute_file_sha256(tflite_p),
            "sizeBytes": int(tflite_p.stat().st_size),
            "format": "tflite",
            "converter": f"litert-torch {export_meta.get('litert_torch_version', getattr(litert_torch, '__version__', '0.10.0'))}",
            "litertVersionTested": str(
                export_meta.get("ai_edge_litert_version", getattr(ai_edge_litert, "__version__", "2.3.0"))
            ),
            "architecture": f"timm/{export_meta.get('model_name', 'mobilenetv4_conv_medium.e500_r256_in1k')}",
            "baseWeightsLicense": "apache-2.0",
            "input": {
                "name": str(in_det["name"]),
                "shape": [int(x) for x in in_det["shape"]],
                "dtype": "float32",
                "layout": "NHWC",
            },
            "output": {
                "name": str(out_det["name"]),
                "shape": [int(x) for x in out_det["shape"]],
                "dtype": "float32",
                "kind": "logits",
            },
            "weightQuantization": weight_quant,
        },
        "preprocessing": preprocessing_spec,
        "classes": classes_entries,
        "uncoveredSpecies": uncovered_entries,
        "calibration": {
            "calibrated": True,
            "method": "temperature",
            "temperature": t_single,
            "fitSplit": "val",
            "eceBins": 15,
            "eceVal": float(quant_drift["tflite_fp16"]["ece"]),
            "eceTest": float(single_audit["ece_tflite"]),
            "nTest": int(single_audit["dataset_size"]),
            "measuredOn": "tflite",
            "displayBands": {
                "high": 0.8,
                "medium": 0.5,
            },
        },
        "ood": {
            "method": "energy",
            "energyTemperature": t_single,
            "energyThreshold": energy_th,
            "minTop1Prob": min_top1,
            "lowConfidenceBelow": min_top1,
            "dangerousGenusMinProbability": twin_veto,
            "dataset": {
                "name": "HF_DATASET_REPO:data/ood-v1",
                "sha256": ood_sha,
                "nCalibration": n_ood_calib,
                "nTest": int(ood_audit["n_samples"]),
            },
            "metricsTest": metrics_test,
            "fusion": {
                "2": {
                    "energyThreshold": energy_th,
                    "minTop1Prob": min_top1,
                },
                "3": {
                    "energyThreshold": energy_th,
                    "minTop1Prob": min_top1,
                },
            },
        },
        "fusionWeights": {
            "method": "entropy_weighted_logits",
            "maxPhotos": 3,
            "gateBeforeFusion": True,
        },
        "dangerousGenera": list(REQUIRED_DANGEROUS_GENERA),
        "shipGates": {
            "passed": False,
            "evaluatedAt": str(ship_report["audit_timestamp"]),
            "reportSha256": compute_file_sha256(ship_gates_report_path),
            "gates": {
                "confidentToxicAsEdible": {
                    "value": int(single_audit["confident_toxic_as_edible_accepted"]),
                    "max": 0,
                    "support": int(single_audit["dataset_size"]),
                    "passed": int(single_audit["confident_toxic_as_edible_accepted"]) == 0,
                    "note": (
                        f"Wydanie testowe {model_version}: "
                        f"{single_audit['confident_toxic_as_edible_accepted']} pojedynczych klatek toxic->edible "
                        f"(raw={single_audit.get('confident_toxic_as_edible_raw', 'n/a')})."
                    ),
                },
                "dangerousPairRateMax": {
                    "value": max_pair_err,
                    "max": 0.05,
                    "support": total_pairs_cnt,
                    "passed": failing_pair_cnt == 0,
                    "note": f"Failing pairs: {failing_pair_cnt}/{total_pairs_cnt}.",
                },
                "highStakesRecallMin": {
                    "value": min_hs_rec,
                    "min": 0.85,
                    "support": total_hs_cnt,
                    "passed": failing_hs_cnt == 0,
                    "note": f"Failing high_stakes species (n >= 8): {failing_hs_cnt}/{total_hs_cnt}.",
                },
                "tfliteAgreement": {
                    "value": float(single_audit["tflite_agreement_overall"]),
                    "min": 0.99,
                    "support": int(single_audit["dataset_size"]),
                    "passed": float(single_audit["tflite_agreement_overall"]) >= 0.99,
                },
                "tfliteAgreementHighRisk": {
                    "value": float(single_audit["tflite_agreement_high_stakes"]),
                    "min": 0.99,
                    "passed": float(single_audit["tflite_agreement_high_stakes"]) >= 0.99,
                },
                "oodKnownKeepRate": {
                    "value": float(single_audit["keep_rate_tflite"]),
                    "min": 0.95,
                    "support": int(single_audit["dataset_size"]),
                    "passed": float(single_audit["keep_rate_tflite"]) >= 0.95,
                },
                "openSetConfidentEdible": {
                    "value": float(ood_audit["confident_edible_rate"]),
                    "max": 0.02,
                    "support": int(ood_audit["n_samples"]),
                    "passed": bool(ood_audit["passed"]),
                    "note": ood_note,
                },
                "fusionGates": {
                    "value": int(multi_audit["fused_toxic_as_edible_count"]),
                    "max": 0,
                    "support": int(multi_audit["n_multi_observations"]),
                    "passed": bool(multi_audit["passed"]),
                    "note": (
                        f"Multi-photo conditional_accuracy={multi_audit['conditional_accuracy']*100:.2f}%, "
                        f"toxic->edible={multi_audit['fused_toxic_as_edible_count']}, T_multi={t_multi}."
                    ),
                },
                "deviceParityP3": {
                    "value": None,
                    "min": 0.99,
                    "passed": False,
                    "note": "Nie zmierzone na telefonach — pakiet P7 gotowy do pomiarów.",
                },
                "latencyP95Single": {
                    "value": None,
                    "max": 2000,
                    "passed": False,
                    "note": "Nie zmierzone na Xiaomi Mi 11 Lite 5G — do pomiaru w P7.",
                },
                "latencyP95Session3": {
                    "value": None,
                    "max": 5000,
                    "passed": False,
                    "note": "Nie zmierzone na Xiaomi Mi 11 Lite 5G — do pomiaru w P7.",
                },
            },
        },
        "minAtlasVersion": "0.1.0",
        "provenance": {
            "trainingDataset": "HF_DATASET_REPO (_v3 20261108 + new16 20261110 + addendum)",
            "trainingDatasetSha256": str(leak_report["splits"]["train"]["sha256"]),
            "splitMethod": "70/15/15 stratified HMAC-SHA256 connected components (author+observation)",
            "licensesUsed": ["CC0", "CC BY", "CC BY-NC", "CC BY-SA", "pd_self"],
            "excludedSources": [
                "DF20",
                "DF24",
                "DF20-trained-weights",
                "DF24-trained-weights",
                "ND",
                "AI-generated",
            ],
            "generatedImagesUsed": False,
            "gitCommit": _current_git_commit(),
            "trackioRun": Path(ship_report["checkpoint"]).parent.name,
            "modelRepo": "HF_MODEL_REPO",
            "modelRevision": model_version,
        },
    }

    validate_manifest_against_schema(manifest, schema_path=schema_path)
    return manifest


def select_golden_records(
    val_rows: list[dict[str, str]],
    active_classes: list[str],
    high_stakes_slugs: set[str],
    target_total: int = 240,
) -> list[dict[str, str]]:
    """Deterministically select >= 200 representative photos from val.csv covering HS, pairs, and Amanita verna."""
    if target_total < 200:
        raise ValueError("Golden pack must contain at least 200 images.")

    by_species: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in val_rows:
        slug = (r.get("species_slug") or "").strip()
        if slug:
            by_species[slug].append(r)

    for slug, rows in by_species.items():
        rows.sort(
            key=lambda row: (
                row.get("photo_id") or row.get("inat_photo_id") or row.get("mo_image_id") or "",
                row.get("canonical_url") or row.get("image_url") or "",
            )
        )

    selected: list[dict[str, str]] = []
    seen_keys: set[str] = set()

    def _row_key(row: dict[str, str]) -> str:
        return (
            row.get("canonical_url")
            or row.get("image_url")
            or row.get("photo_id")
            or json.dumps(row, sort_keys=True)
        )

    def _take(slug: str, count: int) -> None:
        taken = 0
        for row in by_species.get(slug, []):
            if taken >= count:
                break
            k = _row_key(row)
            if k in seen_keys:
                continue
            seen_keys.add(k)
            selected.append(row)
            taken += 1

    # 1. 5 uncovered Amanita verna cases
    _take("amanita_verna", 5)

    # 2. All 26 active high_stakes classes (4 per class = 104)
    for slug in sorted(s for s in high_stakes_slugs if s in active_classes):
        _take(slug, 4)

    # 3. Edible counterparts of the 8 hardest pairs (4 per class = 32)
    for slug in PROBLEMATIC_8_EDIBLE:
        if slug in active_classes:
            _take(slug, 4)

    # 4. Remaining active classes (3 per class so all 66 active classes are covered)
    for slug in sorted(active_classes):
        if slug not in high_stakes_slugs and slug not in PROBLEMATIC_8_EDIBLE:
            _take(slug, 3)

    # 5. Top up deterministically across problematic pairs & active HS until target_total
    round_robin_slugs = list(PROBLEMATIC_8_EDIBLE) + sorted(
        s for s in high_stakes_slugs if s in active_classes
    )
    while len(selected) < target_total:
        progressed = False
        for slug in round_robin_slugs:
            if len(selected) >= target_total:
                break
            before = len(selected)
            _take(slug, 1)
            if len(selected) > before:
                progressed = True
        if not progressed:
            break

    return selected


def build_golden_images_pack(
    *,
    manifest: dict[str, Any],
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    tflite_path: str | Path = DEFAULT_TFLITE,
    val_csv: str | Path = DEFAULT_VAL_CSV,
    image_dir: str | Path = DEFAULT_IMAGE_DIR,
    coverage_csv: str | Path = DEFAULT_COVERAGE_CSV,
    target_total: int = 240,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict[str, Any]:
    """Generate Golden Photos Pack (>= 200 val.csv images) with PyTorch & TFLite FP16 logits."""
    val_p = Path(val_csv).resolve()
    if val_p.name == "test.csv":
        raise ValueError("Refusing to build golden images pack from test.csv.")

    with open(val_p, encoding="utf-8") as f:
        val_rows = list(csv.DictReader(f))

    cov_rows = _load_coverage_table(Path(coverage_csv))
    high_stakes_slugs = {
        r["class"] for r in cov_rows if str(r["high_stakes"]).strip().lower() == "true"
    }
    active_classes = [c["classId"] for c in manifest["classes"]]

    img_dir_p = Path(image_dir)
    lookup = _load_manifest_768_index(img_dir_p)
    prep_spec = manifest["preprocessing"]

    available_val_rows: list[dict[str, str]] = []
    resolved_by_row: list[tuple[dict[str, str], Path, str]] = []
    for row in val_rows:
        resolved = _resolve_val_image(row, img_dir_p, lookup)
        if resolved is not None:
            available_val_rows.append(row)
            resolved_by_row.append((row, resolved[0], resolved[1]))

    resolved_map: dict[int, tuple[Path, str]] = {
        id(row): (full_p, rel_p) for row, full_p, rel_p in resolved_by_row
    }

    selected_rows = select_golden_records(
        val_rows=available_val_rows,
        active_classes=active_classes,
        high_stakes_slugs=high_stakes_slugs,
        target_total=target_total,
    )

    # Prepare PyTorch model (66 classes, float32) and TFLite FP16 runner
    device = torch.device(device_str)
    wrapped_model, meta = load_and_prepare_model(checkpoint_path, exclude_verna=True)
    pt_model = wrapped_model.to(device).eval()
    idx_to_class: dict[int, str] = meta["idx_to_class"]
    tflite_runner = TFLiteRunner(tflite_path, num_threads=4)

    t_prob = float(manifest["calibration"]["temperature"])
    t_energy = float(manifest["ood"].get("energyTemperature", t_prob))

    items: list[dict[str, Any]] = []
    status_counts: dict[str, int] = defaultdict(int)
    agree_count = 0

    for i, row in enumerate(selected_rows):
        full_path, rel_path = resolved_map[id(row)]
        img_sha = compute_file_sha256(full_path)

        # Exact P1 reference preprocessing (NHWC float32)
        nhwc = preprocess(full_path, prep_spec, layout="NHWC")["float"]
        nhwc_batch = np.expand_dims(nhwc, axis=0)

        with torch.no_grad():
            pt_in = torch.from_numpy(nhwc_batch).to(device=device, dtype=torch.float32)
            pt_logits = pt_model(pt_in)[0].cpu().numpy().astype(np.float64)

        tf_logits = tflite_runner.predict_single(nhwc).astype(np.float64)

        decision = decide_single(tf_logits, manifest)
        e_val = energy(tf_logits, t_energy)
        probs_tf = softmax(tf_logits, t_prob)
        p_top1 = float(np.max(probs_tf))

        pt_top1_idx = int(np.argmax(pt_logits))
        tf_top1_idx = int(np.argmax(tf_logits))
        agreed = pt_top1_idx == tf_top1_idx
        if agreed:
            agree_count += 1
        status_counts[decision["status"]] += 1

        pid = (
            row.get("photo_id")
            or row.get("inat_photo_id")
            or row.get("mo_image_id")
            or hashlib.sha256((row.get("canonical_url") or rel_path).encode()).hexdigest()[:12]
        )
        gt_slug = row["species_slug"]

        items.append({
            "index": i,
            "photoId": str(pid),
            "relativePath": rel_path,
            "imageSha256": img_sha,
            "groundTruthSlug": gt_slug,
            "isUncovered": gt_slug == "amanita_verna",
            "isHighStakes": gt_slug in high_stakes_slugs,
            "pytorchLogits": [float(x) for x in pt_logits],
            "tfliteFp16Logits": [float(x) for x in tf_logits],
            "energy": float(e_val),
            "pTop1": p_top1,
            "pytorchTop1Slug": idx_to_class[pt_top1_idx],
            "tfliteTop1Slug": idx_to_class[tf_top1_idx],
            "agreement": agreed,
            "expectedDecision": decision,
        })

    covered_active_classes = sorted(
        {it["groundTruthSlug"] for it in items if not it["isUncovered"]}
    )
    covered_hs = sorted(
        {it["groundTruthSlug"] for it in items if it["isHighStakes"] and not it["isUncovered"]}
    )
    verna_cnt = sum(1 for it in items if it["isUncovered"])

    return {
        "schemaVersion": manifest["schemaVersion"],
        "modelVersion": manifest["modelVersion"],
        "split": "val",
        "tfliteFile": manifest["model"]["file"],
        "tfliteSha256": manifest["model"]["sha256"],
        "preprocessing": prep_spec,
        "temperature": t_prob,
        "energyTemperature": t_energy,
        "energyThreshold": float(manifest["ood"]["energyThreshold"]),
        "minTop1Prob": float(manifest["ood"]["minTop1Prob"]),
        "count": len(items),
        "summary": {
            "activeClassesCovered": len(covered_active_classes),
            "activeHighStakesCovered": len(covered_hs),
            "uncoveredAmanitaVernaCount": verna_cnt,
            "pytorchVsTfliteAgreement": float(agree_count / len(items)) if items else 0.0,
            "decisionStatusCounts": dict(sorted(status_counts.items())),
        },
        "images": items,
    }


def _render_test_release_readme(
    manifest: dict[str, Any],
    golden_pack: dict[str, Any],
    release_files: dict[str, str],
) -> str:
    gates = manifest["shipGates"]["gates"]
    lines = [
        f"# Wydanie Testowe P7/P9: `{manifest['modelVersion']}` (`channel: \"{manifest['channel']}\"`)",
        "",
        (
            "> **UWAGA — WYDANIE TESTOWE:** `shipGates.passed` ma wartość `false`. "
            "Ten pakiet służy wyłącznie do testów integracji P1–P3 i benchmarków wydajności/pamięci na telefonach "
            "(**Xiaomi Mi 11 Lite 5G**, **Huawei P smart 2019**, **Samsung S23**). Nie może trafić na kanał produkcyjny."
        ),
        "",
        "## Zawartość paczki i sumy SHA-256",
        "",
        f"- **Model TFLite FP16:** `artifacts/models/{manifest['model']['file']}` (`{manifest['model']['sizeBytes']:,}` B, SHA-256: `{manifest['model']['sha256']}`)",
        f"- **Manifest JSON (v1.0.1):** `model-manifest.json` (SHA-256: `{release_files['model-manifest.json']}`)",
        f"- **Złote zdjęcia P3 (`val.csv`, N={golden_pack['count']}):** `golden_images.json` (SHA-256: `{release_files['golden_images.json']}`)",
        f"- **Fixture'y P1 (`p1/parity.json`):** SHA-256 `{release_files['p1_parity.json']}`",
        f"- **Fixture'y P2 (`p2/decision_cases.json`, `p2/fusion_cases.json`):** SHA-256 `{release_files['p2_decision_cases.json']}`, `{release_files['p2_fusion_cases.json']}`",
        "",
        "## Wyniki P6 (`test.csv` + `ood_test`)",
        "",
        f"- **`GATE-04` (TFLite FP16 Agreement):** {gates['tfliteAgreement']['value']*100:.2f}% (High-Stakes: {gates['tfliteAgreementHighRisk']['value']*100:.2f}%, status: `{'PASS' if gates['tfliteAgreement']['passed'] else 'FAIL'}`)",
        f"- **`GATE-05` (`ood_test` Confident Edible):** {gates['openSetConfidentEdible']['value']*100:.2f}% — {gates['openSetConfidentEdible'].get('note', '')}",
        f"- **`GATE-06` (Multi-Photo P5):** {gates['fusionGates'].get('note', '')}",
        "",
        "## Pokrycie Złotych Zdjęć (`golden_images.json`)",
        "",
        f"- Liczba zdjęć: **{golden_pack['count']}** (wyłącznie z `val.csv`)",
        f"- Aktywne klasy w głowicy: **{golden_pack['summary']['activeClassesCovered']} / 66**",
        f"- Aktywne klasy `high_stakes`: **{golden_pack['summary']['activeHighStakesCovered']} / 26**",
        f"- Przypadki niepokrytego gatunku *Amanita verna*: **{golden_pack['summary']['uncoveredAmanitaVernaCount']}**",
        f"- Zgodność PyTorch vs TFLite FP16 na złotych zdjęciach: **{golden_pack['summary']['pytorchVsTfliteAgreement']*100:.2f}%**",
        "",
    ]
    return "\n".join(lines)


def generate_test_release_package(
    *,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    tflite_path: str | Path = DEFAULT_TFLITE,
    val_csv: str | Path = DEFAULT_VAL_CSV,
    image_dir: str | Path = DEFAULT_IMAGE_DIR,
    coverage_csv: str | Path = DEFAULT_COVERAGE_CSV,
    schema_path: str | Path = DEFAULT_SCHEMA_PATH,
    ship_gates_report_path: str | Path = DEFAULT_SHIP_GATES_REPORT,
    quant_drift_path: str | Path = DEFAULT_QUANT_DRIFT_REPORT,
    calibration_report_path: str | Path = DEFAULT_CALIBRATION_REPORT,
    export_manifest_path: str | Path = DEFAULT_EXPORT_MANIFEST,
    leak_report_path: str | Path = DEFAULT_LEAK_REPORT,
    ood_ids_path: str | Path = DEFAULT_OOD_IDS,
    p1_parity_path: str | Path = DEFAULT_P1_PARITY,
    golden_output_dir: str | Path = DEFAULT_GOLDEN_OUTPUT_DIR,
    release_output_dir: str | Path = DEFAULT_RELEASE_OUTPUT_DIR,
    model_version: str = "0.1.0-rc1-test",
    target_golden_count: int = 240,
    bundle_release_tree: bool = False,
) -> dict[str, Any]:
    """Generate and validate both model-manifest.json (channel=test) and golden_images.json."""
    release_dir = Path(release_output_dir)
    release_dir.mkdir(parents=True, exist_ok=True)

    p1_parity_for_manifest = Path(p1_parity_path)
    if bundle_release_tree:
        from grzybobranie_ml import make_p1_fixtures, make_p2_fixtures

        make_p1_fixtures.write_fixtures((release_dir / "p1").resolve())
        make_p2_fixtures.write_fixtures((release_dir / "p2").resolve())
        p1_parity_for_manifest = release_dir / "p1" / "parity.json"

    manifest = build_test_manifest(
        tflite_path=tflite_path,
        coverage_csv=coverage_csv,
        ship_gates_report_path=ship_gates_report_path,
        quant_drift_path=quant_drift_path,
        calibration_report_path=calibration_report_path,
        export_manifest_path=export_manifest_path,
        leak_report_path=leak_report_path,
        ood_ids_path=ood_ids_path,
        p1_parity_path=p1_parity_for_manifest,
        schema_path=schema_path,
        model_version=model_version,
        channel="test",
    )

    golden_dir = Path(golden_output_dir)
    golden_dir.mkdir(parents=True, exist_ok=True)
    golden_pack = build_golden_images_pack(
        manifest=manifest,
        checkpoint_path=checkpoint_path,
        tflite_path=tflite_path,
        val_csv=val_csv,
        image_dir=image_dir,
        coverage_csv=coverage_csv,
        target_total=target_golden_count,
    )

    golden_path = golden_dir / "golden_images.json"
    golden_path.write_text(json.dumps(golden_pack, indent=2) + "\n", encoding="utf-8")
    golden_sha = compute_file_sha256(golden_path)
    (golden_dir / "SHA256SUMS").write_text(
        f"{golden_sha}  golden_images.json\n", encoding="utf-8"
    )

    manifest_path = release_dir / "model-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    manifest_sha = compute_file_sha256(manifest_path)

    if bundle_release_tree:
        local_golden = release_dir / "golden_images.json"
        if local_golden.resolve() != golden_path.resolve():
            shutil.copy2(golden_path, local_golden)
        p1_parity_file = release_dir / "p1" / "parity.json"
        p2_dec_file = release_dir / "p2" / "decision_cases.json"
        p2_fus_file = release_dir / "p2" / "fusion_cases.json"
    else:
        p1_parity_file = Path(p1_parity_path)
        p2_dec_file = DEFAULT_P2_DECISION
        p2_fus_file = DEFAULT_P2_FUSION

    release_files = {
        "model-manifest.json": manifest_sha,
        "golden_images.json": golden_sha,
        "p1_parity.json": compute_file_sha256(p1_parity_file) if p1_parity_file.exists() else "",
        "p2_decision_cases.json": compute_file_sha256(p2_dec_file) if p2_dec_file.exists() else "",
        "p2_fusion_cases.json": compute_file_sha256(p2_fus_file) if p2_fus_file.exists() else "",
    }
    readme_path = release_dir / "README_TEST_RELEASE.md"
    readme_path.write_text(
        _render_test_release_readme(manifest, golden_pack, release_files),
        encoding="utf-8",
    )

    tflite_relpath = os.path.relpath(Path(tflite_path).resolve(), release_dir.resolve())
    if bundle_release_tree:
        sum_lines: list[str] = []
        for file_p in sorted(p for p in release_dir.rglob("*") if p.is_file() and p.name != "SHA256SUMS"):
            rel = file_p.relative_to(release_dir).as_posix()
            sum_lines.append(f"{compute_file_sha256(file_p)}  {rel}")
        sum_lines.append(f"{manifest['model']['sha256']}  {tflite_relpath}")
        (release_dir / "SHA256SUMS").write_text("\n".join(sum_lines) + "\n", encoding="utf-8")
    else:
        (release_dir / "SHA256SUMS").write_text(
            f"{manifest_sha}  model-manifest.json\n"
            f"{manifest['model']['sha256']}  {tflite_relpath}\n",
            encoding="utf-8",
        )

    logger.info("Validated and wrote %s (SHA256: %s)", manifest_path, manifest_sha[:12])
    logger.info("Wrote %d golden images to %s (SHA256: %s)", golden_pack["count"], golden_path, golden_sha[:12])

    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "golden_path": str(golden_path),
        "golden_sha256": golden_sha,
        "golden_count": golden_pack["count"],
        "manifest": manifest,
        "golden_summary": golden_pack["summary"],
    }


def load_dotenv_settings(dotenv_path: str | Path = ".env") -> dict[str, str]:
    """Read key=value pairs from .env without overriding non-empty process env vars."""
    settings: dict[str, str] = {}
    p = Path(dotenv_path)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            settings[k.strip()] = v.strip().strip('"').strip("'")
    for key in ("HF_MODEL_REPO", "HF_DATASET_REPO", "HF_TOKEN"):
        env_val = os.environ.get(key, "").strip()
        if env_val:
            settings[key] = env_val
    return settings


def _render_hf_model_card(manifest: dict[str, Any], release_readme_text: str) -> str:
    """Build HF model repository README.md preserving YAML license header."""
    frontmatter = (
        "---\n"
        "license: apache-2.0\n"
        "tags:\n"
        "- image-classification\n"
        "- mobilenetv4\n"
        "- litert\n"
        "- tflite\n"
        "- grzybobranie-ai\n"
        "---\n\n"
    )
    notes = "\n".join([
        "",
        "## Licencja i pochodzenie danych",
        "",
        f"- **Kod i wagi bazowe (`{manifest['model']['architecture']}`):** `{manifest['model']['baseWeightsLicense']}`",
        f"- **Dane treningowe:** {', '.join(manifest['provenance']['licensesUsed'])} (`CC BY-NC` użyte wyłącznie w treningu bezpłatnej aplikacji offline; zakaz `DF20`/`DF24`, `ND` i obrazów generowanych AI).",
        f"- **Status bramek (`shipGates.passed`):** `{manifest['shipGates']['passed']}` (`channel: \"{manifest['channel']}\"`).",
        "",
    ])
    return frontmatter + release_readme_text.strip() + "\n" + notes


def upload_release_to_hf(
    *,
    repo_id: str | None = None,
    token: str | None = None,
    tflite_path: str | Path = DEFAULT_TFLITE,
    release_dir: str | Path = DEFAULT_RELEASE_OUTPUT_DIR,
    golden_dir: str | Path = DEFAULT_GOLDEN_OUTPUT_DIR,
    p1_dir: str | Path = Path("fixtures/p1"),
    p2_dir: str | Path = Path("fixtures/p2"),
    tag: str | None = None,
    dotenv_path: str | Path = ".env",
) -> dict[str, Any]:
    """Upload validated release package to private HF_MODEL_REPO and tag the commit."""
    import huggingface_hub

    env_cfg = load_dotenv_settings(dotenv_path)
    resolved_repo = (repo_id or env_cfg.get("HF_MODEL_REPO", "")).strip()
    if not resolved_repo:
        raise ValueError(
            "HF_MODEL_REPO is not configured. Set HF_MODEL_REPO in .env or pass --hf-repo."
        )

    rel_dir_p = Path(release_dir)
    manifest_p = rel_dir_p / "model-manifest.json"
    manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
    validate_manifest_against_schema(manifest)
    if not manifest["shipGates"]["passed"] and manifest["channel"] != "test":
        raise ValueError("Cannot upload unverified release without channel='test'.")

    tflite_p = Path(tflite_path)
    if not tflite_p.exists():
        raise FileNotFoundError(f"TFLite file not found: {tflite_p}")
    if compute_file_sha256(tflite_p) != manifest["model"]["sha256"]:
        raise ValueError("TFLite SHA-256 does not match model-manifest.json.")

    resolved_token = (token or env_cfg.get("HF_TOKEN") or "").strip() or huggingface_hub.get_token()
    api = huggingface_hub.HfApi(token=resolved_token)
    repo_info = api.repo_info(repo_id=resolved_repo, repo_type="model")
    if not repo_info.private:
        raise ValueError(
            f"Refusing to upload to non-private HF repo {resolved_repo!r} (soul.md §3 pkt 10)."
        )

    release_tag = tag or str(manifest["modelVersion"])
    readme_src = rel_dir_p / "README_TEST_RELEASE.md"
    readme_body = readme_src.read_text(encoding="utf-8") if readme_src.exists() else f"# {release_tag}\n"

    with tempfile.TemporaryDirectory() as tmp_dir:
        stage = Path(tmp_dir)
        shutil.copy2(tflite_p, stage / manifest["model"]["file"])
        shutil.copy2(manifest_p, stage / "model-manifest.json")
        (stage / "README.md").write_text(
            _render_hf_model_card(manifest, readme_body), encoding="utf-8"
        )

        golden_src = Path(golden_dir) / "golden_images.json"
        if golden_src.exists():
            (stage / "p7_parity").mkdir(parents=True, exist_ok=True)
            shutil.copy2(golden_src, stage / "p7_parity" / "golden_images.json")

        for sub_name, src_dir in [("p1", Path(p1_dir)), ("p2", Path(p2_dir))]:
            if src_dir.exists():
                dst_dir = stage / "fixtures" / sub_name
                shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

        sum_lines: list[str] = []
        for file_p in sorted(p for p in stage.rglob("*") if p.is_file()):
            rel = file_p.relative_to(stage).as_posix()
            sum_lines.append(f"{compute_file_sha256(file_p)}  {rel}")
        (stage / "SHA256SUMS").write_text("\n".join(sum_lines) + "\n", encoding="utf-8")

        commit_info = api.upload_folder(
            folder_path=str(stage),
            repo_id=resolved_repo,
            repo_type="model",
            commit_message=f"chore(release): publish {release_tag} ({manifest['channel']})",
        )
        commit_oid = getattr(commit_info, "oid", None)
        api.create_tag(
            repo_id=resolved_repo,
            tag=release_tag,
            revision=commit_oid,
            repo_type="model",
            exist_ok=True,
        )

    logger.info("Uploaded %d files and tagged %s on %s", len(sum_lines) + 1, release_tag, resolved_repo)
    return {
        "repo_id": resolved_repo,
        "tag": release_tag,
        "commit_oid": commit_oid,
        "files_count": len(sum_lines) + 1,
        "uploaded_files": [line.split("  ", 1)[1] for line in sum_lines] + ["SHA256SUMS"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build P7 Golden Photos pack and schema-validated P9 test model-manifest.json."
    )
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--tflite", type=str, default=str(DEFAULT_TFLITE))
    parser.add_argument("--val-csv", type=str, default=str(DEFAULT_VAL_CSV))
    parser.add_argument("--image-dir", type=str, default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--coverage-csv", type=str, default=str(DEFAULT_COVERAGE_CSV))
    parser.add_argument("--schema", type=str, default=str(DEFAULT_SCHEMA_PATH))
    parser.add_argument("--ship-gates-report", type=str, default=str(DEFAULT_SHIP_GATES_REPORT))
    parser.add_argument("--quant-drift", type=str, default=str(DEFAULT_QUANT_DRIFT_REPORT))
    parser.add_argument("--calib-report", type=str, default=str(DEFAULT_CALIBRATION_REPORT))
    parser.add_argument("--export-manifest", type=str, default=str(DEFAULT_EXPORT_MANIFEST))
    parser.add_argument("--leak-report", type=str, default=str(DEFAULT_LEAK_REPORT))
    parser.add_argument("--ood-ids", type=str, default=str(DEFAULT_OOD_IDS))
    parser.add_argument("--p1-parity", type=str, default=str(DEFAULT_P1_PARITY))
    parser.add_argument("--p1-dir", type=str, default="fixtures/p1")
    parser.add_argument("--p2-dir", type=str, default="fixtures/p2")
    parser.add_argument("--golden-out", type=str, default=str(DEFAULT_GOLDEN_OUTPUT_DIR))
    parser.add_argument("--release-out", type=str, default=str(DEFAULT_RELEASE_OUTPUT_DIR))
    parser.add_argument("--model-version", type=str, default="0.1.0-rc1-test")
    parser.add_argument("--golden-count", type=int, default=240)
    parser.add_argument("--bundle-fixtures", action="store_true", help="Write p1/, p2/, and golden_images.json into release-out and hash them in SHA256SUMS")
    parser.add_argument("--upload-hf", action="store_true", help="Upload packaged release to private HF_MODEL_REPO")
    parser.add_argument("--hf-repo", type=str, default=None, help="Override HF_MODEL_REPO from .env")
    parser.add_argument("--hf-tag", type=str, default=None, help="Git tag for HF release (defaults to modelVersion)")
    parser.add_argument("--upload-only", action="store_true", help="Skip regenerating artifacts and upload existing release dir")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if not args.upload_only:
        generate_test_release_package(
            checkpoint_path=args.checkpoint,
            tflite_path=args.tflite,
            val_csv=args.val_csv,
            image_dir=args.image_dir,
            coverage_csv=args.coverage_csv,
            schema_path=args.schema,
            ship_gates_report_path=args.ship_gates_report,
            quant_drift_path=args.quant_drift,
            calibration_report_path=args.calib_report,
            export_manifest_path=args.export_manifest,
            leak_report_path=args.leak_report,
            ood_ids_path=args.ood_ids,
            p1_parity_path=args.p1_parity,
            golden_output_dir=args.golden_out,
            release_output_dir=args.release_out,
            model_version=args.model_version,
            target_golden_count=args.golden_count,
            bundle_release_tree=args.bundle_fixtures,
        )
    if args.upload_hf or args.upload_only:
        res = upload_release_to_hf(
            repo_id=args.hf_repo,
            tflite_path=args.tflite,
            release_dir=args.release_out,
            golden_dir=args.golden_out,
            p1_dir=Path(args.release_out) / "p1" if args.bundle_fixtures else Path(args.p1_dir),
            p2_dir=Path(args.release_out) / "p2" if args.bundle_fixtures else Path(args.p2_dir),
            tag=args.hf_tag,
        )
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
