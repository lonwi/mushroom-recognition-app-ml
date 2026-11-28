"""Unit tests for Step 1 preprocessing ablation module (grzybobranie_ml.ablate_preprocessing)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from grzybobranie_ml.ablate_preprocessing import (
    CalibOnlyOODDataset,
    _assert_not_test_split,
    evaluate_variant_metrics,
    render_ablation_markdown,
)


def test_assert_not_test_split_blocks_test_csv(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Refusing to evaluate test split"):
        _assert_not_test_split(tmp_path / "test.csv")
    _assert_not_test_split(tmp_path / "val.csv")


def test_calib_only_ood_dataset_ignores_test_rows(tmp_path: Path) -> None:
    calib_img = tmp_path / "calib_only.jpg"
    Image.new("RGB", (32, 32), color=(10, 120, 30)).save(calib_img)

    manifest = tmp_path / "manifest_768.csv"
    manifest.write_text("photo_id,path\nc1,calib_only.jpg\n", encoding="utf-8")

    # Note: test row points to a non-existent photo; CalibOnlyOODDataset must skip it completely
    ood_csv = tmp_path / "ood.csv"
    ood_csv.write_text(
        "photo_id,proposed_split,image_url\n"
        "c1,calib,https://example.com/c1.jpg\n"
        "t1,test,https://example.com/missing_test.jpg\n",
        encoding="utf-8",
    )

    ds = CalibOnlyOODDataset(ood_csv, image_dir=tmp_path)
    assert len(ds) == 1
    assert ds.records[0]["photo_id"] == "c1"


def test_evaluate_variant_metrics_and_markdown() -> None:
    class_to_idx = {"boletus_edulis": 0, "rubroboletus_satanas": 1}
    idx_to_class = {0: "boletus_edulis", 1: "rubroboletus_satanas"}
    edible = {"boletus_edulis"}
    toxic = {"rubroboletus_satanas"}
    hs = {"rubroboletus_satanas"}
    pairs = [("boletus_edulis", "rubroboletus_satanas")]

    val_logits = np.array([
        [6.0, 0.0],
        [6.0, 0.0],
        [0.0, 6.0],
        [3.0, 2.0],  # toxic sample with edible top1 and p_toxic ~ 0.22 (caught by veto)
    ], dtype=np.float64)
    val_targets = np.array([0, 0, 1, 1], dtype=np.int64)
    val_records = [
        {"observation_id": "o1", "species_slug": "boletus_edulis"},
        {"observation_id": "o2", "species_slug": "boletus_edulis"},
        {"observation_id": "o3", "species_slug": "rubroboletus_satanas"},
        {"observation_id": "o4", "species_slug": "rubroboletus_satanas"},
    ]
    verna_logits = np.array([[0.0, 0.0]], dtype=np.float64)
    ood_f = np.array([[0.0, 0.0], [5.0, 0.0]], dtype=np.float64)
    ood_nf = np.array([[0.0, 0.0]], dtype=np.float64)

    res = evaluate_variant_metrics(
        val_logits=val_logits,
        val_targets=val_targets,
        val_records=val_records,
        verna_logits=verna_logits,
        ood_calib_fungi_logits=ood_f,
        ood_calib_nonfungi_logits=ood_nf,
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
        edible_slugs=edible,
        toxic_slugs=toxic,
        high_stakes_slugs=hs,
        dangerous_pairs=pairs,
        tflite_val_logits=val_logits.copy(),
    )

    assert res["val_samples"] == 4
    assert res["overall_top1_raw"] == pytest.approx(0.75)
    assert "boletus_edulis <-> rubroboletus_satanas" in res["dangerous_pairs"]["all_pairs"]
    assert res["ood_calib_frozen"]["total"]["n_samples"] == 3
    assert res["ood_calib_frozen"]["total"]["confident_edible_count"] == 1
    assert res["p5_veto_comparison"]["asymmetric_twin_veto_010_5x"]["high_stakes_veto_count"] >= 1
    assert res["tflite_fp16_parity"]["top1_agreement_overall"] == pytest.approx(1.0)

    import copy

    res_p1 = copy.deepcopy(res)
    res_p1["dangerous_pairs"]["all_pairs"]["calocybe_gambosa <-> entoloma_sinuatum"] = {
        "n_a": 100,
        "n_b": 40,
        "n_a_accepted": 90,
        "n_b_accepted": 35,
        "raw_max_err": 0.15,
        "cond_err_a_to_b": 0.0,
        "cond_err_b_to_a": 0.18,
        "cond_max_err": 0.18,
    }
    res_p1["dangerous_pairs"]["worst_pairs"] = {
        "calocybe_gambosa <-> entoloma_sinuatum": res_p1["dangerous_pairs"]["all_pairs"][
            "calocybe_gambosa <-> entoloma_sinuatum"
        ]
    }
    report = {
        "timestamp": "2026-11-25T08:00:00Z",
        "checkpoint": "ckpt.pt",
        "checkpoint_sha256": "a" * 64,
        "tflite_model": "model.tflite",
        "tflite_sha256": "b" * 64,
        "variants": {
            "legacy_448_center": res,
            "p1_box_none_stretch": res_p1,
        },
    }
    md = render_ablation_markdown(report)
    assert "legacy_448_center" in md
    assert "p1_box_none_stretch" in md
    assert "Asymmetric Twin Veto" in md
    assert "@85% TPR" in md
    assert "calocybe_gambosa <-> entoloma_sinuatum" in md
