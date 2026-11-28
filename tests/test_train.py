"""Tests for training module grzybobranie_ml.train."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader

from grzybobranie_ml.train import (
    EnergyBoundedOELoss,
    ModelEMA,
    MushroomSplitDataset,
    SafetyCrossEntropyLoss,
    compute_class_weights,
    create_classifier_model,
    evaluate_model,
    get_train_transforms,
    get_val_transforms,
    load_dangerous_pairs,
    load_high_stakes_species,
    train_pipeline,
    validate_oe_manifest_disjointness,
    verify_model_license,
)


def test_verify_model_license_approved() -> None:
    # Approved Apache-2.0 models
    verify_model_license("mobilenetv4_conv_small.e2400_r224_in1k")
    verify_model_license("tf_efficientnet_lite0.in1k")
    verify_model_license("mobilenetv3_large_100.ra4_e3600_r224_in1k")


def test_verify_model_license_disallowed() -> None:
    # Disallowed models must raise ValueError
    with pytest.raises(ValueError, match="ConvNeXt V2"):
        verify_model_license("convnextv2_nano.fcmae_ft_in22k_in1k")

    with pytest.raises(ValueError, match="Apple"):
        verify_model_license("mobilevitv2_100.cvnets_in1k")

    with pytest.raises(ValueError, match="DF20"):
        verify_model_license("timm_resnet50_ft_df20")


def test_create_classifier_model_head() -> None:
    model = create_classifier_model("mobilenetv4_conv_small.e2400_r224_in1k", num_classes=67, pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 67)


def test_class_weights() -> None:
    records = [
        {"species_slug": "common_species"},
        {"species_slug": "common_species"},
        {"species_slug": "common_species"},
        {"species_slug": "rare_species"},
    ]
    class_to_idx = {"common_species": 0, "rare_species": 1}
    weights = compute_class_weights(records, class_to_idx)
    assert len(weights) == 2
    # Rare species weight must be higher than common species weight
    assert weights[1] > weights[0]
    assert (weights > 0).all()


def test_load_metadata_configs() -> None:
    pairs = load_dangerous_pairs()
    assert len(pairs) == 44
    for p in pairs:
        assert len(p) == 2
        assert isinstance(p[0], str) and isinstance(p[1], str)

    high_stakes = load_high_stakes_species()
    assert len(high_stakes) == 27
    assert "amanita_phalloides" in high_stakes
    assert "amanita_verna" in high_stakes


def test_model_ema() -> None:
    model = torch.nn.Linear(10, 5)
    ema = ModelEMA(model, decay=0.9)
    # Change model weight
    with torch.no_grad():
        model.weight.add_(1.0)
    ema.update(model)
    # EMA weight should have moved towards model weight
    assert not torch.allclose(ema.module.weight, model.weight)


def test_dataset_mock_loader(tmp_path: Path) -> None:
    csv_file = tmp_path / "mock_split.csv"
    csv_file.write_text(
        "photo_id,species_slug,image_url\n"
        "p1,boletus_edulis,https://example.com/1.jpg\n"
        "p2,amanita_phalloides,https://example.com/2.jpg\n",
        encoding="utf-8",
    )

    ds = MushroomSplitDataset(csv_path=csv_file, mock_if_missing=True)
    assert len(ds) == 2
    tensor, label = ds[0]
    assert tensor.shape == (3, 256, 256)
    assert isinstance(label, int)


def test_evaluate_model_mock() -> None:
    class DummyModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b = x.size(0)
            logits = torch.zeros(b, 2)
            logits[:, 0] = 5.0  # always predicts class 0
            return logits

    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return 4

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
            # Targets: 0, 0, 1, 1
            target = 0 if idx < 2 else 1
            return torch.zeros(3, 32, 32), target

    loader = DataLoader(DummyDataset(), batch_size=2)
    class_to_idx = {"amanita_phalloides": 0, "boletus_edulis": 1}
    high_stakes = {"amanita_phalloides"}
    pairs = [("amanita_phalloides", "boletus_edulis")]

    metrics = evaluate_model(
        DummyModel(),
        loader,
        torch.device("cpu"),
        class_to_idx,
        dangerous_pairs=pairs,
        high_stakes_species=high_stakes,
        use_amp=False,
    )

    assert metrics["val_samples"] == 4
    assert metrics["overall_top1"] == 0.5
    # Class 0 recall = 1.0, Class 1 recall = 0.0 -> macro = 0.5
    assert metrics["macro_top1"] == 0.5
    assert metrics["high_stakes_min_recall"] == 1.0
    assert metrics["dangerous_pairs_evaluated"] == 1


def test_smoke_train_pipeline(tmp_path: Path) -> None:
    # Create small synthetic train and val CSVs
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    content = (
        "photo_id,species_slug,image_url\n"
        "p1,amanita_phalloides,https://example.com/1.jpg\n"
        "p2,boletus_edulis,https://example.com/2.jpg\n"
    )
    train_csv.write_text(content, encoding="utf-8")
    val_csv.write_text(content, encoding="utf-8")

    summary = train_pipeline(
        model_name="mobilenetv4_conv_small.e2400_r224_in1k",
        input_size=224,
        train_csv=train_csv,
        val_csv=val_csv,
        output_dir=tmp_path / "runs",
        warmup_epochs=1,
        fine_tune_epochs=1,
        batch_size=2,
        mock_dataset=True,
        num_workers=0,
        device="cpu",
        pretrained=False,
    )

    assert summary["total_epochs"] == 2
    assert Path(summary["best_checkpoint_file"]).exists()
    assert len(summary["best_checkpoint_sha256"]) == 64


def test_val_transforms_none_stretch_preserves_full_frame() -> None:
    """Default none_stretch must preserve outer edges that legacy_448_center cropped out."""
    arr = np.zeros((480, 640, 3), dtype=np.uint8)
    arr[:, :60] = (255, 0, 0)  # red bar on far left edge, outside central 256x256 crop
    img = Image.fromarray(arr, mode="RGB")

    t_none = get_val_transforms(256, crop_mode="none_stretch")(img)
    t_bilinear = get_val_transforms(256, crop_mode="bilinear_none_stretch")(img)
    t_center256 = get_val_transforms(256, crop_mode="center_256")(img)
    t_legacy = get_val_transforms(256, crop_mode="legacy_448_center")(img)

    for t in (t_none, t_bilinear, t_center256, t_legacy):
        assert t.shape == (3, 256, 256)

    # Full-frame none_stretch variants preserve the red left edge (red channel > 0.5)
    assert float(t_none[0, 128, 2]) > 0.5
    assert float(t_bilinear[0, 128, 2]) > 0.5
    # Legacy 448 center crop loses the outer edge completely (red channel == -1.0)
    assert float(t_legacy[0, 128, 2]) == pytest.approx(-1.0, abs=1e-5)

    with pytest.raises(ValueError, match="Unsupported val crop_mode"):
        get_val_transforms(256, crop_mode="invalid_mode")


def test_train_transforms_v2_none_stretch_and_saturation_jitter() -> None:
    tf = get_train_transforms(input_size=256, crop_mode="none_stretch", saturation_jitter=0.05)
    has_resize = any(isinstance(op, T.Resize) and op.size == (256, 256) for op in tf.transforms)
    has_flip = any(isinstance(op, T.RandomHorizontalFlip) and op.p == 0.5 for op in tf.transforms)
    affines = [op for op in tf.transforms if isinstance(op, T.RandomAffine)]
    jitters = [op for op in tf.transforms if isinstance(op, T.ColorJitter)]
    assert has_resize
    assert has_flip
    assert len(affines) == 1
    assert affines[0].translate == (0.05, 0.05)
    assert affines[0].scale == (0.9, 1.1)
    assert len(jitters) == 1
    assert jitters[0].saturation == (0.95, 1.05)


def test_hybrid_medium_v2_architecture_approved() -> None:
    verify_model_license("mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k")
    model = create_classifier_model(
        "mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k",
        num_classes=66,
        pretrained=False,
    )
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 256, 256))
    assert out.shape == (1, 66)


def test_safety_cross_entropy_loss_asymmetry() -> None:
    class_to_idx = {
        "kuehneromyces_mutabilis": 0,  # edible
        "galerina_marginata": 1,       # deadly twin
        "boletus_edulis": 2,           # edible
    }
    pairs = [("kuehneromyces_mutabilis", "galerina_marginata")]
    edible = {"kuehneromyces_mutabilis", "boletus_edulis"}
    toxic = {"galerina_marginata"}

    loss_zero = SafetyCrossEntropyLoss(
        class_to_idx=class_to_idx,
        dangerous_pairs=pairs,
        edible_slugs=edible,
        toxic_slugs=toxic,
        label_smoothing=0.0,
        safety_lambda=0.0,
    )
    loss_safe = SafetyCrossEntropyLoss(
        class_to_idx=class_to_idx,
        dangerous_pairs=pairs,
        edible_slugs=edible,
        toxic_slugs=toxic,
        label_smoothing=0.0,
        safety_lambda=1.0,
    )

    # 1. Edible sample: safety penalty must be identically 0 (does not push edible toward false alarms)
    edible_logits = torch.tensor([[1.0, 2.0, 0.0]], dtype=torch.float32)
    edible_target = torch.tensor([0], dtype=torch.long)
    assert torch.isclose(loss_safe(edible_logits, edible_target), loss_zero(edible_logits, edible_target))

    # 2. Toxic sample misclassified toward its edible twin: loss_safe must be strictly higher
    toxic_logits = torch.tensor([[4.0, 0.0, 0.0]], dtype=torch.float32)
    toxic_target = torch.tensor([1], dtype=torch.long)
    assert loss_safe(toxic_logits, toxic_target) > loss_zero(toxic_logits, toxic_target)


def test_energy_bounded_oe_loss_and_disjointness(tmp_path: Path) -> None:
    oe_loss = EnergyBoundedOELoss(margin_in=-3.5, margin_out=-1.0, temperature=1.0)
    in_logits = torch.tensor([[6.0, 0.0, 0.0]], dtype=torch.float32)  # E ~ -6.0 <= -3.5
    oe_logits = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)  # E = -ln(3) ~ -1.1 < -1.0
    assert float(oe_loss(in_logits, oe_logits)) > 0.0

    bad_ood_path = tmp_path / "ood-v1" / "oe.csv"
    bad_ood_path.parent.mkdir()
    bad_ood_path.write_text("photo_id,license,image_url\nx,cc0,https://example.com/x.jpg\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ood-v1"):
        validate_oe_manifest_disjointness(bad_ood_path, ood_ids_path=tmp_path / "none.txt", excluded_split_csvs=())

    ood_ids = tmp_path / "ood_ids.txt"
    ood_ids.write_text("photo_id\tp_overlap\n", encoding="utf-8")
    mock_test = tmp_path / "test.csv"
    mock_test.write_text("photo_id,image_url\np_test,https://example.com/t.jpg\n", encoding="utf-8")

    overlap_csv = tmp_path / "oe_overlap.csv"
    overlap_csv.write_text("photo_id,license,image_url\np_overlap,cc0-1.0,https://example.com/o.jpg\n", encoding="utf-8")
    with pytest.raises(ValueError, match="OE overlap"):
        validate_oe_manifest_disjointness(overlap_csv, ood_ids_path=ood_ids, excluded_split_csvs=(mock_test,))

    non_cc0 = tmp_path / "oe_by.csv"
    non_cc0.write_text("photo_id,license,image_url\np_clean,cc-by-4.0,https://example.com/c.jpg\n", encoding="utf-8")
    with pytest.raises(ValueError, match="strictly CC0"):
        validate_oe_manifest_disjointness(non_cc0, ood_ids_path=ood_ids, excluded_split_csvs=(mock_test,))

    fake_cc0 = tmp_path / "oe_non_cc0.csv"
    fake_cc0.write_text("photo_id,license,image_url\np_clean,non-cc0,https://example.com/c.jpg\n", encoding="utf-8")
    with pytest.raises(ValueError, match="strictly CC0"):
        validate_oe_manifest_disjointness(fake_cc0, ood_ids_path=ood_ids, excluded_split_csvs=(mock_test,))

    clean_csv = tmp_path / "oe_clean.csv"
    clean_csv.write_text("photo_id,license,image_url\np_clean,cc0-1.0,https://example.com/c.jpg\n", encoding="utf-8")
    assert validate_oe_manifest_disjointness(clean_csv, ood_ids_path=ood_ids, excluded_split_csvs=(mock_test,)) == 1


def test_smoke_train_pipeline_v2_with_safety_oe_and_uncovered(tmp_path: Path) -> None:
    from grzybobranie_ml.train import resolve_val_crop_mode

    assert resolve_val_crop_mode("legacy_rrc") == "legacy_448_center"
    assert resolve_val_crop_mode("none_stretch") == "none_stretch"

    cov_csv = tmp_path / "coverage.csv"
    cov_csv.write_text(
        "class,high_stakes,prog\n"
        "boletus_edulis,false,2\n"
        "galerina_marginata,true,2\n"
        "amanita_verna,true,55\n",
        encoding="utf-8",
    )
    train_csv = tmp_path / "train.csv"
    train_csv.write_text(
        "photo_id,species_slug,image_url\n"
        "b1,boletus_edulis,https://example.com/b1.jpg\n"
        "b2,boletus_edulis,https://example.com/b2.jpg\n"
        "g1,galerina_marginata,https://example.com/g1.jpg\n"
        "g2,galerina_marginata,https://example.com/g2.jpg\n"
        "v1,amanita_verna,https://example.com/v1.jpg\n",
        encoding="utf-8",
    )
    val_csv = tmp_path / "val.csv"
    val_csv.write_text(
        "photo_id,species_slug,image_url\n"
        "vb1,boletus_edulis,https://example.com/vb1.jpg\n"
        "vg1,galerina_marginata,https://example.com/vg1.jpg\n"
        "vv1,amanita_verna,https://example.com/vv1.jpg\n",
        encoding="utf-8",
    )
    oe_csv = tmp_path / "oe_cc0.csv"
    oe_csv.write_text(
        "photo_id,license,image_url\n"
        "oe1,cc0-1.0,https://example.com/oe1.jpg\n"
        "oe2,cc0,https://example.com/oe2.jpg\n",
        encoding="utf-8",
    )

    summary = train_pipeline(
        model_name="mobilenetv4_conv_small.e2400_r224_in1k",
        input_size=224,
        train_csv=train_csv,
        val_csv=val_csv,
        output_dir=tmp_path / "runs",
        warmup_epochs=1,
        fine_tune_epochs=1,
        batch_size=2,
        mock_dataset=True,
        num_workers=0,
        device="cpu",
        pretrained=False,
        crop_mode="none_stretch",
        saturation_jitter=0.05,
        safety_lambda=0.5,
        oe_csv=oe_csv,
        oe_lambda=0.1,
        exclude_uncovered=True,
        coverage_csv=cov_csv,
        ood_ids_path=tmp_path / "missing_ood_ids.txt",
    )

    assert summary["num_classes"] == 2
    assert summary["excluded_species"] == ["amanita_verna"]
    assert summary["oe_lambda"] == pytest.approx(0.1)
    ckpt = torch.load(summary["best_checkpoint_file"], map_location="cpu", weights_only=False)
    assert "amanita_verna" not in ckpt["class_to_idx"]
    assert ckpt["crop_mode"] == "none_stretch"
