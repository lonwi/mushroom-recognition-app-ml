"""Tests for training module grzybobranie_ml.train."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from grzybobranie_ml.train import (
    ModelEMA,
    MushroomSplitDataset,
    compute_class_weights,
    create_classifier_model,
    evaluate_model,
    load_dangerous_pairs,
    load_high_stakes_species,
    train_pipeline,
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
