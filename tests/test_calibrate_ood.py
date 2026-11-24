"""Unit tests for Stage P4 calibration and OOD energy gate module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from grzybobranie_ml.calibrate_ood import (
    OODManifestDataset,
    bootstrap_ece_ci,
    compute_auroc_and_fpr,
    compute_ece_mce,
    compute_energy_vector,
    compute_file_sha256,
    compute_nll,
    fit_temperature_scaling,
    render_reliability_diagram,
)
from grzybobranie_ml.recognition_math import energy as ref_single_energy


def test_temperature_scaling_optimization() -> None:
    """Test that temperature scaling decreases NLL on under-confident synthetic logits."""
    rng = np.random.default_rng(42)
    n = 200
    k = 5
    # Generate true labels
    targets = rng.integers(0, k, size=n)
    # Generate logits scaled down (under-confident)
    logits = rng.standard_normal((n, k)) * 0.5
    for i in range(n):
        logits[i, targets[i]] += 2.0

    nll_raw = compute_nll(logits, targets, temperature=1.0)
    t_opt = fit_temperature_scaling(logits, targets, initial_t=1.0)
    nll_cal = compute_nll(logits, targets, temperature=t_opt)

    assert t_opt > 0.0
    assert nll_cal <= nll_raw + 1e-6
    # Logits were under-confident, so optimal T should be < 1.0
    assert t_opt < 1.0


def test_compute_ece_mce_perfect() -> None:
    """Test ECE on perfectly confident and 100% accurate synthetic predictions."""
    n = 100
    k = 3
    targets = np.zeros(n, dtype=np.int64)
    # Very high logits for true class -> confidence ~ 1.0, accuracy = 1.0
    logits = np.zeros((n, k), dtype=np.float64)
    logits[:, 0] = 30.0

    ece, mce, bin_stats = compute_ece_mce(logits, targets, temperature=1.0, n_bins=15)
    assert ece == pytest.approx(0.0, abs=1e-4)
    assert mce == pytest.approx(0.0, abs=1e-4)
    assert len(bin_stats) == 15


def test_compute_ece_mce_properties() -> None:
    """Test general properties of ECE and MCE: non-negative, MCE >= ECE, bin bounds."""
    rng = np.random.default_rng(123)
    n = 300
    k = 4
    logits = rng.standard_normal((n, k))
    targets = rng.integers(0, k, size=n)

    ece, mce, bin_stats = compute_ece_mce(logits, targets, temperature=1.0, n_bins=10)
    assert 0.0 <= ece <= 1.0
    assert 0.0 <= mce <= 1.0
    assert mce >= ece - 1e-7

    total_count = sum(b["count"] for b in bin_stats)
    assert total_count == n


def test_bootstrap_ece_ci() -> None:
    """Test 95% bootstrap confidence interval bounds for ECE."""
    rng = np.random.default_rng(456)
    n = 150
    k = 4
    logits = rng.standard_normal((n, k))
    targets = rng.integers(0, k, size=n)

    ece, _, _ = compute_ece_mce(logits, targets, temperature=1.0, n_bins=10)
    low_ci, high_ci = bootstrap_ece_ci(logits, targets, temperature=1.0, n_bins=10, n_resamples=100, seed=42)

    assert 0.0 <= low_ci <= high_ci <= 1.0
    # True ECE should generally lie near or within the bootstrap interval
    assert low_ci <= ece + 0.05
    assert high_ci >= ece - 0.05


def test_compute_energy_vector_parity() -> None:
    """Test that compute_energy_vector matches grzybobranie_ml.recognition_math.energy."""
    rng = np.random.default_rng(789)
    logits = rng.standard_normal((25, 10))
    for t in [0.75, 1.0, 1.5]:
        vec_energies = compute_energy_vector(logits, temperature=t)
        for i in range(len(logits)):
            single_e = ref_single_energy(logits[i], T=t)
            assert vec_energies[i] == pytest.approx(single_e, abs=1e-5)


def test_compute_auroc_and_fpr_separable() -> None:
    """Test AUROC and FPR on linearly separable energy distributions."""
    # In-distribution has low energy (-10 to -5)
    id_e = np.linspace(-10.0, -5.0, 100)
    # OOD has high energy (0 to 5)
    ood_e = np.linspace(0.0, 5.0, 100)

    auroc, fpr, threshold = compute_auroc_and_fpr(id_e, ood_e, target_id_tpr=0.95)
    assert auroc == pytest.approx(1.0, abs=1e-5)
    assert fpr == pytest.approx(0.0, abs=1e-5)
    assert threshold < 0.0


def test_compute_auroc_and_fpr_random() -> None:
    """Test AUROC on identical distributions (should be close to 0.5)."""
    rng = np.random.default_rng(999)
    id_e = rng.standard_normal(500)
    ood_e = rng.standard_normal(500)

    auroc, fpr, _ = compute_auroc_and_fpr(id_e, ood_e, target_id_tpr=0.95)
    assert 0.40 <= auroc <= 0.60
    # At 95% TPR, FPR on identical distribution should be ~ 95%
    assert fpr == pytest.approx(0.95, abs=0.05)


def test_render_reliability_diagram() -> None:
    """Test that render_reliability_diagram creates a valid PNG image."""
    raw_bins = [{"count": 10, "acc": 0.5, "conf": 0.6} for _ in range(15)]
    cal_bins = [{"count": 10, "acc": 0.6, "conf": 0.6} for _ in range(15)]

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_file = Path(tmp_dir) / "test_diagram.png"
        render_reliability_diagram(
            raw_bin_stats=raw_bins,
            cal_bin_stats=cal_bins,
            raw_ece=0.10,
            cal_ece=0.02,
            raw_mce=0.15,
            cal_mce=0.04,
            temperature=0.85,
            save_path=save_file,
        )

        assert save_file.exists()
        assert save_file.stat().st_size > 1000
        with Image.open(save_file) as img:
            assert img.size == (1200, 700)
            assert img.format == "PNG"


def test_amanita_verna_slice_exclusion() -> None:
    """Test slicing out class at index 12 from a 67-class tensor."""
    batch_size = 4
    tensor67 = torch.randn(batch_size, 67)
    tensor67[:, 12] = 999.0  # high value at index 12

    sliced66 = torch.cat([tensor67[:, :12], tensor67[:, 13:]], dim=1)
    assert sliced66.shape == (batch_size, 66)
    assert not torch.isclose(sliced66, torch.tensor(999.0)).any()


def test_temperature_scaling_overconfident() -> None:
    """Test that temperature scaling decreases NLL and finds T > 1.0 on over-confident synthetic logits."""
    rng = np.random.default_rng(101)
    n = 300
    k = 4
    targets = rng.integers(0, k, size=n)
    # Logits scaled up (over-confident)
    logits = rng.standard_normal((n, k)) * 10.0
    for i in range(n):
        logits[i, targets[i]] += 5.0

    nll_raw = compute_nll(logits, targets, temperature=1.0)
    t_opt = fit_temperature_scaling(logits, targets, initial_t=1.0)
    nll_cal = compute_nll(logits, targets, temperature=t_opt)

    assert t_opt > 1.0
    assert nll_cal < nll_raw


def test_compute_ece_mce_empty_and_single() -> None:
    """Test robustness of ECE/MCE and bootstrap CI on empty and single-element inputs."""
    empty_logits = np.empty((0, 5), dtype=np.float64)
    empty_targets = np.empty(0, dtype=np.int64)

    ece, mce, bins = compute_ece_mce(empty_logits, empty_targets, 1.0, 15)
    assert ece == 0.0
    assert mce == 0.0
    assert len(bins) == 15

    ci_low, ci_high = bootstrap_ece_ci(empty_logits, empty_targets, 1.0, 15)
    assert ci_low == 0.0
    assert ci_high == 0.0

    single_logits = np.array([[10.0, 0.0, 0.0]])
    single_targets = np.array([0], dtype=np.int64)
    ece1, mce1, _ = compute_ece_mce(single_logits, single_targets, 1.0, 15)
    assert 0.0 <= ece1 <= 1.0
    assert 0.0 <= mce1 <= 1.0
    ci1_low, ci1_high = bootstrap_ece_ci(single_logits, single_targets, 1.0, 15)
    assert ci1_low == ci1_high == ece1


def test_compute_file_sha256() -> None:
    """Test compute_file_sha256 matches known sha256."""
    import hashlib

    with tempfile.TemporaryDirectory() as tmp_dir:
        test_file = Path(tmp_dir) / "test.txt"
        content = b"Grzybobranie AI Stage P4 Test"
        test_file.write_bytes(content)

        expected = hashlib.sha256(content).hexdigest()
        assert compute_file_sha256(test_file) == expected


def test_compute_auroc_and_fpr_keep_rate_guarantee() -> None:
    """Test that compute_auroc_and_fpr guarantees >= 95% keep rate on ID."""
    rng = np.random.default_rng(202)
    id_e = rng.normal(-3.5, 1.0, 13065)
    ood_e = rng.normal(-2.0, 1.0, 4460)

    auroc, fpr, threshold = compute_auroc_and_fpr(id_e, ood_e, target_id_tpr=0.95)
    keep_rate = float(np.mean(id_e <= threshold))
    assert keep_rate >= 0.95
    assert 0.0 <= fpr <= 1.0
    assert 0.5 <= auroc <= 1.0


def test_ood_manifest_dataset_and_transform_fallback() -> None:
    """Test OODManifestDataset loading and fallback to ToTensor when transform is None."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_p = Path(tmp_dir)
        img_p = tmp_p / "sample.jpg"
        img = Image.new("RGB", (32, 32), color=(100, 150, 200))
        img.save(img_p)

        csv_p = tmp_p / "ood.csv"
        csv_p.write_text("photo_id,species_slug\n123,dummy_slug\n", encoding="utf-8")

        manifest_p = tmp_p / "manifest_768.csv"
        manifest_p.write_text("photo_id,path\n123,sample.jpg\n", encoding="utf-8")

        # Test with transform=None -> should return torch.Tensor
        ds = OODManifestDataset(csv_p, image_dir=tmp_p, transform=None)
        assert len(ds) == 1
        tensor, idx = ds[0]
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 32, 32)
        assert idx == 0
