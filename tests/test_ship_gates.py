"""Unit tests for Final Ship Gates audit module (grzybobranie_ml.ship_gates)."""

import numpy as np

from grzybobranie_ml.ship_gates import (
    compute_wilson_ci,
    evaluate_multi_photo_audit,
    evaluate_ood_test_audit,
    evaluate_single_frame_audit,
    evaluate_uncovered_species_verna,
    render_ship_gates_markdown,
)


def test_compute_wilson_ci_boundary_values() -> None:
    """Test Wilson score confidence interval boundary conditions."""
    # n = 0
    assert compute_wilson_ci(0, 0) == (0.0, 0.0)

    # 0 successes out of 100
    low_0, high_0 = compute_wilson_ci(0, 100, confidence=0.95)
    assert low_0 == 0.0
    assert 0.03 < high_0 < 0.04  # ~0.036

    # 100 successes out of 100
    low_100, high_100 = compute_wilson_ci(100, 100, confidence=0.95)
    assert 0.96 < low_100 < 0.97
    assert high_100 == 1.0

    # 50 out of 100 (symmetric around 0.5)
    low_50, high_50 = compute_wilson_ci(50, 100, confidence=0.95)
    assert np.isclose(0.5 - low_50, high_50 - 0.5, atol=1e-5)
    assert 0.39 < low_50 < 0.41
    assert 0.59 < high_50 < 0.61

    # Symmetry property: Wilson(k, n) lower == 1 - Wilson(n-k, n) upper
    k, n = 15, 80
    low_k, high_k = compute_wilson_ci(k, n)
    low_inv, high_inv = compute_wilson_ci(n - k, n)
    assert np.isclose(low_k, 1.0 - high_inv, atol=1e-5)
    assert np.isclose(high_k, 1.0 - low_inv, atol=1e-5)


def test_evaluate_single_frame_audit_synthetic() -> None:
    """Test single frame audit metrics on controlled synthetic logits."""
    # 4 classes: 0: edible_1, 1: toxic_1, 2: edible_2, 3: toxic_2
    class_to_idx = {
        "edible_1": 0,
        "toxic_1": 1,
        "edible_2": 2,
        "toxic_2": 3,
    }
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    edible_slugs = {"edible_1", "edible_2"}
    toxic_slugs = {"toxic_1", "toxic_2"}
    high_stakes_slugs = {"toxic_1", "toxic_2"}
    dangerous_pairs = [("edible_1", "toxic_1")]

    # Construct 10 samples:
    # 0..4: True toxic_1 (5 samples)
    # 5..9: True edible_1 (5 samples)
    targets = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=np.int64)

    # Logits:
    # Samples 0, 1, 2, 3: strongly predict toxic_1 [logits: [0, 10, 0, 0]]
    # Sample 4: strongly predict edible_1 (toxic misclassified as edible!) [logits: [10, 0, 0, 0]]
    # Samples 5, 6, 7, 8, 9: strongly predict edible_1 [logits: [10, 0, 0, 0]]
    tf_logits = np.zeros((10, 4), dtype=np.float32)
    tf_logits[0:4, 1] = 10.0
    tf_logits[4, 0] = 10.0
    tf_logits[5:10, 0] = 10.0

    py_logits = tf_logits.copy()

    records = [{"filepath": f"test_img_{i}.jpg"} for i in range(10)]

    res = evaluate_single_frame_audit(
        tflite_logits=tf_logits,
        py_logits=py_logits,
        targets=targets,
        records=records,
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
        edible_slugs=edible_slugs,
        toxic_slugs=toxic_slugs,
        high_stakes_slugs=high_stakes_slugs,
        dangerous_pairs=dangerous_pairs,
        temperature=1.0,
        energy_threshold=0.0,  # For logit=10, energy is -10, so accepted
        min_top1_prob=0.5,
    )

    assert res["dataset_size"] == 10
    assert res["keep_rate_tflite"] == 1.0
    assert res["tflite_agreement_overall"] == 1.0
    assert res["tflite_logit_mae"] == 0.0

    # Sample 4 is toxic_1 predicted as edible_1 -> confident toxic as edible = 1
    assert res["confident_toxic_as_edible_accepted"] == 1

    # Dangerous pair edible_1 <-> toxic_1:
    pair_res = res["dangerous_pairs"]["pairs"]["edible_1 <-> toxic_1"]
    assert pair_res["n_a"] == 5
    assert pair_res["n_b"] == 5
    # toxic_1 (b) predicted as edible_1 (a) is 1 out of 5 = 20% > 5%
    assert pair_res["confusion_b_to_a_count"] == 1
    assert pair_res["conditional_confusion_b_to_a"] == 0.2
    assert pair_res["passed"] is False
    assert "edible_1 <-> toxic_1" in res["dangerous_pairs"]["failing_pairs"]


def test_evaluate_ood_test_audit() -> None:
    """Test OOD test evaluation on synthetic data."""
    # 2 classes: 0: edible, 1: toxic
    idx_to_class = {0: "boletus_edulis", 1: "amanita_phalloides"}
    edible_slugs = {"boletus_edulis"}

    # Total 2230 records required by assert in evaluate_ood_test_audit
    n_fungi = 1290
    n_nonfungi = 940

    ood_f_records = [{"proposed_split": "test"} for _ in range(n_fungi)]
    ood_nf_records = [{"proposed_split": "test"} for _ in range(n_nonfungi)]

    # Fungi logits: negative energy, but low logits (e.g. zeros -> energy = -T*ln(2))
    # Let's say temperature = 1.0, logits = zeros((2230, 2))
    # E = -1.0 * ln(e^0 + e^0) = -ln(2) = -0.6931
    # If energy_threshold = -1.0, E (-0.6931) > -1.0, so all are rejected as OOD
    ood_f_logits = np.zeros((n_fungi, 2), dtype=np.float32)
    ood_nf_logits = np.zeros((n_nonfungi, 2), dtype=np.float32)

    res = evaluate_ood_test_audit(
        ood_fungi_logits=ood_f_logits,
        ood_fungi_records=ood_f_records,
        ood_nonfungi_logits=ood_nf_logits,
        ood_nonfungi_records=ood_nf_records,
        idx_to_class=idx_to_class,
        edible_slugs=edible_slugs,
        temperature=1.0,
        energy_threshold=-1.0,  # All rejected as OOD
        min_top1_prob=0.5,
    )

    assert res["n_samples"] == 2230
    assert res["energy_rejection_count"] == 2230
    assert res["energy_rejection_rate"] == 1.0
    assert res["accepted_count"] == 0
    assert res["confident_edible_count"] == 0
    assert res["confident_edible_rate"] == 0.0


def test_render_ship_gates_markdown() -> None:
    """render_ship_gates_markdown correctly formats table and decision."""
    report = {
        "audit_timestamp": "2026-11-24T19:00:00Z",
        "checkpoint": "path/to/checkpoint.pt",
        "tflite_model": "path/to/model.tflite",
        "tflite_sha256": "abcdef123456",
        "overall_ship_gates_pass": True,
        "gates_summary_table": [
            {
                "gate_id": "GATE-01",
                "name": "confidentToxicAsEdible",
                "target": "== 0",
                "measured": "0",
                "status": "PASS",
            }
        ],
        "single_frame_audit": {
            "dataset_size": 100,
            "accuracy_raw_pytorch": 0.85,
            "accuracy_raw_tflite": 0.85,
            "tflite_agreement_overall": 1.0,
            "tflite_agreement_high_stakes": 1.0,
            "tflite_logit_mae": 0.01,
            "tflite_max_logit_error": 0.05,
            "ece_tflite": 0.02,
            "mce_tflite": 0.05,
            "keep_rate_tflite": 0.82,
            "accuracy_conditional_tflite": 0.92,
            "confident_toxic_as_edible_accepted": 0,
            "dangerous_pairs": {
                "failing_pairs_count": 1,
                "total_active_pairs": 1,
                "pairs": {
                    "boletus_edulis <-> rubroboletus_satanas": {
                        "n_a": 10,
                        "n_b": 10,
                        "n_a_accepted": 8,
                        "n_b_accepted": 8,
                        "conditional_confusion_a_to_b": 0.0,
                        "conditional_confusion_b_to_a": 0.125,
                        "confusion_a_to_b_count": 0,
                        "confusion_b_to_a_count": 1,
                        "max_confusion": 0.125,
                        "passed": False,
                    }
                },
            },
            "high_stakes_recall": {"min_conditional_recall": 0.90, "species": {}},
        },
        "ood_test_audit": {
            "n_samples": 2230,
            "n_fungi_test": 1290,
            "n_nonfungi_test": 940,
            "energy_rejection_rate": 0.65,
            "confident_edible_rate": 0.015,
            "confident_edible_rate_pct": 1.5,
            "wilson_ci_95": [0.01, 0.02],
            "wilson_upper_bound_95": 0.02,
        },
        "uncovered_amanita_verna_audit": {
            "n_samples": 11,
            "energy_rejected": 5,
            "unclear_rejected": 0,
            "accepted": 6,
            "confident_toxic_as_edible": 0,
            "passed": True,
        },
        "multi_photo_audit": {
            "n_multi_observations": 100,
            "n_multi_frames_total": 250,
            "conditional_accuracy": 0.95,
            "conflict_detected_count": 5,
            "high_stakes_veto_count": 2,
            "fused_toxic_as_edible_count": 0,
            "passed": True,
        },
    }

    md = render_ship_gates_markdown(report)
    assert "# Finalny Audyt Ship Gates" in md
    assert "GATE-01" in md
    assert "ZALICZONY (PASS)" in md
    assert "Tabela Par Niebezpiecznych (DANGEROUS_PAIRS)" in md
    assert "boletus_edulis <-> rubroboletus_satanas" in md


def test_evaluate_uncovered_species_verna_synthetic() -> None:
    """Test uncovered deadly species audit with synthetic logits."""
    idx_to_class = {0: "boletus_edulis", 1: "amanita_phalloides"}
    edible_slugs = {"boletus_edulis"}

    # 3 samples:
    # 0: high energy -> OOD
    # 1: predicts amanita_phalloides (toxic) -> accepted, but not edible
    # 2: predicts boletus_edulis (edible) -> accepted, confident toxic as edible!
    logits = np.array([
        [0.0, 0.0],     # energy > -2.2 -> rejected
        [0.0, 10.0],    # toxic
        [10.0, 0.0],    # edible
    ], dtype=np.float32)

    res = evaluate_uncovered_species_verna(
        verna_logits=logits,
        idx_to_class=idx_to_class,
        edible_slugs=edible_slugs,
        temperature=1.0,
        energy_threshold=-2.2,
        min_top1_prob=0.5,
    )

    assert res["n_samples"] == 3
    assert res["energy_rejected"] == 1
    assert res["accepted"] == 2
    assert res["confident_toxic_as_edible"] == 1
    assert res["passed"] is False


def test_evaluate_multi_photo_audit_synthetic() -> None:
    """Test multi-photo audit on synthetic observations: OK, conflict, and veto cases."""
    class_to_idx = {
        "macrolepiota_procera": 0,      # Edible
        "amanita_phalloides": 1,        # Toxic / High stakes
        "boletus_edulis": 2,            # Edible
    }
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    edible_slugs = {"macrolepiota_procera", "boletus_edulis"}
    toxic_slugs = {"amanita_phalloides"}
    high_stakes_slugs = {"amanita_phalloides"}

    # Construct 3 observations with 2 frames each (total 6 frames):
    # Obs 1 (frames 0, 1): Both predict macrolepiota_procera strongly -> OK
    # Obs 2 (frames 2, 3): Frame 2 predicts macrolepiota, Frame 3 predicts boletus -> CONFLICT
    # Obs 3 (frames 4, 5): True amanita_phalloides, both frames predict edible macrolepiota -> Toxic As Edible
    records = [
        {"observation_id": "obs_1", "species_slug": "macrolepiota_procera"},
        {"observation_id": "obs_1", "species_slug": "macrolepiota_procera"},
        {"observation_id": "obs_2", "species_slug": "macrolepiota_procera"},
        {"observation_id": "obs_2", "species_slug": "macrolepiota_procera"},
        {"observation_id": "obs_3", "species_slug": "amanita_phalloides"},
        {"observation_id": "obs_3", "species_slug": "amanita_phalloides"},
    ]

    logits = np.zeros((6, 3), dtype=np.float32)
    # Obs 1
    logits[0, 0] = 10.0
    logits[1, 0] = 10.0
    # Obs 2 (conflict between class 0 and class 2)
    logits[2, 0] = 10.0
    logits[3, 2] = 10.0
    # Obs 3 (toxic truth, predicted edible class 0)
    logits[4, 0] = 10.0
    logits[5, 0] = 10.0

    res = evaluate_multi_photo_audit(
        records=records,
        logits=logits,
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
        edible_slugs=edible_slugs,
        toxic_slugs=toxic_slugs,
        high_stakes_slugs=high_stakes_slugs,
        temperature_frame=1.0,
        temperature_multi=1.0,
        energy_threshold_frame=0.0,
        min_top1_prob=0.5,
        disagreement_threshold=0.7,
        veto_threshold=0.2,
    )

    assert res["n_multi_observations"] == 3
    assert res["conflict_detected_count"] == 1
    assert res["accepted_count"] == 2
    assert res["correct_count"] == 1
    assert res["fused_toxic_as_edible_count"] == 1
    assert res["passed"] is False
