"""Generate synthetic P2 v1 decision and fusion fixtures.

Write ``fixtures/p2/decision_cases.json``, ``fixtures/p2/fusion_cases.json``
and ``fixtures/p2/SHA256SUMS``. All logits are synthetic (not from a trained
model or real photos). Expected values are computed by
``grzybobranie_ml.recognition_math``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from grzybobranie_ml.recognition_math import (
    STATUS_OK,
    STATUS_OOD,
    decide_single,
    energy,
    fuse,
)

FIXTURE_SCHEMA_VERSION = "1.0.1"
GENERATOR = "grzybobranie_ml.make_p2_fixtures"
TOLERANCE = 1e-5
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "fixtures" / "p2"

# Shared synthetic head (8 classes). Names are labels only — not real taxa.
CLASS_IDS = [f"syn_class_{i}" for i in range(8)]

# Default single-photo thresholds used unless a case overrides them.
DEFAULT_ENERGY_THRESHOLD = -6.0
DEFAULT_MIN_TOP1 = 0.40
DEFAULT_FUSION = {
    "2": {"energyThreshold": -6.3, "minTop1Prob": 0.45},
    "3": {"energyThreshold": -6.5, "minTop1Prob": 0.50},
}


def _dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2) + "\n"


def _probs_to_logits(
    probs: list[float], t_prob: float, energy_target: float, t_energy: float
) -> list[float]:
    """Build logits with a given softmax (at T_prob) and a given energy (at T_energy).

    ``softmax(z / T_prob) = p`` and ``E(z; T_energy) = energy_target``.
    Adding a constant to all logits does not change softmax; it shifts energy
    by the opposite amount.
    """
    p = np.asarray(probs, dtype=np.float64)
    if p.shape != (len(CLASS_IDS),):
        raise ValueError(f"probs must have length {len(CLASS_IDS)}")
    if not np.isclose(p.sum(), 1.0, atol=1e-15):
        raise ValueError("probs must sum to 1")
    if np.any(p <= 0.0):
        # Keep a tiny mass so log is defined; then renormalize.
        p = np.maximum(p, 1e-300)
        p = p / p.sum()
    z0 = t_prob * np.log(p)
    e0 = energy(z0, t_energy)
    shifted = z0 + (e0 - energy_target)
    return [float(x) for x in shifted]


def _manifest(
    *,
    t_cal: float | None,
    t_energy: float | None,
    energy_threshold: float = DEFAULT_ENERGY_THRESHOLD,
    min_top1: float = DEFAULT_MIN_TOP1,
    fusion: dict[str, Any] | None = DEFAULT_FUSION,
    classes: bool = True,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {}
    if t_cal is not None:
        manifest["calibration"] = {"temperature": t_cal}
    ood: dict[str, Any] = {
        "energyThreshold": energy_threshold,
        "minTop1Prob": min_top1,
    }
    if t_energy is not None:
        ood["energyTemperature"] = t_energy
    if fusion is not None:
        ood["fusion"] = fusion
    manifest["ood"] = ood
    if classes:
        manifest["classes"] = list(CLASS_IDS)
    return manifest


def _expected_single(logits: list[float], manifest: dict[str, Any]) -> dict[str, Any]:
    result = decide_single(logits, manifest)
    return {
        "status": result["status"],
        "energy": result["energy"],
        "p_top1": result["p_top1"],
        "candidates": result["candidates"],
    }


def _expected_fuse(
    photos: list[list[float]], manifest: dict[str, Any]
) -> dict[str, Any]:
    result = fuse(photos, manifest)
    return {
        "status": result["status"],
        "energy": result["energy"],
        "p_top1": result["p_top1"],
        "candidates": result["candidates"],
        "m": result["m"],
    }


def _case(
    id_: str,
    description: str,
    manifest: dict[str, Any],
    logits: Any,
    expected: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": id_,
        "description": description,
        "synthetic": True,
        "manifest": manifest,
        "logits": logits,
        "expected": expected,
    }


def build_decision_cases() -> dict[str, Any]:
    t_cal, t_en = 1.5, 1.2

    # Peaked in-distribution: high top-1, low (very negative) energy.
    peaked = [0.92, 0.03, 0.02, 0.01, 0.008, 0.006, 0.004, 0.002]
    ok_logits = _probs_to_logits(peaked, t_cal, energy_target=-8.0, t_energy=t_en)

    # Almost uniform, energy forced above the threshold (OOD) even if we
    # later also look at top-1 — gate order must report OOD first.
    flat = [0.20, 0.16, 0.14, 0.13, 0.12, 0.10, 0.08, 0.07]
    ood_logits = _probs_to_logits(flat, t_cal, energy_target=-2.0, t_energy=t_en)

    # Softmax is relatively flat (top-1 below 0.40) but energy is pushed
    # well below the threshold by a logit shift, so only the softmax gate fires.
    unclear_p = [0.28, 0.22, 0.14, 0.12, 0.09, 0.07, 0.05, 0.03]
    unclear_logits = _probs_to_logits(
        unclear_p, t_cal, energy_target=-8.0, t_energy=t_en
    )

    # Boundary: E == energyThreshold, p_top1 comfortable → ok (strict >).
    boundary_e_logits = _probs_to_logits(
        peaked, t_cal, energy_target=-6.0, t_energy=t_en
    )

    # Boundary: p_top1 == 0.40, energy comfortable → ok (strict <).
    boundary_p = [0.40, 0.25, 0.15, 0.08, 0.05, 0.04, 0.02, 0.01]
    boundary_p_logits = _probs_to_logits(
        boundary_p, t_cal, energy_target=-8.0, t_energy=t_en
    )

    # Large logits: a raw peaked vector, not constructed via log(p), to
    # exercise overflow-prone exp() without the shift in a naive implementation.
    large_logits = [1200.0, 1195.0, 800.0, 0.0, -50.0, -800.0, -1190.0, -1200.0]

    both = _manifest(t_cal=t_cal, t_energy=t_en)
    only_cal = _manifest(t_cal=1.7, t_energy=None)
    only_energy = _manifest(t_cal=None, t_energy=1.1)

    # Same "shape" for the T-source cases, so expected energy / p_top1 differ
    # when the resolved T changes.
    t_source_logits = _probs_to_logits(peaked, 1.7, energy_target=-8.0, t_energy=1.7)

    # Gate-order: energy fails AND top-1 is also below the floor. Status must
    # be out_of_distribution, not unclear.
    both_fail_logits = _probs_to_logits(
        unclear_p, t_cal, energy_target=-2.0, t_energy=t_en
    )

    cases = [
        _case(
            "clear_ok",
            "Synthetic peaked logits: energy well below threshold and top-1 well above minTop1Prob.",
            both,
            ok_logits,
            _expected_single(ok_logits, both),
        ),
        _case(
            "high_energy_ood",
            "Synthetic high energy (E much greater than the threshold) must yield out_of_distribution.",
            both,
            ood_logits,
            _expected_single(ood_logits, both),
        ),
        _case(
            "low_top1_unclear",
            "Energy passes but calibrated top-1 is below minTop1Prob → unclear.",
            both,
            unclear_logits,
            _expected_single(unclear_logits, both),
        ),
        _case(
            "gate_order_energy_first",
            "Both gates would fail; energy is evaluated first so the status is out_of_distribution.",
            both,
            both_fail_logits,
            _expected_single(both_fail_logits, both),
        ),
        _case(
            "temperature_only_calibration",
            "Only calibration.temperature is set; it is used for both softmax and energy.",
            only_cal,
            t_source_logits,
            _expected_single(t_source_logits, only_cal),
        ),
        _case(
            "temperature_only_energy",
            "Only ood.energyTemperature is set; it is used for both softmax and energy.",
            only_energy,
            t_source_logits,
            _expected_single(t_source_logits, only_energy),
        ),
        _case(
            "temperatures_both_differing",
            "calibration.temperature and ood.energyTemperature are both set and differ.",
            both,
            ok_logits,
            _expected_single(ok_logits, both),
        ),
        _case(
            "boundary_energy_equal",
            "E == energyThreshold. The energy comparison is strict (E > threshold), so this is not OOD.",
            both,
            boundary_e_logits,
            _expected_single(boundary_e_logits, both),
        ),
        _case(
            "boundary_top1_equal",
            "p_top1 == minTop1Prob. The softmax comparison is strict (p_top1 < threshold), so this is ok.",
            both,
            boundary_p_logits,
            _expected_single(boundary_p_logits, both),
        ),
        _case(
            "numerical_stability_large_logits",
            "Raw logits on the order of ±1e3; softmax and energy must stay finite and sum to 1.",
            both,
            large_logits,
            _expected_single(large_logits, both),
        ),
    ]
    return {
        "schemaVersion": FIXTURE_SCHEMA_VERSION,
        "generator": GENERATOR,
        "tolerance": TOLERANCE,
        "synthetic": True,
        "cases": cases,
    }


def _peaked_photo(
    t_cal: float,
    t_en: float,
    energy_target: float,
    peak_at: int = 0,
    peak: float = 0.90,
) -> list[float]:
    rest = (1.0 - peak) / (len(CLASS_IDS) - 1)
    probs = [rest] * len(CLASS_IDS)
    probs[peak_at] = peak
    return _probs_to_logits(probs, t_cal, energy_target, t_en)


def _ood_photo(t_cal: float, t_en: float) -> list[float]:
    flat = [0.20, 0.16, 0.14, 0.13, 0.12, 0.10, 0.08, 0.07]
    return _probs_to_logits(flat, t_cal, energy_target=-2.0, t_energy=t_en)


def build_fusion_cases() -> dict[str, Any]:
    t_cal, t_en = 1.5, 1.2
    manifest = _manifest(t_cal=t_cal, t_energy=t_en)

    # Energies sit well below fusion[3].energyThreshold (-6.5). Photo c
    # peaks on a different class than a/b, and disagreement raises fused
    # energy relative to the per-photo values (E of z-bar is not a mean of E_v).
    a = _peaked_photo(t_cal, t_en, -10.0, peak_at=0, peak=0.91)
    b = _peaked_photo(t_cal, t_en, -9.5, peak_at=0, peak=0.85)
    c = _peaked_photo(t_cal, t_en, -9.0, peak_at=1, peak=0.80)
    rejected = _ood_photo(t_cal, t_en)
    unclear_p = [0.28, 0.22, 0.14, 0.12, 0.09, 0.07, 0.05, 0.03]
    unclear = _probs_to_logits(unclear_p, t_cal, energy_target=-8.0, t_energy=t_en)

    large_a = [1200.0, 1190.0, 400.0, 0.0, -10.0, -400.0, -1000.0, -1200.0]
    large_b = [1180.0, 1195.0, 350.0, 1.0, -20.0, -300.0, -900.0, -1100.0]
    large_c = [1100.0, 1090.0, 1080.0, 0.0, -5.0, -200.0, -800.0, -1000.0]

    photos_m3 = [a, b, c]
    photos_m2 = [a, rejected, b]
    photos_m1 = [a, rejected, unclear]
    photos_m0 = [rejected, unclear, _ood_photo(t_cal, t_en)]
    # Shift the third accepted photo to make the fused energy fail fusion[3]
    # while each photo still passes the single-photo gate (E = -6.2 < -6.0
    # but fused threshold is -6.5).
    weak = _peaked_photo(t_cal, t_en, -6.2, peak_at=0, peak=0.70)
    photos_fused_ood = [weak, weak, weak]

    cases = [
        _case(
            "fusion_m3_all_accepted",
            "Three photos pass the per-photo gates; m=3 and ood.fusion['3'] is applied to z-bar.",
            manifest,
            photos_m3,
            _expected_fuse(photos_m3, manifest),
        ),
        _case(
            "fusion_m2_one_rejected_before",
            "The middle photo is rejected before fusion; m=2 and ood.fusion['2'] is applied.",
            manifest,
            photos_m2,
            _expected_fuse(photos_m2, manifest),
        ),
        _case(
            "fusion_m1_single_accepted",
            "Only one photo is accepted. There is no ood.fusion['1'] in the schema, so z-bar (the accepted logits) is scored with the single-photo thresholds.",
            manifest,
            photos_m1,
            _expected_fuse(photos_m1, manifest),
        ),
        _case(
            "fusion_m0_all_rejected",
            "Zero photos pass the per-photo gates; m=0 yields out_of_distribution and no z-bar (energy and p_top1 are null).",
            manifest,
            photos_m0,
            _expected_fuse(photos_m0, manifest),
        ),
        _case(
            "fusion_m3_fails_fused_energy",
            "All three photos pass the single-photo energy gate, but the fused energy is above ood.fusion['3'].energyThreshold.",
            manifest,
            photos_fused_ood,
            _expected_fuse(photos_fused_ood, manifest),
        ),
        _case(
            "fusion_numerical_stability_large_logits",
            "Two accepted photos with raw logits on the order of ±1e3; fused softmax and energy stay finite.",
            manifest,
            [large_a, large_b, large_c],
            _expected_fuse([large_a, large_b, large_c], manifest),
        ),
    ]

    # Sanity: constructed cases should really have the advertised m / status.
    _assert_m(cases[0], 3, STATUS_OK)
    _assert_m(cases[1], 2, STATUS_OK)
    _assert_m(cases[2], 1, STATUS_OK)
    _assert_m(cases[3], 0, STATUS_OOD)
    if cases[4]["expected"]["m"] != 3 or cases[4]["expected"]["status"] != STATUS_OOD:
        raise RuntimeError(
            f"fusion_m3_fails_fused_energy produced m={cases[4]['expected']['m']} "
            f"status={cases[4]['expected']['status']}; adjust energy targets"
        )

    return {
        "schemaVersion": FIXTURE_SCHEMA_VERSION,
        "generator": GENERATOR,
        "tolerance": TOLERANCE,
        "synthetic": True,
        "cases": cases,
    }


def _assert_m(case: dict[str, Any], m: int, status: str) -> None:
    got_m = case["expected"]["m"]
    got_s = case["expected"]["status"]
    if got_m != m or got_s != status:
        raise RuntimeError(
            f"{case['id']}: expected m={m} status={status}, got m={got_m} status={got_s}"
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_fixtures(output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    mapping = {
        "decision_cases.json": build_decision_cases(),
        "fusion_cases.json": build_fusion_cases(),
    }
    for name, payload in mapping.items():
        path = output_dir / name
        path.write_text(_dump(payload), encoding="utf-8")
        written[name] = _sha256(path)
    sums = output_dir / "SHA256SUMS"
    lines = [f"{digest}  {name}\n" for name, digest in written.items()]
    sums.write_text("".join(lines), encoding="utf-8")
    written["SHA256SUMS"] = _sha256(sums)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory for decision_cases.json, fusion_cases.json and SHA256SUMS",
    )
    args = parser.parse_args(argv)
    written = write_fixtures(args.output_dir.resolve())
    for name, digest in written.items():
        print(f"{digest}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
