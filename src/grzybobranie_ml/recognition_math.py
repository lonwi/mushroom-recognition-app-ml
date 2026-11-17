"""P2 v1 reference math for the energy OOD gate and entropy-weighted fusion.

Contract sources (read in this order of authority):

- ``docs/decisions/2026-11-16-schema-v1-clarifications.md``
- ``schemas/model-manifest.schema.json`` (v1.0.1)
- ``docs/prompty.md`` (P5, P7)
- ``docs/decisions/2026-11-16-p2-reference-math-assumptions.md`` (explicit
  assumptions where the contract is silent)

All intermediates are ``float64``. Public functions return plain Python
floats / lists so fixtures and the mobile-app tests can compare them
with tolerance ``1e-5``.

Gate order (P5 / schema):

1. If ``E(x) > energyThreshold`` → ``out_of_distribution``.
2. Else if ``p_top1 < minTop1Prob`` → ``unclear``.
3. Else → ``ok`` with candidates.

Both comparisons are **strict** (``>`` / ``<``): equality with the
threshold does not reject. See the assumptions note.

Temperatures (clarifications §b):

- Probabilities (softmax, fusion weights) use ``calibration.temperature``.
- Energy uses ``ood.energyTemperature``, falling back to
  ``calibration.temperature``.
- If only one of the two fields is present, that single T is used for both.

Fusion (clarifications §c, P5, schema ``fusionWeights.method``):

    p_v = softmax(z_v / T_prob)
    H_v = -Σ_k p_v[k] ln p_v[k]
    c_v = 1 - H_v / ln K
    w_v = c_v / Σ_u c_u    (equal weights 1/m if Σ c = 0)
    z̄   = Σ_v w_v z_v

Per-photo gates run first (``gateBeforeFusion = true``). ``m`` is the
number of photos with status ``ok``. Both gates are then re-evaluated on
``z̄`` with ``ood.fusion[m]`` when ``m >= 2``. ``m = 0`` is
``out_of_distribution``. ``m = 1`` is scored with the single-photo
thresholds (there is no ``ood.fusion["1"]`` in the schema).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

STATUS_OOD = "out_of_distribution"
STATUS_UNCLEAR = "unclear"
STATUS_OK = "ok"

TOP_K_CANDIDATES = 3
MAX_PHOTOS = 3
EPSILON = 0.0  # Σc <= 0 → equal weights (schema: "równe wagi gdy Σc = 0")


def _as_logits(z: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(z, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"logits must be a 1-D vector, got shape {arr.shape}")
    if arr.size < 2:
        raise ValueError("logits must contain at least 2 classes")
    if not np.all(np.isfinite(arr)):
        raise ValueError("logits must be finite")
    return arr


def _positive_t(T: float, name: str) -> float:
    value = float(np.float64(T))
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite number > 0, got {T!r}")
    return value


def softmax(z: Sequence[float] | np.ndarray, T: float) -> np.ndarray:
    """Return ``softmax(z / T)`` as a float64 vector that sums to 1.

    Numerically stable via subtraction of ``max(z / T)`` before ``exp``.
    """
    logits = _as_logits(z)
    temperature = _positive_t(T, "T")
    scaled = logits / temperature
    shifted = scaled - np.max(scaled)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def energy(z: Sequence[float] | np.ndarray, T: float) -> float:
    """Return the Liu et al. (2020) energy ``E(x) = -T · logsumexp(z / T)``.

    Computed in float64 with a stable log-sum-exp (subtract ``max(z / T)``).
    Lower (more negative) energy is more in-distribution.
    """
    logits = _as_logits(z)
    temperature = _positive_t(T, "T")
    scaled = logits / temperature
    peak = np.max(scaled)
    logsumexp = peak + np.log(np.sum(np.exp(scaled - peak)))
    return float(-temperature * logsumexp)


def resolve_temperatures(manifest: Mapping[str, Any]) -> tuple[float, float]:
    """Return ``(T_prob, T_energy)`` from a manifest (or a fixture subset).

    Rules from clarifications §b / schema descriptions of
    ``calibration.temperature`` and ``ood.energyTemperature``.
    """
    calibration = manifest.get("calibration") or {}
    ood = manifest.get("ood") or {}
    t_cal = calibration.get("temperature", None)
    t_energy = ood.get("energyTemperature", None)
    if t_cal is None and t_energy is None:
        raise ValueError(
            "manifest must provide calibration.temperature and/or ood.energyTemperature"
        )
    if t_cal is not None and t_energy is not None:
        return _positive_t(t_cal, "calibration.temperature"), _positive_t(
            t_energy, "ood.energyTemperature"
        )
    if t_cal is not None:
        t = _positive_t(t_cal, "calibration.temperature")
        return t, t
    t = _positive_t(t_energy, "ood.energyTemperature")
    return t, t


def _class_ids(
    k: int, manifest: Mapping[str, Any], class_ids: Sequence[str] | None
) -> list[str]:
    if class_ids is not None:
        ids = [str(item) for item in class_ids]
        if len(ids) != k:
            raise ValueError(f"class_ids length {len(ids)} != K={k}")
        return ids
    classes = manifest.get("classes")
    if isinstance(classes, list) and classes:
        ids: list[str] = []
        for i, item in enumerate(classes):
            if isinstance(item, Mapping):
                ids.append(str(item.get("classId") or item.get("id") or f"class_{i}"))
            else:
                ids.append(str(item))
        if len(ids) != k:
            raise ValueError(f"manifest classes length {len(ids)} != K={k}")
        return ids
    return [f"class_{i}" for i in range(k)]


def _candidates(
    probabilities: np.ndarray, class_ids: Sequence[str]
) -> list[dict[str, Any]]:
    """Top-3 classes by calibrated softmax, ties broken by lower index.

    Assumption (see the decision note): the schema describes a dangerous-genus
    warning on "pozycja 2–3", and the outdated prototype used a top-3 list.
    There is no documented cumulative-probability cutoff, so this reference
    returns at most ``TOP_K_CANDIDATES`` entries, or all K if K < 3.
    """
    # Higher probability first; ties broken by lower class index.
    indices = sorted(
        range(int(probabilities.size)),
        key=lambda i: (-float(probabilities[i]), i),
    )
    top = indices[: min(TOP_K_CANDIDATES, len(indices))]
    return [
        {
            "index": int(i),
            "classId": str(class_ids[i]),
            "probability": float(probabilities[i]),
            "rank": rank,
        }
        for rank, i in enumerate(top, start=1)
    ]


def _require_threshold(ood: Mapping[str, Any], key: str) -> float:
    if key not in ood or ood[key] is None:
        raise ValueError(f"manifest ood.{key} is required to run the gate")
    value = float(np.float64(ood[key]))
    if not np.isfinite(value):
        raise ValueError(f"manifest ood.{key} must be finite")
    return value


def _apply_gates(
    logits: np.ndarray,
    energy_threshold: float,
    min_top1_prob: float,
    t_prob: float,
    t_energy: float,
    class_ids: Sequence[str],
) -> dict[str, Any]:
    e = energy(logits, t_energy)
    probs = softmax(logits, t_prob)
    p_top1 = float(np.max(probs))
    result: dict[str, Any] = {
        "energy": e,
        "p_top1": p_top1,
        "candidates": [],
    }
    # Strict inequalities: E == threshold passes the energy gate;
    # p_top1 == minTop1Prob passes the softmax gate.
    if e > energy_threshold:
        result["status"] = STATUS_OOD
        return result
    if p_top1 < min_top1_prob:
        result["status"] = STATUS_UNCLEAR
        return result
    result["status"] = STATUS_OK
    result["candidates"] = _candidates(probs, class_ids)
    return result


def decide_single(
    logits: Sequence[float] | np.ndarray,
    manifest: Mapping[str, Any],
    class_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Apply the single-photo energy + minTop1Prob gates.

    Returns ``{status, energy, p_top1, candidates}``. Candidates are a
    top-3 list only when ``status == "ok"``; otherwise the list is empty
    (the scanner does not return a species on a rejected photo).
    """
    arr = _as_logits(logits)
    t_prob, t_energy = resolve_temperatures(manifest)
    ood = manifest.get("ood") or {}
    energy_threshold = _require_threshold(ood, "energyThreshold")
    min_top1_prob = _require_threshold(ood, "minTop1Prob")
    ids = _class_ids(int(arr.size), manifest, class_ids)
    return _apply_gates(arr, energy_threshold, min_top1_prob, t_prob, t_energy, ids)


def _entropy(probabilities: np.ndarray) -> float:
    # 0 * ln 0 is defined as 0 in information theory.
    positive = probabilities[probabilities > 0.0]
    return float(-np.sum(positive * np.log(positive)))


def _fusion_weights(accepted: Sequence[np.ndarray], t_prob: float) -> np.ndarray:
    """Entropy-weighted logits, P5 / schema ``entropy_weighted_logits``.

    ``c_v = 1 - H_v / ln K``. A peaked distribution has small entropy and
    therefore a larger weight. A uniform distribution has ``H = ln K`` and
    ``c = 0``. If every accepted photo is uniform, ``Σc = 0`` and the
    weights fall back to ``1/m`` (schema wording).
    """
    k = int(accepted[0].size)
    log_k = float(np.log(np.float64(k)))
    confidences = np.empty(len(accepted), dtype=np.float64)
    for i, logits in enumerate(accepted):
        if int(logits.size) != k:
            raise ValueError("all accepted photos must have the same K")
        probs = softmax(logits, t_prob)
        confidences[i] = 1.0 - (_entropy(probs) / log_k)
    total = float(np.sum(confidences))
    if total <= EPSILON:
        return np.full(len(accepted), 1.0 / len(accepted), dtype=np.float64)
    return confidences / total


def _fusion_thresholds(manifest: Mapping[str, Any], m: int) -> tuple[float, float]:
    ood = manifest.get("ood") or {}
    if m <= 1:
        override = (ood.get("fusion") or {}).get(str(m)) if m == 1 else None
        if override:
            return (
                _require_threshold(override, "energyThreshold"),
                _require_threshold(override, "minTop1Prob"),
            )
        return (
            _require_threshold(ood, "energyThreshold"),
            _require_threshold(ood, "minTop1Prob"),
        )
    fusion = ood.get("fusion") or {}
    # JSON object keys are strings "2" / "3"; also accept int keys.
    entry = fusion.get(str(m), fusion.get(m))
    if entry is None:
        raise ValueError(f"ood.fusion[{m}] is missing; fusion for m={m} is unavailable")
    return (
        _require_threshold(entry, "energyThreshold"),
        _require_threshold(entry, "minTop1Prob"),
    )


def fuse(
    list_of_logits: Sequence[Sequence[float] | np.ndarray],
    manifest: Mapping[str, Any],
    class_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Gate each photo, then entropy-weight the accepted logits and re-gate.

    Returns ``{status, energy, p_top1, candidates, m, weights, per_photo}``.
    ``energy`` / ``p_top1`` / ``candidates`` describe the fused decision
    (``z̄``). They are ``None`` / ``[]`` when ``m = 0`` (no ``z̄``).
    At most ``MAX_PHOTOS`` (3) photos are accepted as input; more is an error.
    """
    if len(list_of_logits) > MAX_PHOTOS:
        raise ValueError(
            f"at most {MAX_PHOTOS} photos are allowed (maxPhotos), got {len(list_of_logits)}"
        )
    photos = [_as_logits(item) for item in list_of_logits]
    k = int(photos[0].size) if photos else 0
    for arr in photos:
        if int(arr.size) != k:
            raise ValueError("all photos must have the same K")
    ids = _class_ids(k, manifest, class_ids) if photos else []

    per_photo = [decide_single(arr, manifest, class_ids=ids) for arr in photos]
    accepted = [
        arr
        for arr, decision in zip(photos, per_photo)
        if decision["status"] == STATUS_OK
    ]
    m = len(accepted)

    if m == 0:
        return {
            "status": STATUS_OOD,
            "energy": None,
            "p_top1": None,
            "candidates": [],
            "m": 0,
            "weights": [],
            "per_photo": per_photo,
        }

    t_prob, t_energy = resolve_temperatures(manifest)
    weights = _fusion_weights(accepted, t_prob)
    stacked = np.stack(accepted, axis=0)
    z_bar = np.sum(weights[:, None] * stacked, axis=0)
    energy_threshold, min_top1_prob = _fusion_thresholds(manifest, m)
    fused = _apply_gates(z_bar, energy_threshold, min_top1_prob, t_prob, t_energy, ids)
    fused["m"] = m
    fused["weights"] = [float(w) for w in weights]
    fused["per_photo"] = per_photo
    return fused


__all__ = [
    "MAX_PHOTOS",
    "STATUS_OK",
    "STATUS_OOD",
    "STATUS_UNCLEAR",
    "TOP_K_CANDIDATES",
    "decide_single",
    "energy",
    "fuse",
    "resolve_temperatures",
    "softmax",
]
