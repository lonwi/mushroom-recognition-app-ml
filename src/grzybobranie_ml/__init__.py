"""Standalone ML pipeline package for Grzybobranie AI."""

from __future__ import annotations

__version__ = "0.1.0"

from grzybobranie_ml.license_policy import (
    is_allowed_license,
    is_ui_safe_license,
    normalize_license,
)
from grzybobranie_ml.recognition_math import (
    decide_single,
    energy,
    fuse,
    resolve_temperatures,
    softmax,
)
from grzybobranie_ml.split import (
    compute_connected_components,
    compute_fusion_observation_stats,
    resolve_split_salt,
    stratified_hmac_split,
)

__all__ = [
    "__version__",
    "compute_connected_components",
    "compute_fusion_observation_stats",
    "decide_single",
    "energy",
    "fuse",
    "is_allowed_license",
    "is_ui_safe_license",
    "normalize_license",
    "resolve_split_salt",
    "resolve_temperatures",
    "softmax",
    "stratified_hmac_split",
]
