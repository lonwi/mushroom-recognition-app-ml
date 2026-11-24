"""Verify logit drift, top-1 accuracy, ECE, and energy gate behavior after quantization on val.csv.

Thin CLI wrapper around grzybobranie_ml.quantization_drift.
"""

from grzybobranie_ml.quantization_drift import (
    compute_metrics,
    evaluate_runner,
    main,
    verify_quantization_drift,
)

__all__ = ["compute_metrics", "evaluate_runner", "main", "verify_quantization_drift"]

if __name__ == "__main__":
    main()
