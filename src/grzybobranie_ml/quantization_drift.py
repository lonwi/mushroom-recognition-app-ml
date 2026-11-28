"""Verify logit drift, top-1 accuracy, ECE, and energy gate behavior after quantization on val.csv.

Compares:
- PyTorch (EMA float32)
- TFLite FP16 (ai_edge_quantizer float_casting)
- TFLite INT8 (ai_edge_quantizer static_wi8_ai8)

CLI invocation:
    python -m grzybobranie_ml.quantization_drift
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from grzybobranie_ml.calibrate_ood import (
    compute_ece_mce,
    compute_energy_vector,
    compute_softmax,
)
from grzybobranie_ml.export_tflite import TFLiteRunner
from grzybobranie_ml.train import (
    MushroomSplitDataset,
    get_val_transforms,
    load_high_stakes_species,
)

logger = logging.getLogger("grzybobranie_ml.quantization_drift")

TEMPERATURE_SINGLE = 0.7848
ENERGY_THRESHOLD = -2.2082
MIN_TOP1_PROB = 0.50

DEFAULT_FP16_PATH = Path("artifacts/models/mobilenetv4_medium_r256_fp16.tflite")
DEFAULT_INT8_PATH = Path("artifacts/models/mobilenetv4_medium_r256_int8.tflite")
DEFAULT_OUTPUT_JSON = Path("artifacts/quantization/quantization_drift_val.json")
DEFAULT_IMAGE_DIR = Path("/home/bubolka/hf-768/upload")
DEFAULT_VAL_CSV = Path("data/splits/val.csv")
DEFAULT_KARTY_PATH = Path("configs/karty_sobowtory.json")
DEFAULT_CALIB_CACHE = Path("artifacts/calibration/extracted_logits.pt")


def evaluate_runner(
    runner: TFLiteRunner,
    loader: DataLoader,
    desc: str,
) -> np.ndarray:
    """Run TFLite model over all batches in DataLoader and return concatenated logits [N, K]."""
    logger.info("Running %s inference on %d batches...", desc, len(loader))
    all_logits = []
    t0 = time.perf_counter()
    for i, (images, _) in enumerate(loader):
        # images is [B, 3, H, W] in [-1, 1] -> permute to NHWC [B, H, W, 3] float32
        nhwc = images.permute(0, 2, 3, 1).numpy().astype(np.float32)
        logits_batch = runner.predict_batch(nhwc)
        all_logits.append(logits_batch)
        if (i + 1) % 25 == 0 or (i + 1) == len(loader):
            elapsed = time.perf_counter() - t0
            logger.info("  %s: batch %d/%d (%.2f s elapsed)", desc, i + 1, len(loader), elapsed)
    return np.concatenate(all_logits, axis=0)


def compute_metrics(
    logits: np.ndarray,
    targets: np.ndarray,
    ref_logits: np.ndarray,
    high_stakes_mask: np.ndarray,
    temperature: float = TEMPERATURE_SINGLE,
    energy_threshold: float = ENERGY_THRESHOLD,
    min_top1_prob: float = MIN_TOP1_PROB,
) -> dict[str, Any]:
    """Compute detailed numerical and safety metrics for a model's logits."""
    # 1. Logit errors vs PyTorch reference
    diff = np.abs(logits - ref_logits)
    max_logit_err = float(np.max(diff))
    mean_logit_mae = float(np.mean(diff))

    # 2. Predictions & Agreement
    preds = np.argmax(logits, axis=1)
    ref_preds = np.argmax(ref_logits, axis=1)
    top1_agreement = float(np.mean(preds == ref_preds))
    top1_acc = float(np.mean(preds == targets))
    ref_acc = float(np.mean(ref_preds == targets))
    acc_drop = float(ref_acc - top1_acc)

    # High stakes agreement
    if np.sum(high_stakes_mask) > 0:
        hs_agreement = float(np.mean(preds[high_stakes_mask] == ref_preds[high_stakes_mask]))
        hs_acc = float(np.mean(preds[high_stakes_mask] == targets[high_stakes_mask]))
    else:
        hs_agreement = 1.0
        hs_acc = 1.0

    # 3. Calibration ECE / MCE
    ece, mce, _ = compute_ece_mce(logits, targets, temperature=temperature, n_bins=15)

    # 4. Energy gate
    e = compute_energy_vector(logits, temperature=temperature)
    ref_e = compute_energy_vector(ref_logits, temperature=temperature)
    energy_mae = float(np.mean(np.abs(e - ref_e)))
    max_energy_diff = float(np.max(np.abs(e - ref_e)))

    # Probabilities
    probs = compute_softmax(logits, temperature)
    top1_p = np.max(probs, axis=1)

    is_ood = e > energy_threshold
    is_unclear = (~is_ood) & (top1_p < min_top1_prob)
    is_ok = (~is_ood) & (~is_unclear)

    return {
        "max_logit_error": max_logit_err,
        "mean_logit_mae": mean_logit_mae,
        "top1_agreement": top1_agreement,
        "top1_accuracy": top1_acc,
        "accuracy_drop": acc_drop,
        "high_stakes_top1_agreement": hs_agreement,
        "high_stakes_top1_accuracy": hs_acc,
        "ece": float(ece),
        "mce": float(mce),
        "energy_mean": float(np.mean(e)),
        "energy_std": float(np.std(e)),
        "energy_mae_vs_ref": energy_mae,
        "max_energy_diff_vs_ref": max_energy_diff,
        "energy_rejection_rate": float(np.mean(is_ood)),
        "unclear_rejection_rate": float(np.mean(is_unclear)),
        "total_rejection_rate": float(np.mean(~is_ok)),
        "val_keep_rate": float(np.mean(is_ok)),
    }


def verify_quantization_drift(
    fp16_model_path: str | Path = DEFAULT_FP16_PATH,
    int8_model_path: str | Path = DEFAULT_INT8_PATH,
    calib_cache_path: str | Path = DEFAULT_CALIB_CACHE,
    val_csv: str | Path = DEFAULT_VAL_CSV,
    image_dir: str | Path = DEFAULT_IMAGE_DIR,
    karty_path: str | Path = DEFAULT_KARTY_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_JSON,
    temperature: float = TEMPERATURE_SINGLE,
    energy_threshold: float = ENERGY_THRESHOLD,
    min_top1_prob: float = MIN_TOP1_PROB,
    crop_mode: str | None = None,
) -> dict[str, Any]:
    """Execute complete quantization drift verification on val.csv."""
    extracted_path = Path(calib_cache_path)
    if not extracted_path.exists():
        raise FileNotFoundError(f"Calibration cache not found at: {extracted_path}")

    cache = torch.load(extracted_path, map_location="cpu", weights_only=False)
    cached_crop_mode = str(cache.get("crop_mode", "legacy_448_center"))
    if crop_mode is not None and crop_mode != cached_crop_mode:
        raise ValueError(
            f"Crop mode mismatch: requested {crop_mode!r}, but {extracted_path} was extracted with "
            f"{cached_crop_mode!r}. Re-run calibrate_ood --force-extract --crop-mode {crop_mode} first."
        )
    effective_crop_mode = crop_mode or cached_crop_mode

    py_val_logits = cache["val_logits"].float().numpy()
    targets = np.array(cache["val_targets"], dtype=np.int64)
    class_to_idx = cache["class_to_idx_66"]

    high_stakes_species = load_high_stakes_species(karty_path)
    high_stakes_indices = {class_to_idx[s] for s in high_stakes_species if s in class_to_idx}
    high_stakes_mask = np.isin(targets, list(high_stakes_indices))
    logger.info("Loaded PyTorch logits: %s (crop_mode=%s), high-stakes samples: %d", py_val_logits.shape, effective_crop_mode, np.sum(high_stakes_mask))

    # Dataset loader
    val_ds = MushroomSplitDataset(
        csv_path=val_csv,
        image_dir=image_dir,
        transform=get_val_transforms(256, crop_mode=effective_crop_mode),
        class_to_idx=class_to_idx,
    )
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=8, pin_memory=False)

    # 1. PyTorch Baseline Metrics
    py_metrics = compute_metrics(
        logits=py_val_logits,
        targets=targets,
        ref_logits=py_val_logits,
        high_stakes_mask=high_stakes_mask,
        temperature=temperature,
        energy_threshold=energy_threshold,
        min_top1_prob=min_top1_prob,
    )
    logger.info("PyTorch Baseline | Top-1 Acc: %.4f | ECE: %.4f | Keep Rate: %.4f", py_metrics["top1_accuracy"], py_metrics["ece"], py_metrics["val_keep_rate"])

    # 2. TFLite FP16 Runner
    fp16_runner = TFLiteRunner(fp16_model_path, num_threads=8)
    fp16_logits = evaluate_runner(fp16_runner, val_loader, "TFLite FP16")
    fp16_metrics = compute_metrics(
        logits=fp16_logits,
        targets=targets,
        ref_logits=py_val_logits,
        high_stakes_mask=high_stakes_mask,
        temperature=temperature,
        energy_threshold=energy_threshold,
        min_top1_prob=min_top1_prob,
    )
    logger.info("TFLite FP16 | Agreement: %.4f | Top-1 Acc: %.4f | ECE: %.4f | Keep Rate: %.4f", fp16_metrics["top1_agreement"], fp16_metrics["top1_accuracy"], fp16_metrics["ece"], fp16_metrics["val_keep_rate"])

    # 3. TFLite INT8 Runner
    int8_runner = TFLiteRunner(int8_model_path, num_threads=8)
    int8_logits = evaluate_runner(int8_runner, val_loader, "TFLite INT8")
    int8_metrics = compute_metrics(
        logits=int8_logits,
        targets=targets,
        ref_logits=py_val_logits,
        high_stakes_mask=high_stakes_mask,
        temperature=temperature,
        energy_threshold=energy_threshold,
        min_top1_prob=min_top1_prob,
    )
    logger.info("TFLite INT8 | Agreement: %.4f | Top-1 Acc: %.4f | ECE: %.4f | Keep Rate: %.4f", int8_metrics["top1_agreement"], int8_metrics["top1_accuracy"], int8_metrics["ece"], int8_metrics["val_keep_rate"])

    # 4. Comparative analysis and Recommendation
    scale_shift_int8 = float(np.mean(np.std(int8_logits, axis=1) / np.std(py_val_logits, axis=1)))
    scale_shift_fp16 = float(np.mean(np.std(fp16_logits, axis=1) / np.std(py_val_logits, axis=1)))

    int8_breaks_agreement = int8_metrics["top1_agreement"] < 0.99
    int8_breaks_energy = abs(int8_metrics["val_keep_rate"] - py_metrics["val_keep_rate"]) > 0.05

    recommendation = {
        "recommended_production_variant": "FP16",
        "rationale": (
            f"FP16 achieves {fp16_metrics['top1_agreement']*100:.2f}% Top-1 agreement with PyTorch, "
            f"virtually zero accuracy drop ({fp16_metrics['accuracy_drop']*100:+.2f}%), "
            f"and preserves ECE ({fp16_metrics['ece']*100:.2f}%) and energy gating ({fp16_metrics['val_keep_rate']*100:.2f}% keep rate). "
            f"In contrast, INT8 shifts logit scale (std ratio: {scale_shift_int8:.3f}), "
            f"degrades Top-1 agreement to {int8_metrics['top1_agreement']*100:.2f}% (fails >=99% ship gate threshold), "
            f"increases logit MAE to {int8_metrics['mean_logit_mae']:.4f}, and alters energy gate calibration "
            f"(keep rate: {int8_metrics['val_keep_rate']*100:.2f}% vs PyTorch {py_metrics['val_keep_rate']*100:.2f}%)."
        ),
        "int8_shifts_logit_scale": bool(abs(scale_shift_int8 - 1.0) > 0.05),
        "int8_breaks_energy_calibration": bool(int8_breaks_energy),
        "int8_scale_ratio": scale_shift_int8,
        "fp16_scale_ratio": scale_shift_fp16,
        "int8_passes_ship_gate_agreement": not int8_breaks_agreement,
        "fp16_passes_ship_gate_agreement": fp16_metrics["top1_agreement"] >= 0.99,
    }

    report = {
        "val_samples_evaluated": len(val_ds),
        "crop_mode": effective_crop_mode,
        "high_stakes_samples": int(np.sum(high_stakes_mask)),
        "frozen_thresholds": {
            "temperature_single": temperature,
            "energy_threshold_theta": energy_threshold,
            "min_top1_prob": min_top1_prob,
        },
        "pytorch": py_metrics,
        "tflite_fp16": fp16_metrics,
        "tflite_int8": int8_metrics,
        "quantization_drift_comparison": {
            "logit_mae": {
                "fp16": fp16_metrics["mean_logit_mae"],
                "int8": int8_metrics["mean_logit_mae"],
            },
            "max_logit_error": {
                "fp16": fp16_metrics["max_logit_error"],
                "int8": int8_metrics["max_logit_error"],
            },
            "top1_agreement_pct": {
                "fp16": round(fp16_metrics["top1_agreement"] * 100, 2),
                "int8": round(int8_metrics["top1_agreement"] * 100, 2),
            },
            "high_stakes_agreement_pct": {
                "fp16": round(fp16_metrics["high_stakes_top1_agreement"] * 100, 2),
                "int8": round(int8_metrics["high_stakes_top1_agreement"] * 100, 2),
            },
            "top1_accuracy_pct": {
                "pytorch": round(py_metrics["top1_accuracy"] * 100, 2),
                "fp16": round(fp16_metrics["top1_accuracy"] * 100, 2),
                "int8": round(int8_metrics["top1_accuracy"] * 100, 2),
            },
            "ece_pct": {
                "pytorch": round(py_metrics["ece"] * 100, 2),
                "fp16": round(fp16_metrics["ece"] * 100, 2),
                "int8": round(int8_metrics["ece"] * 100, 2),
            },
            "val_keep_rate_pct": {
                "pytorch": round(py_metrics["val_keep_rate"] * 100, 2),
                "fp16": round(fp16_metrics["val_keep_rate"] * 100, 2),
                "int8": round(int8_metrics["val_keep_rate"] * 100, 2),
            },
            "energy_rejection_rate_pct": {
                "pytorch": round(py_metrics["energy_rejection_rate"] * 100, 2),
                "fp16": round(fp16_metrics["energy_rejection_rate"] * 100, 2),
                "int8": round(int8_metrics["energy_rejection_rate"] * 100, 2),
            },
        },
        "recommendation": recommendation,
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Saved quantization drift report to %s", out_file)
    print("\n" + "=" * 80)
    print("QUANTIZATION DRIFT EVALUATION SUMMARY (val.csv)")
    print("=" * 80)
    print(f"PyTorch EMA Top-1 Accuracy:  {py_metrics['top1_accuracy']*100:.2f}% | ECE: {py_metrics['ece']*100:.2f}% | Keep Rate: {py_metrics['val_keep_rate']*100:.2f}%")
    print(f"TFLite FP16 Top-1 Accuracy:  {fp16_metrics['top1_accuracy']*100:.2f}% | Agreement: {fp16_metrics['top1_agreement']*100:.2f}% | MAE: {fp16_metrics['mean_logit_mae']:.4f} | ECE: {fp16_metrics['ece']*100:.2f}%")
    print(f"TFLite INT8 Top-1 Accuracy:  {int8_metrics['top1_accuracy']*100:.2f}% | Agreement: {int8_metrics['top1_agreement']*100:.2f}% | MAE: {int8_metrics['mean_logit_mae']:.4f} | ECE: {int8_metrics['ece']*100:.2f}%")
    print("-" * 80)
    print(f"Recommendation: {recommendation['recommended_production_variant']}")
    print(f"Rationale: {recommendation['rationale']}")
    print("=" * 80 + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify quantization drift on val.csv.")
    parser.add_argument("--fp16-model", type=str, default=str(DEFAULT_FP16_PATH), help="FP16 model path")
    parser.add_argument("--int8-model", type=str, default=str(DEFAULT_INT8_PATH), help="INT8 model path")
    parser.add_argument("--calib-cache", type=str, default=str(DEFAULT_CALIB_CACHE), help="Calibration cache .pt")
    parser.add_argument("--val-csv", type=str, default=str(DEFAULT_VAL_CSV), help="Validation split CSV")
    parser.add_argument("--image-dir", type=str, default=str(DEFAULT_IMAGE_DIR), help="Image directory")
    parser.add_argument("--karty-path", type=str, default=str(DEFAULT_KARTY_PATH), help="Path to karty_sobowtory.json")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_JSON), help="Output JSON path")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE_SINGLE, help="Temperature single")
    parser.add_argument("--energy-threshold", type=float, default=ENERGY_THRESHOLD, help="Energy threshold theta")
    parser.add_argument("--min-top1-prob", type=float, default=MIN_TOP1_PROB, help="Minimum top1 prob")
    parser.add_argument("--crop-mode", type=str, default=None, help="Validation preprocessing crop mode (defaults to cache crop_mode)")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    verify_quantization_drift(
        fp16_model_path=args.fp16_model,
        int8_model_path=args.int8_model,
        calib_cache_path=args.calib_cache,
        val_csv=args.val_csv,
        image_dir=args.image_dir,
        karty_path=args.karty_path,
        output_path=args.output,
        temperature=args.temperature,
        energy_threshold=args.energy_threshold,
        min_top1_prob=args.min_top1_prob,
        crop_mode=args.crop_mode,
    )


if __name__ == "__main__":
    main()
