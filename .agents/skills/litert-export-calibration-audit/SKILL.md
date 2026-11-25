---
name: litert-export-calibration-audit
description: >-
  Runbook for Stages P4–P6 in grzybobranie-ml: temperature scaling, energy OOD sweeps,
  4-step multi-photo logit fusion, LiteRT FP32/FP16/INT8 export, quantization drift
  verification, and frozen Ship Gates evaluation. Use when calibrating models, checking
  FP16 vs INT8 parity, or running final test audits.
---

# LiteRT Export, OOD Calibration & Ship Gates Runbook (P4–P6)

## Critical Mathematical & Safety Guardrails

1. **Temperature Scaling (`calibrate_ood.py`):**
   - Always pass `line_search_fn="strong_wolfe"` to `torch.optim.LBFGS` (without line search, scalar optimization stalls early).
   - When finding energy thresholds from a target ID TPR, always pass `method="higher"` to `np.percentile` so validation keep rate is strictly `>= target_id_tpr`.
   - Store and validate `checkpoint_path` + `checkpoint_sha256` inside cached logit files (`extracted_logits.pt`), or pass `--force-extract` when changing checkpoints.
2. **Multi-Photo Logit Fusion (`multi_photo.py`):**
   - Logit averaging alters softmax margin separation; always verify post-fusion ECE on `val.csv` and fit a dedicated `T_multi` (`0.5146` vs single-frame `0.7848`) exclusively on validation groups.
   - Run Step 4 (`HIGH_STAKES_VETO`, `p_toxic >= 0.20`) across **all** input frames (`veto_check_all_frames=True`), including frames rejected by per-frame OOD energy, whenever fused Top-1 is edible.
3. **LiteRT Export & Quantization Parity (`export_tflite.py`, `quantization_drift.py`):**
   - Resolve flatbuffer opcodes with `ai_edge_litert.schema_py_generated.BuiltinOperator` (not `ai_edge_quantizer`'s `TFL_OP_CODE_TO_NAME`, which omits runtime ops like opcode `6` = `DEQUANTIZE`).
   - Run `quantization_drift` on `val.csv` before touching `test.csv`: verify `tfliteAgreement >= 0.99` and logit std scale ratio (`~1.0`). Do not ship INT8 if logit scale compression (`< 0.95`) destabilizes the energy threshold `E(x) > theta`; prefer FP16.

## Standard Execution Sequence

```bash
# 1. Stage P4 calibration & OOD evaluation (val.csv + ood_calib/ood_test reporting)
.venv/bin/python -m grzybobranie_ml.calibrate_ood \
  --checkpoint artifacts/train_runs/<run_id>/best_val_checkpoint.pt

# 2. OOD threshold sweep (val.csv + ood_calib ONLY) + P5 multi-photo validation
.venv/bin/python -m grzybobranie_ml.multi_photo

# 3. Export FP32, FP16, and INT8 (representative sample strictly from train/val) + drift check
.venv/bin/python -m grzybobranie_ml.export_tflite \
  --checkpoint artifacts/train_runs/<run_id>/best_val_checkpoint.pt \
  --verify-drift

# 4. Single frozen Ship Gates audit on test.csv & ood_test (run ONLY after freezing thresholds)
.venv/bin/python -m grzybobranie_ml.ship_gates
```

## Verification Checklist

- Confirm `git diff data/splits/test.csv` is empty prior to Step 4.
- Run `.venv/bin/pytest tests/` and `.venv/bin/ruff check .` after any module edits.
- Confirm `artifacts/quantization/export_manifest.json` contains canonical operator names (no numeric opcode strings like `"6"`).
