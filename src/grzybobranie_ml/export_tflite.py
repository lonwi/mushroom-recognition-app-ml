"""Export trained PyTorch checkpoints to LiteRT / TFLite (FP32, FP16, INT8).

Implements Stage P6 model conversion pipeline:
- Slices out excluded species (Amanita verna at index 12) to produce K=66 active classes.
- Wraps model in NHWC layout [1, N, N, 3] -> raw logits [1, K].
- Exports FP32 baseline via litert_torch.
- Exports FP16 mobile candidate via ai_edge_quantizer float_casting.
- Exports INT8 candidate via static calibration strictly on train split samples.
- Provides SHA-256 digests, model file sizes, and operator inventory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import ai_edge_litert
import ai_edge_litert.interpreter as tflite
import ai_edge_quantizer as aeq
import litert_torch
import numpy as np
import torch
from ai_edge_quantizer import qtyping
from ai_edge_quantizer.utils import tfl_flatbuffer_utils
from torch import nn

from grzybobranie_ml.train import (
    MushroomSplitDataset,
    create_classifier_model,
    get_val_transforms,
)

logger = logging.getLogger("grzybobranie_ml.export_tflite")

DEFAULT_CHECKPOINT = Path(
    "artifacts/train_runs/mobilenetv4_conv_medium.e500_r256_in1k_r256_1795514748/best_val_checkpoint.pt"
)
DEFAULT_IMAGE_DIR = Path("/home/bubolka/hf-768/upload")
DEFAULT_TRAIN_CSV = Path("data/splits/train.csv")
DEFAULT_OUTPUT_DIR = Path("artifacts/models")
DEFAULT_QUANT_DIR = Path("artifacts/quantization")


class NHWCWrapper(nn.Module):
    """Wrapper module that takes NHWC [B, H, W, 3] input and passes NCHW to internal model."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [B, H, W, C] -> permute to [B, C, H, W]
        x_nchw = x.permute(0, 3, 1, 2)
        return self.model(x_nchw)


def compute_file_sha256(path: str | Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def inspect_tflite_operators(model_path: str | Path) -> list[str]:
    """Inspect and return sorted unique operator names from a TFLite file."""
    import ai_edge_litert.schema_py_generated as schema

    op_code_map = {
        v: k for k, v in schema.BuiltinOperator.__dict__.items() if isinstance(v, int)
    }
    model = tfl_flatbuffer_utils.read_model(str(model_path))
    op_names: set[str] = set()
    for op_code in model.operatorCodes:
        builtin = op_code.builtinCode
        name = op_code_map.get(builtin)
        if name is None:
            name_obj = tfl_flatbuffer_utils.TFL_OP_CODE_TO_NAME.get(builtin, str(builtin))
            name = getattr(name_obj, "value", str(name_obj))
        op_names.add(str(name))
    return sorted(op_names)


def _disable_fused_attention(model: nn.Module) -> int:
    """Disable PyTorch SDPA fast-path (`fused_attn`) for accurate LiteRT lowering."""
    disabled = 0
    for mod in model.modules():
        if getattr(mod, "fused_attn", False):
            mod.fused_attn = False
            disabled += 1
    if disabled > 0:
        logger.info(
            "Disabled fused_attn on %d attention modules to preserve LiteRT parity.",
            disabled,
        )
    return disabled


def load_and_prepare_model(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    exclude_verna: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load checkpoint with EMA weights, slice out Amanita verna (idx 12) if requested, and wrap in NHWC."""
    ckpt_p = Path(checkpoint_path)
    if not ckpt_p.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_p}")

    ckpt = torch.load(ckpt_p, map_location="cpu", weights_only=False)
    model_name = ckpt.get("model_name", "mobilenetv4_conv_medium.e500_r256_in1k")
    input_size = int(ckpt.get("input_size", 256))
    class_to_idx = dict(ckpt.get("class_to_idx", {}))
    num_classes = len(class_to_idx)

    logger.info("Loading architecture %s with %d classes from %s", model_name, num_classes, ckpt_p)
    model = create_classifier_model(model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt.get("ema_state_dict", ckpt.get("model_state_dict")))
    model.eval()

    if exclude_verna and "amanita_verna" in class_to_idx:
        verna_idx = class_to_idx["amanita_verna"]
        orig_w = model.classifier.weight.data
        orig_b = model.classifier.bias.data

        sliced_w = torch.cat([orig_w[:verna_idx], orig_w[verna_idx + 1:]], dim=0)
        sliced_b = torch.cat([orig_b[:verna_idx], orig_b[verna_idx + 1:]], dim=0)

        model.reset_classifier(num_classes - 1)
        model.classifier.weight.data.copy_(sliced_w)
        model.classifier.bias.data.copy_(sliced_b)
        model.eval()

        active_class_to_idx = {
            c: (idx if idx < verna_idx else idx - 1)
            for c, idx in class_to_idx.items()
            if c != "amanita_verna"
        }
        active_classes = sorted(active_class_to_idx.keys(), key=lambda c: active_class_to_idx[c])
        active_idx_to_class = {i: c for c, i in active_class_to_idx.items()}
        logger.info(
            "Sliced out amanita_verna at idx %d. Model classifier now has %d output classes.",
            verna_idx,
            len(active_classes),
        )
    else:
        active_class_to_idx = class_to_idx
        active_classes = sorted(active_class_to_idx.keys(), key=lambda c: active_class_to_idx[c])
        active_idx_to_class = {v: k for k, v in class_to_idx.items()}

    wrapped_model = NHWCWrapper(model).eval()
    _disable_fused_attention(wrapped_model)

    meta = {
        "model_name": model_name,
        "input_size": input_size,
        "active_classes": active_classes,
        "class_to_idx": active_class_to_idx,
        "idx_to_class": active_idx_to_class,
        "checkpoint_sha256": compute_file_sha256(ckpt_p),
    }
    return wrapped_model, meta


def export_fp32(
    wrapped_model: nn.Module,
    input_size: int = 256,
    output_path: str | Path = DEFAULT_OUTPUT_DIR / "mobilenetv4_medium_r256_fp32.tflite",
) -> Path:
    """Export wrapped PyTorch model to standard FP32 TFLite via litert_torch."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    _disable_fused_attention(wrapped_model)

    dummy_nhwc = torch.randn(1, input_size, input_size, 3, dtype=torch.float32)
    logger.info("Converting PyTorch model to FP32 TFLite at %s...", out_p)
    edge_model = litert_torch.convert(wrapped_model, (dummy_nhwc,))
    edge_model.export(str(out_p))

    logger.info("Exported FP32 TFLite: %s (%.2f MB)", out_p, out_p.stat().st_size / (1024 * 1024))
    return out_p


def quantize_fp16(
    fp32_tflite_path: str | Path,
    output_path: str | Path = DEFAULT_OUTPUT_DIR / "mobilenetv4_medium_r256_fp16.tflite",
) -> Path:
    """Quantize FP32 TFLite model to FP16 weights via ai_edge_quantizer float_casting."""
    in_p = Path(fp32_tflite_path)
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Quantizing FP32 model %s to FP16 -> %s...", in_p, out_p)
    qt = aeq.Quantizer(str(in_p))
    cfg = qtyping.OpQuantizationConfig(
        weight_tensor_config=qtyping.TensorQuantizationConfig(
            num_bits=16, dtype=qtyping.TensorDataType.FLOAT
        ),
        compute_precision=qtyping.ComputePrecision.FLOAT,
        explicit_dequantize=True,
    )
    for op_name in [
        qtyping.TFLOperationName.CONV_2D,
        qtyping.TFLOperationName.DEPTHWISE_CONV_2D,
        qtyping.TFLOperationName.FULLY_CONNECTED,
    ]:
        qt.update_quantization_recipe(
            regex=".*",
            operation_name=op_name,
            op_config=cfg,
            algorithm_key="float_casting",
        )

    qt.quantize(serialize_to_path=str(out_p))
    logger.info("Exported FP16 TFLite: %s (%.2f MB)", out_p, out_p.stat().st_size / (1024 * 1024))
    return out_p


def quantize_int8(
    fp32_tflite_path: str | Path,
    train_csv: str | Path = DEFAULT_TRAIN_CSV,
    image_dir: str | Path = DEFAULT_IMAGE_DIR,
    output_path: str | Path = DEFAULT_OUTPUT_DIR / "mobilenetv4_medium_r256_int8.tflite",
    num_calib_samples: int = 200,
    input_size: int = 256,
    crop_mode: str = "none_stretch",
) -> Path:
    """Quantize FP32 TFLite model to full INT8 static quantization using training samples strictly."""
    in_p = Path(fp32_tflite_path)
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Preparing %d calibration samples from %s (strictly training split)...",
        num_calib_samples,
        train_csv,
    )
    val_transform = get_val_transforms(input_size, crop_mode=crop_mode)
    train_ds = MushroomSplitDataset(
        csv_path=train_csv,
        image_dir=image_dir,
        transform=val_transform,
        mock_if_missing=False,
    )

    n_available = len(train_ds)
    n_samples = min(num_calib_samples, n_available)

    # Deterministic step sampling across the train dataset
    indices = np.linspace(0, n_available - 1, n_samples, dtype=int)
    calib_data = []
    for idx in indices:
        tensor, _ = train_ds[int(idx)]
        # tensor is [3, H, W] in [-1, 1] -> permute to [1, H, W, 3] float32
        nhwc = tensor.permute(1, 2, 0).unsqueeze(0).numpy().astype(np.float32)
        calib_data.append({"args_0": nhwc})

    logger.info("Running INT8 static calibration on %d samples with recipe static_wi8_ai8...", len(calib_data))
    qt = aeq.Quantizer(str(in_p), aeq.recipe.static_wi8_ai8())
    calib_results = qt.calibrate({"serving_default": calib_data})
    qt.quantize(calib_results, serialize_to_path=str(out_p))

    logger.info("Exported INT8 TFLite: %s (%.2f MB)", out_p, out_p.stat().st_size / (1024 * 1024))
    return out_p


class TFLiteRunner:
    """Unified inference runner for FP32, FP16, and INT8 TFLite models."""

    def __init__(self, model_path: str | Path, num_threads: int = 4) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"TFLite model not found: {self.model_path}")

        self.interpreter = tflite.Interpreter(
            model_path=str(self.model_path),
            num_threads=num_threads,
        )
        self.interpreter.allocate_tensors()
        self.in_details = self.interpreter.get_input_details()[0]
        self.out_details = self.interpreter.get_output_details()[0]

        self.in_is_int8 = self.in_details["dtype"] == np.int8
        self.out_is_int8 = self.out_details["dtype"] == np.int8

        if self.in_is_int8:
            qp = self.in_details.get("quantization_parameters", {})
            self.scale_in = float(qp["scales"][0]) if len(qp.get("scales", [])) > 0 else 1.0
            self.zp_in = int(qp["zero_points"][0]) if len(qp.get("zero_points", [])) > 0 else 0

        if self.out_is_int8:
            qp = self.out_details.get("quantization_parameters", {})
            self.scale_out = float(qp["scales"][0]) if len(qp.get("scales", [])) > 0 else 1.0
            self.zp_out = int(qp["zero_points"][0]) if len(qp.get("zero_points", [])) > 0 else 0

    def predict_single(self, nhwc_float32: np.ndarray) -> np.ndarray:
        """Run inference on a single image [1, H, W, 3] or [H, W, 3] float32, returning [K] logits."""
        if nhwc_float32.ndim == 3:
            nhwc_float32 = np.expand_dims(nhwc_float32, 0)

        if self.in_is_int8:
            q_in = np.clip(
                np.round(nhwc_float32 / self.scale_in) + self.zp_in,
                -128,
                127,
            ).astype(np.int8)
            self.interpreter.set_tensor(self.in_details["index"], q_in)
        else:
            self.interpreter.set_tensor(self.in_details["index"], nhwc_float32.astype(np.float32))

        self.interpreter.invoke()
        raw_out = self.interpreter.get_tensor(self.out_details["index"])

        if self.out_is_int8:
            dequant = (raw_out.astype(np.float32) - self.zp_out) * self.scale_out
            return dequant[0]
        return raw_out[0].astype(np.float32)

    def predict_batch(self, nhwc_batch: np.ndarray) -> np.ndarray:
        """Run inference on batch [B, H, W, 3] float32, returning [B, K] logits."""
        if nhwc_batch.ndim == 3:
            nhwc_batch = np.expand_dims(nhwc_batch, 0)
        b = nhwc_batch.shape[0]
        results = [self.predict_single(nhwc_batch[i : i + 1]) for i in range(b)]
        return np.stack(results, axis=0)


def _default_model_prefix(model_name: str, input_size: int) -> str:
    if model_name == "mobilenetv4_conv_medium.e500_r256_in1k" and input_size == 256:
        return "mobilenetv4_medium_r256"
    base = model_name.split(".")[0].replace("/", "_")
    return f"{base}_r{input_size}"


def export_all_models(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    train_csv: str | Path = DEFAULT_TRAIN_CSV,
    image_dir: str | Path = DEFAULT_IMAGE_DIR,
    quant_dir: str | Path = DEFAULT_QUANT_DIR,
    num_calib_samples: int = 200,
    model_prefix: str | None = None,
    crop_mode: str = "none_stretch",
) -> dict[str, Any]:
    """Execute complete Stage P6 export of FP32, FP16, and INT8 models."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    q_dir = Path(quant_dir)
    q_dir.mkdir(parents=True, exist_ok=True)

    wrapped_model, meta = load_and_prepare_model(checkpoint_path, exclude_verna=True)
    input_size = meta["input_size"]
    prefix = model_prefix or _default_model_prefix(meta["model_name"], input_size)

    fp32_path = out_dir / f"{prefix}_fp32.tflite"
    fp16_path = out_dir / f"{prefix}_fp16.tflite"
    int8_path = out_dir / f"{prefix}_int8.tflite"

    # 1. FP32
    export_fp32(wrapped_model, input_size=input_size, output_path=fp32_path)
    # 2. FP16
    quantize_fp16(fp32_path, output_path=fp16_path)
    # 3. INT8
    quantize_int8(
        fp32_path,
        train_csv=train_csv,
        image_dir=image_dir,
        output_path=int8_path,
        num_calib_samples=num_calib_samples,
        input_size=input_size,
        crop_mode=crop_mode,
    )

    models_info = {}
    for name, path in [("fp32", fp32_path), ("fp16", fp16_path), ("int8", int8_path)]:
        size_bytes = path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        sha256 = compute_file_sha256(path)
        ops = inspect_tflite_operators(path)
        models_info[name] = {
            "path": str(path),
            "filename": path.name,
            "size_bytes": size_bytes,
            "size_mb": round(size_mb, 2),
            "sha256": sha256,
            "operators": ops,
        }
        logger.info(
            "Model %s: %s (%.2f MB, SHA256: %s)",
            name.upper(),
            path.name,
            size_mb,
            sha256,
        )

    export_manifest = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": meta["checkpoint_sha256"],
        "model_name": meta["model_name"],
        "input_size": input_size,
        "active_classes_count": len(meta["active_classes"]),
        "active_classes": meta["active_classes"],
        "models": models_info,
        "litert_torch_version": getattr(litert_torch, "__version__", "unknown"),
        "ai_edge_litert_version": getattr(ai_edge_litert, "__version__", "unknown"),
    }

    manifest_file = q_dir / "export_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(export_manifest, f, indent=2)

    logger.info("Saved export manifest to %s", manifest_file)
    return export_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="P6 TFLite Export and Quantization.")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT), help="Model checkpoint path")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory for .tflite")
    parser.add_argument("--train-csv", type=str, default=str(DEFAULT_TRAIN_CSV), help="Training CSV for INT8 calibration")
    parser.add_argument("--image-dir", type=str, default=str(DEFAULT_IMAGE_DIR), help="Image directory")
    parser.add_argument("--quant-dir", type=str, default=str(DEFAULT_QUANT_DIR), help="Quantization artifacts directory")
    parser.add_argument("--num-calib", type=int, default=200, help="Number of samples for INT8 calibration")
    parser.add_argument("--model-prefix", type=str, default=None, help="Output TFLite filename prefix")
    parser.add_argument("--crop-mode", type=str, default="none_stretch", help="Preprocessing crop mode for INT8 calib/drift")
    parser.add_argument("--verify-drift", action="store_true", help="Run quantization drift verification on val.csv after export")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    manifest = export_all_models(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        train_csv=args.train_csv,
        image_dir=args.image_dir,
        quant_dir=args.quant_dir,
        num_calib_samples=args.num_calib,
        model_prefix=args.model_prefix,
        crop_mode=args.crop_mode,
    )

    if args.verify_drift:
        from grzybobranie_ml.quantization_drift import verify_quantization_drift

        fp16_p = manifest["models"]["fp16"]["path"]
        int8_p = manifest["models"]["int8"]["path"]
        verify_quantization_drift(
            fp16_model_path=fp16_p,
            int8_model_path=int8_p,
            image_dir=args.image_dir,
            output_path=Path(args.quant_dir) / "quantization_drift_val.json",
            crop_mode=args.crop_mode,
        )


if __name__ == "__main__":
    main()
