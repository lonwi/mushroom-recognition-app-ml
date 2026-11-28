"""Unit tests for LiteRT / TFLite export module (grzybobranie_ml.export_tflite)."""

import hashlib
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from grzybobranie_ml.export_tflite import (
    NHWCWrapper,
    TFLiteRunner,
    compute_file_sha256,
    inspect_tflite_operators,
)


def test_nhwc_wrapper() -> None:
    """NHWCWrapper permutes [B, H, W, C] to [B, C, H, W] for the underlying model."""
    class DummyNCHWModel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            assert x.shape == (2, 3, 32, 32), f"Expected NCHW shape (2, 3, 32, 32), got {x.shape}"
            return x.mean(dim=(2, 3))  # [2, 3]

    dummy = DummyNCHWModel()
    wrapper = NHWCWrapper(dummy)

    nhwc_input = torch.randn(2, 32, 32, 3)
    out = wrapper(nhwc_input)
    assert out.shape == (2, 3)


def test_compute_file_sha256() -> None:
    """compute_file_sha256 accurately calculates hex digest."""
    with tempfile.NamedTemporaryFile("wb", delete=False) as f:
        data = b"Grzybobranie AI TFLite Export Test 2026"
        f.write(data)
        f_path = Path(f.name)

    try:
        expected = hashlib.sha256(data).hexdigest()
        assert compute_file_sha256(f_path) == expected
    finally:
        f_path.unlink()


def test_classifier_slicing_logic() -> None:
    """Ensure head slicing removes the target class while preserving all other class weights/biases."""
    in_features = 128
    num_classes = 67
    target_idx = 12

    weight = torch.randn(num_classes, in_features)
    bias = torch.randn(num_classes)

    linear = nn.Linear(in_features, num_classes)
    linear.weight.data.copy_(weight)
    linear.bias.data.copy_(bias)

    # Perform slicing as in load_and_prepare_model
    sliced_w = torch.cat([linear.weight.data[:target_idx], linear.weight.data[target_idx + 1:]], dim=0)
    sliced_b = torch.cat([linear.bias.data[:target_idx], linear.bias.data[target_idx + 1:]], dim=0)

    assert sliced_w.shape == (66, in_features)
    assert sliced_b.shape == (66,)

    # Verify classes before target_idx match exactly
    assert torch.equal(sliced_w[:target_idx], weight[:target_idx])
    assert torch.equal(sliced_b[:target_idx], bias[:target_idx])

    # Verify classes after target_idx match exactly shifted by 1
    assert torch.equal(sliced_w[target_idx:], weight[target_idx + 1:])
    assert torch.equal(sliced_b[target_idx:], bias[target_idx + 1:])


@pytest.mark.skipif(
    not Path("artifacts/models/mobilenetv4_medium_r256_fp16.tflite").exists(),
    reason="mobilenetv4_medium_r256_fp16.tflite model not found",
)
def test_tflite_runner_fp16() -> None:
    """TFLiteRunner loads FP16 model and performs inference on single and batch inputs."""
    model_path = "artifacts/models/mobilenetv4_medium_r256_fp16.tflite"
    runner = TFLiteRunner(model_path, num_threads=2)

    # Single image [256, 256, 3]
    dummy_single = np.zeros((256, 256, 3), dtype=np.float32)
    logits_single = runner.predict_single(dummy_single)
    assert isinstance(logits_single, np.ndarray)
    assert logits_single.shape == (66,)
    assert not np.isnan(logits_single).any()

    # Batch of 2 images [2, 256, 256, 3]
    dummy_batch = np.zeros((2, 256, 256, 3), dtype=np.float32)
    logits_batch = runner.predict_batch(dummy_batch)
    assert isinstance(logits_batch, np.ndarray)
    assert logits_batch.shape == (2, 66)
    np.testing.assert_allclose(logits_batch[0], logits_single, atol=1e-5)


@pytest.mark.skipif(
    not Path("artifacts/models/mobilenetv4_medium_r256_int8.tflite").exists(),
    reason="mobilenetv4_medium_r256_int8.tflite model not found",
)
def test_tflite_runner_int8() -> None:
    """TFLiteRunner correctly handles quantized INT8 input/output and returns dequantized float32 logits."""
    model_path = "artifacts/models/mobilenetv4_medium_r256_int8.tflite"
    runner = TFLiteRunner(model_path, num_threads=2)
    assert runner.in_is_int8 is True

    dummy_single = np.zeros((256, 256, 3), dtype=np.float32)
    logits_single = runner.predict_single(dummy_single)
    assert logits_single.shape == (66,)
    assert logits_single.dtype == np.float32
    assert not np.isnan(logits_single).any()


@pytest.mark.skipif(
    not Path("artifacts/models/mobilenetv4_medium_r256_fp16.tflite").exists(),
    reason="mobilenetv4_medium_r256_fp16.tflite model not found",
)
def test_inspect_tflite_operators() -> None:
    """inspect_tflite_operators lists valid opcodes from TFLite flatbuffer."""
    model_path = "artifacts/models/mobilenetv4_medium_r256_fp16.tflite"
    ops = inspect_tflite_operators(model_path)
    assert isinstance(ops, list)
    assert len(ops) > 0
    # MobileNetV4 contains CONV_2D and DEQUANTIZE
    assert any("CONV" in op for op in ops)
    # Ensure no raw integer codes like "6" remain
    assert "6" not in ops
    assert "DEQUANTIZE" in ops


def test_dummy_export_and_fp16_quantize() -> None:
    """End-to-end unit test: export dummy NHWC model to FP32 and quantize to FP16 in tempdir."""
    from grzybobranie_ml.export_tflite import export_fp32, quantize_fp16

    class TinyClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(8, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(self.pool(torch.relu(self.conv(x))).flatten(1))

    tiny_model = NHWCWrapper(TinyClassifier()).eval()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_p = Path(tmp_dir)
        fp32_p = tmp_p / "tiny_fp32.tflite"
        fp16_p = tmp_p / "tiny_fp16.tflite"

        export_fp32(tiny_model, input_size=16, output_path=fp32_p)
        assert fp32_p.exists()
        assert fp32_p.stat().st_size > 0

        quantize_fp16(fp32_p, output_path=fp16_p)
        assert fp16_p.exists()
        assert fp16_p.stat().st_size > 0

        # Verify runner works on the exported model
        runner = TFLiteRunner(fp16_p, num_threads=1)
        dummy_in = np.ones((16, 16, 3), dtype=np.float32)
        out = runner.predict_single(dummy_in)
        assert out.shape == (4,)
        assert not np.isnan(out).any()

        # Check operator inspection on newly exported model
        ops = inspect_tflite_operators(fp16_p)
        assert "DEQUANTIZE" in ops
        assert "6" not in ops


def test_disable_fused_attention_helper() -> None:
    """_disable_fused_attention sets fused_attn=False on nested attention modules."""
    from grzybobranie_ml.export_tflite import _disable_fused_attention

    class MockAttn(nn.Module):
        def __init__(self, fused: bool) -> None:
            super().__init__()
            self.fused_attn = fused

    class MockModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a1 = MockAttn(True)
            self.a2 = MockAttn(False)
            self.sub = nn.Sequential(MockAttn(True))

    m = MockModel()
    count = _disable_fused_attention(m)
    assert count == 2
    assert m.a1.fused_attn is False
    assert m.a2.fused_attn is False
    assert m.sub[0].fused_attn is False


