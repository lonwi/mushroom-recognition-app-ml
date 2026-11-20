"""Benchmark script for Stage P2 architecture candidates on mobile constraints.

Evaluates:
- mobilenetv3_large_100.ra4_e3600_r224_in1k
- mobilenetv4_conv_small.e2400_r224_in1k
- mobilenetv4_conv_medium.e500_r256_in1k
- mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k
- tf_efficientnet_lite0.in1k
- efficientnet_b0.ra4_e3600_r224_in1k

Measures:
- Parameter counts (1000 classes vs 67 classes)
- Arithmetic operations (GMACs) for 224x224 and 256x256
- Export to .tflite via litert_torch
- LiteRT runtime latency (XNNPACK CPU)
- Model file sizes (FP32, FP16, INT8 estimates)
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import ai_edge_litert.interpreter as tflite
import litert_torch
import numpy as np
import timm
import torch

MODELS = [
    {
        "name": "mobilenetv4_conv_small.e2400_r224_in1k",
        "family": "MobileNetV4",
        "native_res": 224,
        "license": "Apache-2.0",
        "doc_source": "https://arxiv.org/abs/2404.10518",
    },
    {
        "name": "mobilenetv3_large_100.ra4_e3600_r224_in1k",
        "family": "MobileNetV3",
        "native_res": 224,
        "license": "Apache-2.0",
        "doc_source": "https://arxiv.org/abs/1905.02244",
    },
    {
        "name": "tf_efficientnet_lite0.in1k",
        "family": "EfficientNet-Lite",
        "native_res": 224,
        "license": "Apache-2.0",
        "doc_source": "https://blog.tensorflow.org/2020/03/higher-accuracy-on-vision-models-with-efficientnet-lite.html",
    },
    {
        "name": "efficientnet_b0.ra4_e3600_r224_in1k",
        "family": "EfficientNet",
        "native_res": 224,
        "license": "Apache-2.0",
        "doc_source": "https://arxiv.org/abs/1905.11946",
    },
    {
        "name": "mobilenetv4_conv_medium.e500_r256_in1k",
        "family": "MobileNetV4",
        "native_res": 256,
        "license": "Apache-2.0",
        "doc_source": "https://huggingface.co/timm/mobilenetv4_conv_medium.e500_r256_in1k",
    },
    {
        "name": "mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k",
        "family": "MobileNetV4 (Hybrid)",
        "native_res": 256,
        "license": "Apache-2.0",
        "doc_source": "https://huggingface.co/timm/mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k",
    },
]

OUTPUT_DIR = "artifacts/benchmark"


def count_gmacs(model: torch.nn.Module, x: torch.Tensor) -> float:
    """Calculate GMACs for Conv2d and Linear layers."""
    total_macs = [0]
    hooks = []

    def hook_conv(module, inp, out):
        b, c_out, h_out, w_out = out.shape
        c_in = module.in_channels
        k_h, k_w = module.kernel_size
        groups = module.groups
        macs = b * c_out * h_out * w_out * (c_in // groups) * k_h * k_w
        total_macs[0] += macs

    def hook_linear(module, inp, out):
        b, out_f = out.shape
        in_f = module.in_features
        macs = b * out_f * in_f
        total_macs[0] += macs

    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            hooks.append(m.register_forward_hook(hook_conv))
        elif isinstance(m, torch.nn.Linear):
            hooks.append(m.register_forward_hook(hook_linear))

    with torch.no_grad():
        _ = model(x)

    for h in hooks:
        h.remove()

    return total_macs[0] / 1e9


def benchmark_litert_file(tflite_path: str, num_threads: int = 4, iters: int = 40) -> float:
    """Benchmark TFLite model using LiteRT interpreter with XNNPACK on CPU."""
    interp = tflite.Interpreter(model_path=tflite_path, num_threads=num_threads)
    interp.allocate_tensors()
    in_details = interp.get_input_details()
    dummy = np.random.randn(*in_details[0]["shape"]).astype(np.float32)

    # Warmup
    for _ in range(5):
        interp.set_tensor(in_details[0]["index"], dummy)
        interp.invoke()

    t0 = time.perf_counter()
    for _ in range(iters):
        interp.set_tensor(in_details[0]["index"], dummy)
        interp.invoke()
    t1 = time.perf_counter()
    return (t1 - t0) / iters * 1000


def run_benchmarks() -> list[dict[str, Any]]:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = []

    print("=========================================================================================")
    print("STAGE P2: ARCHITECTURE BENCHMARK MATRIX (K=67, 224x224 & 256x256)")
    print("=========================================================================================")

    for spec in MODELS:
        name = spec["name"]
        print(f"\nEvaluating candidate: {name} ({spec['family']})...")

        # 1. Parameter counts
        m_1k = timm.create_model(name, num_classes=1000, pretrained=False)
        params_1k = sum(p.numel() for p in m_1k.parameters()) / 1e6

        model = timm.create_model(name, num_classes=67, pretrained=False)
        model.eval()
        params_67 = sum(p.numel() for p in model.parameters()) / 1e6

        # 2. Arithmetic complexity (GMACs)
        x224 = torch.randn(1, 3, 224, 224)
        x256 = torch.randn(1, 3, 256, 256)
        gmacs_224 = count_gmacs(model, x224)
        gmacs_256 = count_gmacs(model, x256)

        # 3. Export to LiteRT .tflite for both resolutions
        slug = name.split(".")[0]

        # 224 export
        tflite_224_path = os.path.join(OUTPUT_DIR, f"{slug}_r224.tflite")
        print(f"  Converting 224x224 -> {tflite_224_path}...")
        edge_224 = litert_torch.convert(model, (x224,))
        edge_224.export(tflite_224_path)
        size_224_bytes = os.path.getsize(tflite_224_path)

        # 256 export
        tflite_256_path = os.path.join(OUTPUT_DIR, f"{slug}_r256.tflite")
        print(f"  Converting 256x256 -> {tflite_256_path}...")
        edge_256 = litert_torch.convert(model, (x256,))
        edge_256.export(tflite_256_path)
        size_256_bytes = os.path.getsize(tflite_256_path)

        # 4. LiteRT Latency on CPU (4 threads, XNNPACK)
        lat_litert_224 = benchmark_litert_file(tflite_224_path, num_threads=4)
        lat_litert_256 = benchmark_litert_file(tflite_256_path, num_threads=4)

        entry = {
            "model_name": name,
            "family": spec["family"],
            "license": spec["license"],
            "doc_source": spec["doc_source"],
            "native_res": spec["native_res"],
            "params_1k_m": round(params_1k, 2),
            "params_67_m": round(params_67, 2),
            "gmacs_224": round(gmacs_224, 3),
            "gmacs_256": round(gmacs_256, 3),
            "size_fp32_mb": round(size_224_bytes / 1024 / 1024, 2),
            "size_256_fp32_mb": round(size_256_bytes / 1024 / 1024, 2),
            "size_fp16_est_mb": round((size_224_bytes / 2) / 1024 / 1024, 2),
            "size_int8_est_mb": round((size_224_bytes / 4) / 1024 / 1024, 2),
            "litert_cpu_ms_224": round(lat_litert_224, 2),
            "litert_cpu_ms_256": round(lat_litert_256, 2),
            "tflite_224_file": tflite_224_path,
            "tflite_256_file": tflite_256_path,
        }
        results.append(entry)
        print(
            f"  [RESULT] Params: {entry['params_67_m']}M | Size: {entry['size_fp32_mb']}MB FP32 "
            f"({entry['size_int8_est_mb']}MB INT8) | GMACs: {entry['gmacs_224']} (224) / {entry['gmacs_256']} (256) | "
            f"LiteRT CPU: {entry['litert_cpu_ms_224']} ms (224) / {entry['litert_cpu_ms_256']} ms (256)"
        )

    out_json = os.path.join(OUTPUT_DIR, "p2_architecture_comparison.json")
    with open(out_json, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)
    print(f"\nSaved full benchmark results to {out_json}")
    return results


if __name__ == "__main__":
    run_benchmarks()
