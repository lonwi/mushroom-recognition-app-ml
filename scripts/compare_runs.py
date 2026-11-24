"""Compare baseline 224 run vs 256 run on validation metrics and dangerous pairs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from grzybobranie_ml.train import (
    MushroomSplitDataset,
    create_classifier_model,
    get_val_transforms,
    load_dangerous_pairs,
)


def evaluate_checkpoint(run_dir: str | Path, model_name: str, input_size: int) -> dict:
    ckpt_path = Path(run_dir) / "best_val_checkpoint.pt"
    ckpt = torch.load(ckpt_path, map_location="cuda")
    model = create_classifier_model(model_name, num_classes=67, pretrained=False).to("cuda")
    state = ckpt.get("ema_state_dict", ckpt.get("model_state_dict"))
    model.load_state_dict(state)
    model.eval()

    val_ds = MushroomSplitDataset(
        "data/splits/val.csv",
        "/home/bubolka/hf-768/upload",
        transform=get_val_transforms(input_size),
    )
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)

    class_to_idx = val_ds.class_to_idx
    num_classes = len(class_to_idx)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    per_class_total = np.zeros(num_classes, dtype=np.int64)
    per_class_correct = np.zeros(num_classes, dtype=np.int64)
    correct_top1 = 0
    correct_top5 = 0
    total_samples = 0

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to("cuda", non_blocking=True)
            targets = targets.to("cuda", non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(images)
            preds_top1 = torch.argmax(logits, dim=1)
            _, preds_top5 = torch.topk(logits, k=5, dim=1)

            correct_top1 += (preds_top1 == targets).sum().item()
            correct_top5 += (preds_top5 == targets.view(-1, 1).expand_as(preds_top5)).any(dim=1).sum().item()
            total_samples += images.size(0)

            for t, p in zip(targets.cpu().numpy(), preds_top1.cpu().numpy()):
                per_class_total[t] += 1
                if t == p:
                    per_class_correct[t] += 1
                confusion[t, p] += 1

    recalls = np.divide(
        per_class_correct,
        per_class_total,
        out=np.zeros(num_classes, dtype=np.float64),
        where=per_class_total > 0,
    )
    macro_top1 = float(np.mean(recalls))
    overall_top1 = float(correct_top1 / total_samples)
    overall_top5 = float(correct_top5 / total_samples)

    return {
        "confusion": confusion,
        "per_class_total": per_class_total,
        "class_to_idx": class_to_idx,
        "overall_top1": overall_top1,
        "overall_top5": overall_top5,
        "macro_top1": macro_top1,
        "recalls": recalls,
    }


def main() -> None:
    res224 = evaluate_checkpoint(
        "artifacts/train_runs/mobilenetv4_conv_small.e2400_r224_in1k_r224_1795463774",
        "mobilenetv4_conv_small.e2400_r224_in1k",
        224,
    )
    res256 = evaluate_checkpoint(
        "artifacts/train_runs/mobilenetv4_conv_medium.e500_r256_in1k_r256_1795514748",
        "mobilenetv4_conv_medium.e500_r256_in1k",
        256,
    )

    print("=================== OVERALL COMPARISON ===================")
    print(f"Baseline 224 (Small)  | Top-1: {res224['overall_top1']*100:.2f}% | Top-5: {res224['overall_top5']*100:.2f}% | Macro Top-1: {res224['macro_top1']*100:.2f}%")
    print(f"Candidate 256 (Medium)| Top-1: {res256['overall_top1']*100:.2f}% | Top-5: {res256['overall_top5']*100:.2f}% | Macro Top-1: {res256['macro_top1']*100:.2f}%")
    delta_top1 = (res256['overall_top1'] - res224['overall_top1']) * 100
    delta_macro = (res256['macro_top1'] - res224['macro_top1']) * 100
    print(f"DELTA:                 Top-1: {delta_top1:+.2f} pp | Macro Top-1: {delta_macro:+.2f} pp")

    with open("configs/karty_sobowtory.json", encoding="utf-8") as f:
        karty_data = json.load(f)

    slug_to_pl = {c["slug"]: f"{c['nazwa_pl']} ({c['nazwa_lacinska']}) [{c['status_jadalnosci']}]" for c in karty_data.get("karty", [])}
    pairs = load_dangerous_pairs()
    c2i = res256["class_to_idx"]

    comparison_pairs = []
    for a, b in pairs:
        if a in c2i and b in c2i:
            ia, ib = c2i[a], c2i[b]
            # 224
            na224, nb224 = res224["per_class_total"][ia], res224["per_class_total"][ib]
            eab224 = res224["confusion"][ia, ib] / na224 if na224 > 0 else 0.0
            eba224 = res224["confusion"][ib, ia] / nb224 if nb224 > 0 else 0.0
            max224 = max(eab224, eba224)

            # 256
            na256, nb256 = res256["per_class_total"][ia], res256["per_class_total"][ib]
            eab256 = res256["confusion"][ia, ib] / na256 if na256 > 0 else 0.0
            eba256 = res256["confusion"][ib, ia] / nb256 if nb256 > 0 else 0.0
            max256 = max(eab256, eba256)

            comparison_pairs.append({
                "slug_a": a,
                "name_a": slug_to_pl.get(a, a),
                "slug_b": b,
                "name_b": slug_to_pl.get(b, b),
                "max224": float(max224),
                "max256": float(max256),
                "cab256": int(res256["confusion"][ia, ib]),
                "na256": int(na256),
                "eab256": float(eab256),
                "cba256": int(res256["confusion"][ib, ia]),
                "nb256": int(nb256),
                "eba256": float(eba256),
                "violating_224": bool(max224 > 0.05),
                "violating_256": bool(max256 > 0.05),
            })

    # Sort by 256 max error descending
    comparison_pairs.sort(key=lambda x: x["max256"], reverse=True)

    print("\n=================== PAIRS EXCEEDING 0.05 IN EITHER RUN ===================")
    for p in comparison_pairs:
        if p["violating_224"] or p["violating_256"]:
            print(f"{p['slug_a']} <-> {p['slug_b']}")
            print(f"  224 max err: {p['max224']*100:.2f}% | 256 max err: {p['max256']*100:.2f}% | Delta: {(p['max256']-p['max224'])*100:+.2f} pp")
            print(f"  256 details: {p['slug_a']}->{p['slug_b']}: {p['cab256']}/{p['na256']} ({p['eab256']*100:.2f}%) | {p['slug_b']}->{p['slug_a']}: {p['cba256']}/{p['nb256']} ({p['eba256']*100:.2f}%)")

    # High stakes recall comparison
    from grzybobranie_ml.train import load_high_stakes_species
    high_stakes = load_high_stakes_species()
    print("\n=================== HIGH STAKES SPECIES RECALL ===================")
    hs_data = []
    for slug in high_stakes:
        if slug in c2i:
            idx = c2i[slug]
            r224 = float(res224["recalls"][idx])
            r256 = float(res256["recalls"][idx])
            n = int(res256["per_class_total"][idx])
            hs_data.append((slug, n, r224, r256))

    hs_data.sort(key=lambda x: x[3])
    for slug, n, r224, r256 in hs_data:
        print(f"{slug:<30} (n={n:<3}): 224: {r224*100:.2f}% | 256: {r256*100:.2f}% | Delta: {(r256-r224)*100:+.2f} pp")


if __name__ == "__main__":
    main()
