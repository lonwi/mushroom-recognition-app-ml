"""Analyze violating DANGEROUS_PAIRS on validation set."""

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


def main() -> None:
    run_dir = Path("artifacts/train_runs/mobilenetv4_conv_small.e2400_r224_in1k_r224_1795463774")
    ckpt = torch.load(run_dir / "best_val_checkpoint.pt", map_location="cuda")

    model = create_classifier_model(
        "mobilenetv4_conv_small.e2400_r224_in1k",
        num_classes=67,
        pretrained=False,
    ).to("cuda")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    val_ds = MushroomSplitDataset(
        "data/splits/val.csv",
        image_dir="/home/bubolka/hf-768/upload",
        transform=get_val_transforms(224),
    )
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=4)

    class_to_idx = val_ds.class_to_idx
    num_classes = len(class_to_idx)

    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    per_class_total = np.zeros(num_classes, dtype=np.int64)

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to("cuda")
            targets = targets.to("cuda")
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(images)
            preds = torch.argmax(logits, dim=1)
            for t, p in zip(targets.cpu().numpy(), preds.cpu().numpy()):
                per_class_total[t] += 1
                confusion[t, p] += 1

    with open("configs/karty_sobowtory.json", encoding="utf-8") as f:
        karty_data = json.load(f)

    slug_to_pl = {}
    for card in karty_data.get("karty", []):
        pl_name = card.get("nazwa_pl", "")
        lat_name = card.get("nazwa_lacinska", "")
        status = card.get("status_jadalnosci", "")
        slug_to_pl[card["slug"]] = f"{pl_name} ({lat_name}) [{status}]"

    pairs = load_dangerous_pairs()
    violating = []

    for a, b in pairs:
        if a in class_to_idx and b in class_to_idx:
            ia, ib = class_to_idx[a], class_to_idx[b]
            na, nb = per_class_total[ia], per_class_total[ib]
            count_a_as_b = confusion[ia, ib]
            count_b_as_a = confusion[ib, ia]
            err_a_to_b = count_a_as_b / na if na > 0 else 0.0
            err_b_to_a = count_b_as_a / nb if nb > 0 else 0.0
            max_err = max(err_a_to_b, err_b_to_a)
            if max_err > 0.05:
                violating.append({
                    "species_a": a,
                    "name_a": slug_to_pl.get(a, a),
                    "species_b": b,
                    "name_b": slug_to_pl.get(b, b),
                    "count_a_as_b": int(count_a_as_b),
                    "n_a": int(na),
                    "err_a_to_b": round(float(err_a_to_b), 4),
                    "count_b_as_a": int(count_b_as_a),
                    "n_b": int(nb),
                    "err_b_to_a": round(float(err_b_to_a), 4),
                    "max_err": round(float(max_err), 4),
                })

    violating.sort(key=lambda x: x["max_err"], reverse=True)
    print(f"Total violating pairs (>0.05): {len(violating)}")
    print(json.dumps(violating, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
