"""Extract and cache logits for val, amanita_verna, and OOD datasets."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from grzybobranie_ml.train import (
    MushroomSplitDataset,
    create_classifier_model,
    get_val_transforms,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("extract_logits")


class GenericOODDataset(Dataset):
    """Dataset for OOD images mapped through manifest_768.csv."""

    def __init__(
        self,
        csv_path: str | Path,
        image_dir: str | Path,
        transform: Any = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.image_dir = Path(image_dir)
        self.transform = transform

        # Build path lookup
        self.path_lookup: dict[str, Path] = {}
        manifest_file = self.image_dir / "manifest_768.csv"
        if manifest_file.exists():
            with open(manifest_file, encoding="utf-8") as mf:
                for r in csv.DictReader(mf):
                    rel = r.get("path")
                    if rel:
                        full_p = self.image_dir / rel
                        if r.get("photo_id"):
                            self.path_lookup[r["photo_id"]] = full_p
                        if r.get("image_url"):
                            self.path_lookup[r["image_url"]] = full_p
                        if r.get("source_url"):
                            self.path_lookup[r["source_url"]] = full_p

        self.records: list[dict[str, str]] = []
        self.image_paths: list[Path] = []

        with open(self.csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                pid = row.get("photo_id")
                url = row.get("image_url")
                src = row.get("source_url")
                p = self.path_lookup.get(url) or self.path_lookup.get(pid) or self.path_lookup.get(src)
                if p and p.exists():
                    self.records.append(row)
                    self.image_paths.append(p)
                else:
                    raise FileNotFoundError(f"OOD image not found for row: {pid} / {url}")

        logger.info("Loaded OOD dataset %s: %d images", self.csv_path.name, len(self.records))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        p = self.image_paths[idx]
        with Image.open(p) as img:
            image = img.convert("RGB")
        if self.transform is not None:
            tensor = self.transform(image)
        else:
            import torchvision.transforms as T
            tensor = T.ToTensor()(image)
        return tensor, idx


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    ckpt_path = Path("artifacts/train_runs/mobilenetv4_conv_medium.e500_r256_in1k_r256_1795514748/best_val_checkpoint.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model_name = ckpt["model_name"]
    input_size = ckpt["input_size"]
    class_to_idx_67 = ckpt["class_to_idx"]
    sorted_classes = sorted(class_to_idx_67.keys())
    verna_idx = class_to_idx_67["amanita_verna"]
    assert verna_idx == 12

    # Active 66 classes
    active_classes = [c for c in sorted_classes if c != "amanita_verna"]
    class_to_idx_66 = {c: i for i, c in enumerate(active_classes)}
    idx_to_class_66 = {i: c for i, c in enumerate(active_classes)}

    logger.info("Model: %s, input_size: %d, active classes: %d (verna excluded at 12)", model_name, input_size, len(active_classes))

    # Create model and load EMA state
    model = create_classifier_model(model_name, num_classes=67, pretrained=False).to(device)
    model.load_state_dict(ckpt["ema_state_dict"])
    model.eval()

    image_dir = "/home/bubolka/hf-768/upload"
    transform = get_val_transforms(input_size)

    # 1. Val dataset (66 classes)
    val_ds = MushroomSplitDataset(
        "data/splits/val.csv",
        image_dir=image_dir,
        transform=transform,
        class_to_idx=class_to_idx_66,
    )
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)

    # 2. Amanita verna dataset (11 records)
    verna_ds = MushroomSplitDataset(
        "data/splits/val.csv",
        image_dir=image_dir,
        transform=transform,
        class_to_idx={"amanita_verna": 0},
    )
    verna_loader = DataLoader(verna_ds, batch_size=128, shuffle=False, num_workers=4)

    # 3. OOD fungi
    ood_fungi_ds = GenericOODDataset("data/ood-v1/ood_fungi.csv", image_dir=image_dir, transform=transform)
    ood_fungi_loader = DataLoader(ood_fungi_ds, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)

    # 4. OOD nonfungi
    ood_nonfungi_ds = GenericOODDataset("data/ood-v1/ood_nonfungi.csv", image_dir=image_dir, transform=transform)
    ood_nonfungi_loader = DataLoader(ood_nonfungi_ds, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)

    def extract(loader: DataLoader, desc: str) -> tuple[torch.Tensor, list[int]]:
        logger.info("Extracting %s (%d batches)...", desc, len(loader))
        all_logits = []
        all_targets = []
        with torch.no_grad():
            for images, targets in loader:
                images = images.to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                    out67 = model(images)
                # Slice out index 12 (amanita_verna) to obtain 66 logits
                out66 = torch.cat([out67[:, :12], out67[:, 13:]], dim=1)
                all_logits.append(out66.cpu().float())
                all_targets.extend(targets.tolist())
        return torch.cat(all_logits, dim=0), all_targets

    val_logits, val_targets = extract(val_loader, "val (66 classes)")
    verna_logits, _ = extract(verna_loader, "amanita_verna (11 images)")
    ood_fungi_logits, _ = extract(ood_fungi_loader, "ood_fungi (2580 images)")
    ood_nonfungi_logits, _ = extract(ood_nonfungi_loader, "ood_nonfungi (1880 images)")

    import hashlib

    def compute_sha256(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(1024 * 1024):
                h.update(chunk)
        return h.hexdigest()

    ckpt_sha256 = compute_sha256(ckpt_path)
    output_path = Path("artifacts/calibration/extracted_logits.pt")
    torch.save(
        {
            "checkpoint_path": str(ckpt_path),
            "checkpoint_sha256": ckpt_sha256,
            "class_to_idx_66": class_to_idx_66,
            "idx_to_class_66": idx_to_class_66,
            "active_classes": active_classes,
            "val_logits": val_logits,
            "val_targets": val_targets,
            "val_records": val_ds.records,
            "verna_logits": verna_logits,
            "verna_records": verna_ds.records,
            "ood_fungi_logits": ood_fungi_logits,
            "ood_fungi_records": ood_fungi_ds.records,
            "ood_nonfungi_logits": ood_nonfungi_logits,
            "ood_nonfungi_records": ood_nonfungi_ds.records,
        },
        output_path,
    )
    logger.info("Saved extracted logits to %s (size: %.2f MB)", output_path, output_path.stat().st_size / (1024 * 1024))


if __name__ == "__main__":
    main()
