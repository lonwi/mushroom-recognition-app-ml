#!/usr/bin/env python3
"""Initialize repository directories, split salt, and verify WSL2 GPU environment on a clean machine."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DIRS = [
    REPO_ROOT / ".agents" / "skills",
    REPO_ROOT / "configs",
    REPO_ROOT / "docs" / "decisions",
    REPO_ROOT / "schemas",
    REPO_ROOT / "scripts",
    REPO_ROOT / "src",
    REPO_ROOT / "tests",
    REPO_ROOT / "data" / "hf_raw",
    REPO_ROOT / "data" / "ood-v1",
    REPO_ROOT / "data" / "local_drop",
    REPO_ROOT / "data" / "splits",
    REPO_ROOT / "data" / "png_cache" / "224",
    REPO_ROOT / "data" / "png_cache" / "256",
    REPO_ROOT / "artifacts" / "checkpoints",
    REPO_ROOT / "artifacts" / "exports",
    REPO_ROOT / "artifacts" / "logs",
]

GITKEEP_DIRS = [d for d in REQUIRED_DIRS if "data" in d.parts or "artifacts" in d.parts]
SALT_PATH = Path.home() / ".config" / "grzybobranie" / "split_salt"


def check_filesystem() -> None:
    resolved = str(REPO_ROOT.resolve())
    if resolved.startswith("/mnt/"):
        print(f"⚠️ OSTRZEŻENIE: Repozytorium znajduje się na dysku Windows ({resolved}).")
        print("   Zgodnie z zasadami przenieś je na partycję ext4 w WSL2 (np. ~/grzybobranie-ml)!")
    else:
        print(f"✅ System plików OK (natywny Linux ext4): {resolved}")


def create_directories() -> None:
    for directory in REQUIRED_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
    for directory in GITKEEP_DIRS:
        gitkeep = directory / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()
    print(f"✅ Utworzono i zweryfikowano {len(REQUIRED_DIRS)} katalogów roboczych (z plikami .gitkeep).")


def ensure_split_salt() -> None:
    SALT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if SALT_PATH.is_file() and SALT_PATH.read_text(encoding="utf-8").strip():
        salt = SALT_PATH.read_text(encoding="utf-8").strip()
        fp = hashlib.sha256(salt.encode("utf-8")).hexdigest()[:8]
        print(f"✅ Klucz split_salt istnieje poza repo ({SALT_PATH}, fingerprint: {fp}).")
    else:
        salt = hashlib.sha256(os.urandom(32)).hexdigest()
        SALT_PATH.write_text(salt + "\n", encoding="utf-8")
        fp = hashlib.sha256(salt.encode("utf-8")).hexdigest()[:8]
        print(f"✅ Wygenerowano nowy klucz split_salt w {SALT_PATH} (fingerprint: {fp}).")


def check_gpu() -> None:
    try:
        import torch
        if not torch.cuda.is_available():
            print("ℹ️ PyTorch jest zainstalowany, ale CUDA jest niedostępna (używasz CPU).")
            return
        dev_name = torch.cuda.get_device_name(0)
        arch_list = torch.cuda.get_arch_list()
        cap = torch.cuda.get_device_capability(0)
        has_sm120 = "sm_120" in arch_list
        print(f"✅ GPU: {dev_name} | Capability: {cap} | sm_120: {has_sm120} | bf16: {torch.cuda.is_bf16_supported()}")
    except ImportError:
        print("ℹ️ PyTorch nie jest zainstalowany w tym środowisku. Wykonaj instalację według README.md.")


def main() -> int:
    print("=== Inicjalizacja środowiska Grzybobranie ML ===")
    check_filesystem()
    create_directories()
    ensure_split_salt()
    check_gpu()
    print("=== Gotowe! ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())