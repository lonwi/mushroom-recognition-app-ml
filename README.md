# Grzybobranie AI — Standalone Machine Learning Pipeline

This repository is the standalone Machine Learning (ML) pipeline for the **Grzybobranie AI** mobile application. When complete, it will cover dataset ingestion, local photo drop-in, leak-free train/val/test splitting, smart caching, `timm` + PyTorch training/fine-tuning, energy-based Out-of-Distribution (OOD) calibration, and LiteRT (`.tflite`) model export.

> **Status:** v0.1.0 — skeleton/bootstrap; pipeline under construction.

What exists today is the project rule set, packaging metadata (`pyproject.toml`), environment bootstrap (`scripts/init_env.py`), a working-directory layout under `data/` and `artifacts/`, and this documentation. The `grzybobranie_ml` package modules, configs, schemas, and tests are not implemented yet.

## 1. Key Architecture & Project Rules

1. **Whole-Image Classifier (**$K$ **Species, Zero Aggregate Classes):**
   * The model classifies full images into $K$ mushroom species (the `mini+` baseline targets $K = 67$, configurable via planned `configs/pokrycie-*.csv`).
   * There are **no** `not_a_mushroom` or `unknown_mushroom` classes in the model head.
   * Non-mushrooms and unknown species are rejected using an **Energy OOD Gate**: $E(x) = -T \cdot \text{logsumexp}(\text{logits} / T)$ and `minTop1Prob`.
   * Target species with insufficient training data are placed in `uncoveredSpecies` inside `model-manifest.json` (the app Atlas still displays them, while the scanner degrades gracefully). The thresholds are minimum training-image counts: $< 55$ for `high_stakes`/toxic species, $< 40$ for others. Do not add uncovered species to the head as `in_model: false`.

2. **Filesystem Requirement (Linux / WSL2 `ext4`):**
   * Clone and run this repository on a native Linux `ext4` filesystem (if using WSL2, clone inside the Linux home tree and **never on `/mnt/c`**, as DrvFS severely degrades I/O throughput during image caching and training).
   * The virtual environment lives at `.venv/` inside the repository. Datasets, caches, splits, checkpoints, logs, and exports live under `data/` and `artifacts/` in the repository (file contents are gitignored; the directory structure is kept via `.gitkeep`).

3. **Remote Storage (Hugging Face):**
   * Repository IDs are not hard-coded. Set them in a local `.env` file copied from `.env.example`:
     * `HF_DATASET_REPO` — private Hugging Face dataset repository (training and OOD manifests).
     * `HF_MODEL_REPO` — private Hugging Face model repository (tagged releases of `.tflite` + `model-manifest.json`).
   * Optional: `HF_TOKEN` for authenticated access (never commit a real token).
   * **Optional Local Image Seed:** If you have a local folder of previously downloaded images, a future `cache_manager` entry point will accept `--seed-dir /path/to/images` to avoid re-downloading existing files.

4. **Licensing & Safety Guardrails:**
   * **Training Photos:** Only `CC0`, `CC BY`, `CC BY-NC`, `CC BY-SA` (versioned), and `pd_self` are allowed.
   * **Forbidden Sources:** `DF20` and `DF24` (both images and pre-trained/fine-tuned weights `*ft_df20*`), AI-generated images, and `ND` licenses are strictly prohibited.
   * **Backbone Weights:** Only `Apache-2.0`, `MIT`, or `BSD` models from `timm` are allowed. `ConvNeXt V2` (`CC BY-NC`) and `MobileViT`/`FastViT` (Apple license) are disabled by default.

## 2. Repository Structure

```text
.
├── AGENTS.md                  # Rules for AI agents / bots
├── README.md                  # Human & developer guide (this file)
├── .env.example               # Template for HF repo IDs and optional token/salt path
├── pyproject.toml             # Python dependencies & pytest configuration
├── scripts/
│   └── init_env.py            # Environment, directory & split_salt initializer
├── src/
│   └── grzybobranie_ml/       # Package root (empty in v0.1.0; modules planned)
├── tests/                     # Planned: contract, preprocessing & math unit tests
├── configs/
│   ├── pokrycie-67.csv        # mini+ coverage and min training-image thresholds
│   └── karty_sobowtory.json   # look-alike / dangerous pairs
├── schemas/
│   └── model-manifest.schema.json  # model-manifest.json contract (v1.0.1)
├── docs/
│   ├── soul.md                # Training-bot constitution
│   ├── prompty.md             # Stage prompts P0–P9
│   ├── skille.md              # Suggested skills and connectors
│   └── decisions/
│       ├── decyzje.md
│       ├── krok3-decyzje.md
│       └── 2026-11-16-schema-v1-clarifications.md
├── .agents/
│   └── skills/                # Planned: local agent skills
├── data/                      # [GITIGNORED CONTENT] Working data (structure kept)
│   ├── hf_raw/                # Planned: downloaded Hugging Face CSV manifests
│   ├── ood-v1/                # Planned: OOD calibration/test set & ood_ids.txt
│   ├── local_drop/            # Planned: staging folder for custom/local photo batches
│   ├── splits/                # Planned: attributions.jsonl & splits.json
│   └── png_cache/             # Planned: preprocessed 224x224 and 256x256 PNGs
└── artifacts/                 # [GITIGNORED CONTENT] Training outputs (structure kept)
    ├── checkpoints/           # Planned: best PyTorch checkpoints (.pth)
    ├── exports/               # Planned: exported .tflite models & model-manifest.json
    └── logs/                  # Planned: training JSONL & Trackio logs
```

Still planned and not in this repository yet: `data/local_drop/example_drop.csv` and content under `.agents/skills/`.

## 3. Quickstart (Fresh Machine Setup)

### Prerequisites
* **OS:** Linux or WSL2 (Ubuntu 22.04 / 24.04 on `ext4`). If using WSL2, install the NVIDIA GPU driver on Windows only (never install `cuda-drivers` or `nvidia-driver-*` inside WSL2).
* **Python:** `>= 3.11` with `venv` (`sudo apt update && sudo apt install -y python3-venv python3-pip git`).
* **GPU:** NVIDIA GPU with CUDA support (for RTX 50-series Blackwell `sm_120`, use CUDA $\ge 12.8$ / $13.x$ wheels).

### Step 1: Clone the Repository & Create Virtual Environment

```bash
git clone git@github.com:<your-org>/<repo>.git
cd <repo>

# Create and activate a clean virtual environment inside the repo
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### Step 2: Install PyTorch (CUDA) & Project Dependencies

```bash
# 1. Install PyTorch with CUDA support (use standard PyPI or the appropriate CUDA wheel index)
pip install torch torchvision

# 2. Install repository dependencies and configure PYTHONPATH for src/
pip install -e ".[dev]"

# 3. Initialize working directories, verify the repo is not on /mnt/*, and ensure split_salt exists
python3 scripts/init_env.py
```

> **Note on `split_salt`:** `scripts/init_env.py` checks `~/.config/grzybobranie/split_salt` (override later via `GRZYBOBRANIE_SPLIT_SALT_PATH` in `.env`; see `.env.example`). To reproduce an exact historical train/val/test split on a new machine, copy your existing `split_salt` file to that path before running `python -m grzybobranie_ml.prepare_data`. Otherwise, a new cryptographic salt is generated automatically. Never commit the salt or a filled `.env`.

### Step 3: Configure Hugging Face Access

```bash
cp .env.example .env   # then set HF_DATASET_REPO, HF_MODEL_REPO, and HF_TOKEN if needed
hf auth login
hf auth whoami
```

## 4. Roadmap (P0–P9) & Planned CLI

All Python entry points use the package layout `python -m grzybobranie_ml.<module>` from the repo virtualenv (`.venv`). These modules are **not implemented in v0.1.0**; the commands below are the target interface.

| Stage | Status | Purpose |
|---|---|---|
| **P0** Environment bootstrap & inventory | **Partial** | Working directories, split salt, and GPU/filesystem checks exist (`scripts/init_env.py`). A written inventory, package modules, configs, and schemas are still planned. |
| **P1** Data audit & leak-free split | **Planned** | Ingest HF manifests + optional `data/local_drop/`, enforce the license filter, drop OOD IDs, HMAC-salted connected-component split (70/15/15), PNG cache. |
| **P2** Architecture candidates | **Planned** | Select 2–3 `timm` backbones with allowed licenses, for input sizes 224 and 256. |
| **P3** Training & fine-tuning | **Planned** | Frozen head, then full fine-tuning; checkpoints under `artifacts/checkpoints/`. |
| **P4** Temperature scaling & ECE | **Planned** | Fit $T$ and report ECE on `.tflite` logits. |
| **P5** Energy OOD gate & fusion | **Planned** | Calibrate `energyThreshold`, `minTop1Prob`, and 1–3 photo fusion on `data/ood-v1/`. |
| **P6** LiteRT export & quantization | **Planned** | Export `.tflite` (fp16 / int8) under `artifacts/exports/`. |
| **P7** On-device parity | **Planned** | Preprocessing and decision fixtures; phone measurements before locking input size. |
| **P8** Ship gates & model card | **Planned** | Fail-closed acceptance gates and `docs/model-card.md`. |
| **P9** Manifest release | **Planned** | Validate and publish `model-manifest.json` to `HF_MODEL_REPO`. |

Planned commands (do not run them yet; the modules are missing):

```bash
python -m grzybobranie_ml.local_drop \
  --images-dir /path/to/my_photos \
  --metadata /path/to/metadata.csv \
  --push-to-hf          # optional: uploads a validated batch to HF_DATASET_REPO

python -m grzybobranie_ml.prepare_data --coverage-csv configs/pokrycie-67.csv
python -m grzybobranie_ml.cache_manager --sizes 224 256 --workers 8

python -m grzybobranie_ml.train \
  --model mobilenetv4_conv_small.e2400_r224_in1k \
  --input-size 224

python -m grzybobranie_ml.train \
  --finetune-from artifacts/checkpoints/mobilenetv4_conv_small_224_best.pth \
  --input-size 224 \
  --epochs-frozen 2 \
  --epochs-finetune 8 \
  --lr-backbone 5e-5

python -m grzybobranie_ml.export_tflite \
  --checkpoint artifacts/checkpoints/mobilenetv4_conv_small_224_best.pth
python -m grzybobranie_ml.calibrate_ood \
  --tflite artifacts/exports/grzybobranie-m1.tflite
python -m grzybobranie_ml.ship_gates \
  --manifest artifacts/exports/model-manifest.json
python -m grzybobranie_ml.manifest
```

Unit tests (none collected in v0.1.0):

```bash
pytest
```

## 5. Connecting an AI Agent or Bot

Read `AGENTS.md` before changing this repository. The training-bot constitution and stage prompts are in `docs/soul.md` and `docs/prompty.md` (P0–P9). Local skills under `.agents/skills/` may be added later.

## 6. Licencja

Ten projekt jest udostępniany na licencji [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). Pełny tekst znajduje się w pliku `LICENSE`.
