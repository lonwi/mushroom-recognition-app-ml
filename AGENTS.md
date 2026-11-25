# AGENTS.md — Grzybobranie AI Machine Learning Repository

This repository is the standalone ML pipeline for **Grzybobranie AI** (Stages P0–P6 implemented). Any AI agent, bot, or human developer MUST read `README.md` first and follow these rules before running or modifying code.

## 1. Project rules

1. **Whole-image classifier, no aggregate classes.** The model maps a full photo to $K$ species. `not_a_mushroom` and `unknown_mushroom` stay out of the head. Out-of-distribution rejection uses the energy gate $E(x) = -T \cdot \text{logsumexp}(z / T)$ and `minTop1Prob`.
2. **`uncoveredSpecies` thresholds** are minimum training-image counts: $\ge 55$ for `high_stakes` / toxic species and $\ge 40$ for others to enter the head. Species that fail those thresholds go to `uncoveredSpecies` in `model-manifest.json`, never to `classes` with `in_model: false`.
3. **Paths.** Use the repository `.venv` (`source .venv/bin/activate`). Keep code, `data/`, and `artifacts/` inside this repository on native Linux `ext4` (WSL2: never `/mnt/c`). Do not hard-code personal home directories.
4. **Hugging Face remotes** come from the environment (see `.env.example`): `HF_DATASET_REPO` and `HF_MODEL_REPO`. Do not embed private repository names or tokens in source, docs, or commits. Copy `.env.example` to `.env` locally; `.env` is gitignored.
5. **Licensing.** Training photos: only `CC0`, `CC BY`, `CC BY-NC`, `CC BY-SA` (versioned), and `pd_self`. Forbidden: `DF20` and `DF24` (images and weights matching `*ft_df20*`), AI-generated images, illustrations, and `ND` licenses. Backbones from `timm`: only `Apache-2.0`, `MIT`, or `BSD` by default (`ConvNeXt V2` and Apple-licensed `MobileViT`/`FastViT` stay disabled).
6. **Safety.** The model must not output edibility verdicts. Look-alike pairs live in `configs/karty_sobowtory.json`. Planned ship-gate targets (fail closed): `confidentToxicAsEdible == 0`, per-pair confusion $\le 0.05$, `highStakesRecallMin >= 0.85`, `tfliteAgreement >= 0.99`.

## 2. CLI and layout

All modules belong in `src/grzybobranie_ml/`. The only supported invocation is:

```bash
python -m grzybobranie_ml.<module>
```

Implemented modules: `license_policy`, `split`, `prepare_data`, `preprocess`, `recognition_math`, `train`, `calibrate_ood`, `multi_photo`, `export_tflite`, `quantization_drift`, `ship_gates`. Planned: `local_drop`, `cache_manager`, `manifest`. Environment bootstrap is `python3 scripts/init_env.py`.

## 3. Operating rules

- Never invent metrics. Every reported number must trace to an artifact on disk (JSON/JSONL, log, or command output from this session).
- Never commit secrets, tokens, credentials, or `split_salt`. The salt file stays outside git (default `~/.config/grzybobranie/split_salt`, overridable with `GRZYBOBRANIE_SPLIT_SALT_PATH`).
- Do not start training, download datasets, or call unimplemented modules unless the user explicitly asked for that step.
- Prefer extending `src/grzybobranie_ml/<module>.py` over adding flat standalone scripts (keep CLI entry points invocable via `python -m grzybobranie_ml.<module>`).
- **Training & EMA invariants:** Any `ModelEMA` wrapper must synchronize both `named_parameters()` and `named_buffers()` (BatchNorm running stats). On local RTX 5070 + ext4, configure `DataLoader` with `num_workers=8`, `pin_memory=True`, `persistent_workers=True`, and `prefetch_factor=2` when `num_workers > 0`.
- **Test split immutability:** Never use `data/splits/test.csv` or `ood_test` for temperature fitting, OOD threshold sweeps, representative INT8 calibration, or visual samples. Freeze all thresholds on `val.csv` and `ood_calib` before a single final `ship_gates` pass.
- When acting as the training bot, read `docs/soul.md` and execute stages P0–P9 from `docs/prompty.md`. The manifest contract is `schemas/model-manifest.schema.json` (see also `docs/decisions/2026-11-16-schema-v1-clarifications.md`).
