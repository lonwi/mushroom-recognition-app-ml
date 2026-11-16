# Schema v1 clarifications (approved 2026-11-16)

Approved by Wojciech Bubółka on 16 November 2026 (Europe/Warsaw). These notes amend `schemas/model-manifest.schema.json` to v1.0.1 (description-only; major version stays 1; no new manifest fields). They do not change measured metrics — none were computed in this step.

## a) `minTop1Prob` uses calibrated softmax

`minTop1Prob` is evaluated on $\mathrm{softmax}(z / T)$, where $T$ is `calibration.temperature`. The same $T$ is used for fusion weights (entropy-weighted logits).

## b) Two temperature fields, one fallback rule

- All probabilities use `calibration.temperature`.
- Energy $E(x) = -T \cdot \mathrm{logsumexp}(z / T)$ uses `ood.energyTemperature`.
- If `ood.energyTemperature` is absent, energy uses `calibration.temperature`.
- If only one of the two fields is present, that single $T$ serves both probabilities and energy.

## c) Fusion after per-photo gates

1. Apply both gates (energy and `minTop1Prob`) to each photo first (`fusionWeights.gateBeforeFusion` is `true`).
2. $m$ is the number of **accepted** photos, not the number of input photos.
3. Average / fuse the accepted logits to $\bar{z}$.
4. Recompute **both** gates on $\bar{z}$, with thresholds from `ood.fusion[m]` and the same $T$ rules as above.
5. If `ood.fusion[m]` is missing, fusion for that $m$ is unavailable.

## d) `uncoveredSpecies` and look-alikes

A species listed in `uncoveredSpecies` is never returned by the scanner (Atlas may still show it). If that species has a look-alike in the model head, the app ResultSheet warns via `uncovered_look_alikes`. Uncovered species must not appear in `classes` with `in_model: false`.

Minimum training-image counts to keep a class in the head remain: $\ge 55$ for `high_stakes` / toxic species, $\ge 40$ for others. These are data-count thresholds, not energy/OOD thresholds.

## e) Rejection statuses, input size, baseline preprocessing

- Rejection statuses are `out_of_distribution` (energy) and `unclear` (`minTop1Prob`). There is no `not_a_mushroom` class.
- Input size $N$ is not fixed by the schema; it is 224 or 256 as declared in `preprocessing.inputSize`.
- Baseline preprocessing: resize filter `box` with `antialias`, crop `none_stretch`.

## P2 decision fixtures (agreed format)

Share these fixtures with the mobile app; do not fork a second format.

- `decision_cases.json` and `fusion_cases.json` contain: input logits, $T$, thresholds, expected status (`out_of_distribution` / `unclear` / `ok`), and candidates. Numeric tolerance: $1 \times 10^{-5}$.
- P1 preprocessing parity fixtures use the app format `ml/fixtures/p1/parity.json`.

Related files: `docs/soul.md`, `docs/prompty.md`, `docs/decisions/decyzje.md`, `docs/decisions/krok3-decyzje.md`.
