# P2 v1 reference math — explicit assumptions

Recorded 16 November 2026 (Europe/Warsaw) with the reference implementation in `src/grzybobranie_ml/recognition_math.py` and the synthetic fixtures in `fixtures/p2/`. These points fill gaps where the approved contract (`docs/decisions/2026-11-16-schema-v1-clarifications.md`, `schemas/model-manifest.schema.json` v1.0.1, `docs/prompty.md` P5) is silent or only implied. They should be confirmed by Wojciech and the React Native architect before the app treats the fixtures as frozen.

## Entropy-weighted fusion

Taken verbatim from P5 and the schema description of `fusionWeights.method = entropy_weighted_logits`:

- `p_v = softmax(z_v / T_prob)` with `T_prob = calibration.temperature` (or the single available T).
- `H_v = -Σ_k p_v[k] ln p_v[k]` (natural log). `0 ln 0` is treated as 0.
- `c_v = 1 - H_v / ln K`, where `K` is the length of the logit vector (head size).
- `w_v = c_v / Σ_u c_u`. If `Σ c = 0` (every accepted photo is uniform), weights are `1/m`, as the schema says ("równe wagi gdy Σc = 0").
- `z̄ = Σ_v w_v z_v` over **accepted** photos only.

`m` is the number of photos whose per-photo status is `ok`. Input is at most 3 photos (`fusionWeights.maxPhotos`). More than 3 is an error, not a silent truncate.

## `m = 1` and `m = 0`

The schema defines `ood.fusion` keys `"2"` and `"3"` only. Interpretation used here:

- `m = 0`: no `z̄`; status is `out_of_distribution`; `energy` and `p_top1` are `null`; candidates are empty.
- `m = 1`: `z̄` is the single accepted logit vector (weight 1). It is scored with the **single-photo** thresholds `ood.energyThreshold` and `ood.minTop1Prob`, unless a caller explicitly provides `ood.fusion["1"]`. This is not a schema key; it is a convenience, not a required fixture field.
- `m ∈ {2, 3}`: both gates are re-evaluated on `z̄` with `ood.fusion[m]`. If that key is missing, the reference raises `ValueError` (fusion unavailable), rather than inventing a fourth status.

## Candidate list

The contract requires a `candidates` list on an `ok` decision but does not specify top-k versus a cumulative-probability cutoff. This reference returns **top-3** by calibrated softmax `p = softmax(z / T_prob)`, ties broken by **lower class index**. If `K < 3`, all classes are returned. Rank starts at 1. Each entry is `{index, classId, probability, rank}`.

Rationale: the schema's `dangerousGenusMinProbability` text talks about "pozycja 2–3", and the outdated prototype (`ml/recognition_math.py`) also used a top-3 list. There is no documented cumulative-mass rule (e.g. "include classes until Σp ≥ 0.95").

On `out_of_distribution` and `unclear`, `candidates` is **empty** (the scanner does not return a species). The P5 phrase "candidates + dangerous_genus / low_confidence" is **not** implemented in this v1 fixture: warning flags would need genera and `lowConfidenceBelow` on every case, and they are not part of the agreed expected object `{status, energy, p_top1, candidates}`. That can be added later without changing the gate math.

## Boundary inclusivity

Both gates use **strict** inequalities, following the schema and P5 wording:

- `E > energyThreshold` → `out_of_distribution`. Equality is **not** OOD.
- `p_top1 < minTop1Prob` → `unclear`. Equality is **not** unclear (status is `ok` if the energy gate also passed).

Fixtures `boundary_energy_equal` and `boundary_top1_equal` lock this in.

## Other implementation choices

- All internals are float64. Public numbers are Python `float`.
- Fixture documents use `schemaVersion: "1.0.1"` (the manifest contract they implement), `generator: "grzybobranie_ml.make_p2_fixtures"`, `tolerance: 1e-5`. Every case is tagged `"synthetic": true`.
- A missing `ood.energyThreshold` or `ood.minTop1Prob` raises `ValueError` rather than returning a status `unavailable` (that status is not in the P2 trio).
- Fusion `expected` objects also include `m` (the accepted-photo count), which the decision note's fixture sketch did not list but which is required to interpret the case.
