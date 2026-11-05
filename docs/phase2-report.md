# Phase 2: baseline and SLARDA evaluation

## Status and setup

Phase 2 completed on CPU against dataset v1. The evaluated holdout is Hai Phong
2025 (8,736 windows; 30.87% rain-positive). The processed window file SHA256 is
`f9e00c906fe9b1215ae8e99e7e5f18ee6680a0e154a2a7f06f1c9fcc74280d0a`.

The experiment source was based on commit
`98f0fad3ab63e209b302c2fcc7b9535896cb42a7`. Since Phase 2 changes remain
uncommitted, the exact experiment source/config fingerprint is recorded as
`f84034a860d4c001e1f7e20e99e55e7a455076c96fb6ce5cd94a3f9d6483aaf7` in
`artifacts/metrics/experiment_config.json`. The config used seeds 17, 23, 42,
batch size 512, and at most eight epochs.

## Protocol

- B0 predicts rain at t+1 when precipitation at the final observed input hour
  is at least 0.1 mm.
- B1 is balanced logistic regression trained on flattened, scaled source
  windows.
- B2 is a class-weighted CNN trained only with labeled Hanoi source-train
  windows.
- B3 starts from the corresponding B2 checkpoint, freezes that source model,
  and adapts a copy using a GRU autoregressive domain discriminator and EMA
  teacher pseudo-labels above 0.9 confidence. Hai Phong adaptation labels remain
  hidden.
- The StandardScaler is fitted only on Hanoi source-train window observations.
  Epoch and probability-threshold selection use Hanoi source validation. The
  Hai Phong 2025 partition is loaded for evaluation only after all three-seed
  checkpoints and thresholds have been selected.

## Target final-test results

| Model | Macro-F1 | Rain F1 | Rain recall | PR-AUC | ROC-AUC | Balanced accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 Persistence | 0.8801 | 0.8343 | 0.8343 | 0.7472 | 0.8801 | 0.8801 |
| B1 Logistic regression | 0.8572 | 0.8011 | 0.7887 | 0.8805 | 0.9371 | 0.8541 |
| B2 CNN source-only | 0.7839 ± 0.0095 | 0.7108 ± 0.0168 | 0.7617 ± 0.0927 | 0.8004 ± 0.0083 | 0.8934 ± 0.0071 | 0.7962 ± 0.0185 |
| B3 CNN + SLARDA | 0.7889 ± 0.0050 | 0.7148 ± 0.0120 | 0.7519 ± 0.0416 | 0.8005 ± 0.0091 | 0.8914 ± 0.0066 | 0.7976 ± 0.0119 |

B2 and B3 values are mean ± sample standard deviation over seeds 17, 23, and
42. B0 and B1 are deterministic and are shown once. All models were evaluated
at thresholds fixed on source validation (B0 uses its fixed 0.1 mm persistence
rule). PR-AUC is reported as average precision.

## Gate results and interpretation

- B1 source-validation Macro-F1 is 0.8420, above the source-train majority
  predictor's 0.3243; the signal and label pipeline pass this gate.
- B2 training loss fell in every seed: 0.5159→0.2632, 0.5272→0.2602, and
  0.5320→0.2687. Losses and emitted metrics were finite.
- B3 completed adaptation and generated target metrics for all three seeds.
  The average number of confident target pseudo-labels per batch was recorded
  in each run JSON.
- B3 gains 0.0050 Macro-F1 and 0.0014 balanced accuracy over B2 on average,
  while its mean rain recall is 0.0098 lower. Its PR-AUC is effectively
  unchanged. This is a small, mixed result, not evidence that adaptation
  reliably improves rain detection.
- Persistence is strongest by Macro-F1 and rain recall. Logistic regression
  has the highest PR-AUC and ROC-AUC. The neural baselines do not outperform
  these simpler references in this experiment.

The confusion-matrix SVG selects, for each CNN method, the seed with the
highest source-validation Macro-F1. It is a representative run, not a sum over
seeds. Per-seed target confusion matrices and metrics are stored in the run
JSON files.

## Artifacts and verification

- `artifacts/metrics/baseline_summary.csv`: validation and target metrics,
  including per-seed scores and mean/std rows.
- `artifacts/metrics/confusion_matrices.svg`: B0-B3 target confusion matrices.
- `artifacts/metrics/runs/`: B0, B1, and per-seed B2/B3 configs, dataset hash,
  base Git SHA, source fingerprint, training history, selected threshold,
  metrics, and prediction hashes.
- `artifacts/checkpoints/`: CNN checkpoints and logistic-regression model;
  the source-train scaler is in `artifacts/metrics/source_train_scaler.npz`.
- Ruff and Python bytecode compilation passed. Tests were not run.

The DINOv3 comparison belongs to Phase 3. No target-test labels were used to
select thresholds, epochs, or hyperparameters in this phase.
