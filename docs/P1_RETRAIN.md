# P1: released training and inference fixes

Use `adzoo/drivetransformer/configs/drivetransformer/drivetransformer_base_p1fix_10ep.py`
for the four-fix Base training setup used in the completed September 2026 run.
Run from the repository root, with the original ResNet-50 weights at
`ckpts/resnet50-19c8e357.pth` and the existing Bench2Drive data paths available.

| Interface | P1 setting |
| --- | --- |
| Dataset timestamps | Frame interval 0.1 seconds |
| Dataset acceleration | Horizontal x/y; do not broadcast z |
| Evaluation attention | Dropout disabled; training dropout retained |
| Fixed-distance heading | Decode predicted sine with sin-heading |

The online timestamp scale remains 1.0 and online acceleration remains x/y.
The original PID gains and braking policy are retained. The decoder fixes
heading interpretation; it does not add a new controller or guarantee a
closed-loop improvement. Old-weight inference correction and P1 retraining
must be reported separately.

The model default retains the original motion loss.
The P1 entry explicitly sets `motion_cls_valid_only=False` to preserve the
original motion classification loss. The older `*_consistency.py` profiles
also enable valid-agent loss normalization; they are a broader experiment
and should not be substituted for the four-fix P1 training entry.

The frozen budget is 29,350 updates at microbatch 20 on four GPUs (global 80),
approximately ten original-data epochs / 2,348,000 sample presentations.
Changing world size or microbatch changes this exposure budget. ResNet-50
uses image pretraining; DriveTransformer and optimizer start from scratch
(`load_from=None`, `resume_from=None`). Local Text/TensorBoard hooks match
the original run; this entry does not configure W&B upload.

This portable entry changes only config inheritance, pretrained-weight and
output directory paths relative to the archived training configuration.
Check resolved-config equivalence before claiming reproduction on new data
or a different environment. Training is not started by importing a config.

Use `drivetransformer_base_legacy_replay.py` or the explicit interface matrix
for historical replay. Keep input/config/checkpoint hashes, route sets,
seeds, technical retries and simulator environment with evaluation results.
LAW, resolution/crop experiments and the v4 controller remain separate.

CPU validation (from repository root with the project environment):

```sh
CUDA_VISIBLE_DEVICES= python -m unittest discover -s tests -v
```

CUDA attention checks require `RUN_CUDA_TESTS=1` and an available GPU;
otherwise they are explicitly skipped. Do not run them on occupied GPUs.
