For the completed four-fix P1 Base training setup, use
[`P1_RETRAIN.md`](P1_RETRAIN.md) and `drivetransformer_base_p1fix_10ep.py`.
It retains the original motion loss. The broader physical-v1 experiment and
its September 9 validation status below are historical documentation.

DriveTransformer consistency changes — 2026-09-09

The physical-v1 profiles use 0.1-second dataset frame timestamps, x/y
acceleration, deterministic evaluation attention, and sin-heading decoding.
Motion classification uses valid matched agents and distributed valid-agent
normalization. These are behavior changes; loading an old checkpoint does not
make it a model trained with these conventions.

Use the explicit `drivetransformer_base_legacy_replay.py` profile for historical
R50 behavior. It intentionally preserves the old asymmetry: dataset timestamps
are 0.5 seconds and acceleration broadcasts z, while the online agent still
uses physical time and x/y acceleration. `tools/build_interface_matrix.py`
generates C0–C5 old-checkpoint diagnostics for either original Base or frozen
DINO-B configs. It does not start training or evaluation.

The corrected sin decoder restores direction only in the forward half-plane.
It still uses the historical radial approximation for the existing direction
PID. Do not use those points as exact arc-length geometry or as vehicle boxes
for collision checking. Target speed retains the original point [5] to [10]
window, corresponding to 0.6–1.1 seconds when labels start at 0.1 seconds.

Tests were executed in the remote DriveTransformer environment, with its
custom mmcv extensions and actual A100 xformers. Run commands from the full
repository root with that environment's Python and the root on PYTHONPATH.
The local delivery is a review mirror, not a complete runnable repository.

```bash
# CPU contracts, motion masks/gradients, optimizer parameter groups and profiles
CUDA_VISIBLE_DEVICES= BASELINE_REPO=/data5T/wenjj/Projects/DriveTransformer \
  python -m unittest discover -s tests -p test_consistency.py -v

# Requires a selected CUDA device; tests use small tensors
RUN_CUDA_TESTS=1 python -m unittest discover -s tests \
  -p test_consistency.py -v -k actual_xformers
```

`tools/probe_data_contracts.py` checks the complete dataset pipeline and samples
physical time from train/val metadata. `tools/smoke_consistency_model.py`
requires an explicit config, checkpoint and new output path, uses 192×528
images with the full Base architecture, caps allocator memory and makes no
optimizer update. It verifies real loss/backward, repeatability after memory
reset, and second-frame inference. It is not a training or accuracy benchmark.

The completed checks covered 20 CPU tests and 2 CUDA tests, including FP32 and
BF16 attention and checkpoint behavior. The full Base smoke used about 2686
MiB; the task-proxy-v2 entry point also ran on 3 real validation frames. Full
resolution distributed optimizer training and CARLA regression remain to be
run; the valid-count distributed normalization has a simulated sparse-rank
test, not a real multi-GPU integration test.

Use `tools/ol_eval_tasks_v2.py` for the new metric protocol. It defaults to full
clip streaming, checks checkpoint compatibility, records hashes/seed, and
refuses to overwrite results unless explicitly requested. Its AP, map raster
IoU and center-distance collision proxy are custom diagnostics. Do not merge
them numerically with previous evaluator versions or call them official
nuScenes/vehicle collision metrics. Mode top-1 and oriented-box continuous
collision metrics are not implemented here.

`tools/audit_closed_loop.py --require-complete` refuses to accept a partial route
set as a complete score. Keep route XML, source records, seed and evaluator
version with every result. The FT DINO-B result currently contains 88/94
routes; its available-route mean is provisional.

Consistency configs inherit training schedules unless explicitly overridden.
The separate prepared T0–T4 adaptation configs specify 4 workers, microbatch
20, 5000 updates, LR 2e-5 and 200 warmup updates: 400,000 additional samples
from the same frozen DINO-B checkpoint, with fresh optimizer/scheduler state.
Their validation and online contract is physical for every case. Launching
with a different batch or world size requires a revised matched budget.

The frozen dev24_v1 routes now live in `evaluation/routes/dev24_v1/`.
`tools/build_interface_matrix.py` binds generated C0-C5 configs to this subset;
`tools/evaluate_dev24.sh` launches the fixed checkout's agent with all 24
routes, or one six-route shard via `--shard`. See `docs/DEV24_EVALUATION.md`
for cloud paths and a dry-run example. Four routing tests and 60 combinations
of existing configs and shards passed, together with real agent/controller
imports and the native route parser. These checks did not run CARLA or load
model weights; full closed-loop validation remains pending.
