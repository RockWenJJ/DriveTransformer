# Validated experiment integration — 2026-09-21

Integrates the completed res480/P1, no-map, map-label correction and frozen
ViT-B S32/S16 implementations into `dev_wenjj`, with an opt-in native CARLA
crash workaround. This integration does not establish that every variant
improves driving quality.

Sources merged with history:
- `6ac3bf088fb43c840ac18ca085eca85293a8d6be`: res480 10ep, agent/ego-only queries,
  no-map offline evaluation, frozen ViT-B, optional output_stride16 with matching
  head img_stride16. Default backbone output remains stride32.
- `9ed4329`: map-label deduplication/geometric ROI clipping, empty-map background
  supervision, target-point-count correctness, diagnostics and regression tests.
  Training label fixes are explicit in the map-datafix recipe; validation labels
  remain legacy. Decoder diagnostic options retain their original defaults.
- Research `efac44a`: CARLA 0.9.15 dormant-sensor null shared-state guard; packaged
  under `tools/carla_sensor_guard`, disabled unless explicitly selected. This is a
  temporary process-local workaround pinned to one executable SHA256, not a
  global simulator replacement. See its README for evidence and limitations.

New experiment recipes keep 29350 iterations, 20 samples/GPU, four GPUs, 10
original-data epoch equivalents, original P1 controller, four P1 contracts and
original motion-loss normalization. Fixed historical W&B run IDs are removed
and the historical absolute res480 output directory is made relative. Existing
running worktrees, checkpoints and results are not edited.

Integration validation:
- CPU regression suite, including P1/controller, mapping, stride geometry,
  route selection and fail-closed native launcher checks; opt-in CUDA kernel
  checks reuse prior same-source/environment evidence.
- Five fully resolved recipes match their trained sources except work_dir and
  logging identity metadata.
- Final mapfix and S16 checkpoints strictly load into the merged models with
  finite tensors (147639916 and 176128860 parameters respectively).
- Core map/no-map/ViT implementations match the previously verified bytes.
- The original P1 controller, agent and interface-contract files are unchanged.

LAW, v4 controller, cancelled planning-only variants and unrelated in-progress
experiments are outside this integration. This commit does not launch training
or evaluation. Detailed run identities, command logs and push receipts live in
RockWenJJ/A100_DriveTransformer_Research under
`research/reports/validated_merge_20260921/`.
