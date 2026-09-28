# DT-Qwen08-LearnToken

Controlled extension of DT-Qwen08-FrozenSem. `LearnTokenQwenBackbone.q_lang`
is one FP32 `[1,1,1024]` parameter shared by every frame. It is initialized with
normal std 0.02 using a private seed 0, so all existing DT/adaptor tensors retain
the same initialization. Native Qwen vocabulary/weights are unchanged.

Six cameras share the frozen native ViT and Patch Merger. CAM_FRONT merged
features and the unchanged English prompt form the native non-thinking input;
`q_lang` is appended after its assistant prefix at the next native text MRoPE
position. The last-layer normalized token output is projected to 512 dimensions,
concatenated with the final `ego + ego_pos` feature, fused, and fed to both planning
heads. Detection, motion, earlier planning layers and memory use the original
NoMap interfaces.

The legacy DT runtime owns the token, optimizer state, checkpoint and DDP
reduction. `token_bridge.py` connects it to a frozen modern Qwen worker with a
first-order custom autograd operation. Each forward saves its token and detached
front features under a request ID. During backward the worker recomputes each
language chunk with autograd enabled, applies the received hidden-state gradient,
and returns the summed token VJP. This bounds memory independently of the DT
batch size. Native Qwen parameters remain `requires_grad=False` and in eval mode;
the six-camera visual forward is always no-grad. Inference saves no backward state.
Higher-order gradients and repeated backward on one request are intentionally
unsupported; mismatched, duplicate and unconsumed requests fail explicitly.

Validation scripts cover native positions/forward/VJP parity, all native weight
hashes before/after backward, out-of-order requests, the cross-runtime autograd
edge, real planning gradients and optimizer updates, unchanged branch outputs,
strict checkpoint reload and identical baseline initialization. The first three
actual updates additionally audit nonzero token changes and DDP rank equality.
BF16 repeated backward can have small floating point variation; reference VJP
comparison records relative L2 with a 1% bound. Recomputed forward must be exact.

Train with `drivetransformer_qwen08_learntoken_10ep.py`: original NoMap data,
AdamW/losses/P1, resolution 480x960, 4x20, 29350 updates, seed 0, from scratch
except the frozen pretrained Qwen snapshot. Do not load FrozenSem's trained DT.
