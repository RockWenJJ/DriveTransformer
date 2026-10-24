_base_ = ["./drivetransformer_base_dim768.py"]

# with_cp (reentrant gradient checkpointing) double-marks DDP gradient
# buckets ("Expected to mark a variable ready only once") and crashes DDP
# training on torch 2.1. Disable it here; fall back on micro-batch if OOM.
model = dict(
    pts_bbox_head=dict(
        agent_prep_decoder=dict(transformerlayers=dict(with_cp=False)),
        map_prep_decoder=dict(transformerlayers=dict(with_cp=False)),
        transformer=dict(
            decoder=dict(transformerlayers=dict(with_cp=False)))),
)
