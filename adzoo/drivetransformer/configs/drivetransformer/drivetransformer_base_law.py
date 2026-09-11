"""LAW auxiliary training for dev_wenjj Base. Existing Base remains unchanged.

The 1.5 s target is 15 saved Bench2Drive frames at 10 Hz. Pairing checks
raw frame_idx, not the legacy dataset timestamp=frame_idx/2 convention.
"""
_base_ = ['./drivetransformer_base.py']

model = dict(law_cfg=dict(
    num_cams=6, pool_size=(2, 4), num_heads=8, num_layers=2,
    dropout=0.0, use_action=True, loss_weight=0.2))

law_future_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='ResizeCropFlipImage'),
    dict(type='NormalizeMultiviewImage',
         mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
    dict(type='DefaultFormatBundle3D', class_names=[], with_gt=False, with_label=False),
    dict(type='CustomCollect3D', keys=['img']),
]
data = dict(train=dict(law_future=dict(
    frame_offset=15, frame_idx_step=1, frame_period_seconds=0.1,
    pipeline=law_future_pipeline)))

custom_hooks = [
    dict(type='CustomSetEpochInfoHook'),
    dict(type='LAWLossWeightHook', max_weight=0.2, start_iter=0, warmup_iters=1000),
]
