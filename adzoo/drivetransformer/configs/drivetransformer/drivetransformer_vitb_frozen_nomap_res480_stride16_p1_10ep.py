"""Retain ViT patch resolution: 480x960 -> 30x60, no adapter downsampling.

The stride-1 3x3 adapter preserves the same trainable layers and parameter count
as stride-32. Frozen pretrained ViT-B; adapter/DriveTransformer train from scratch.
"""
_base_ = ['./drivetransformer_vitb_frozen_nomap_res480_p1_10ep.py']
model = dict(img_backbone=dict(output_stride=16))
work_dir = './adzoo/drivetransformer/work_dirs/vitb_frozen_nomap_res480_stride16_p1_10ep'
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook'),
    dict(type='WandbLoggerHook', init_kwargs=dict(project='DriveTransformer',
    name='dt_base_dinov3_vitb16_frozen_p1fix_res480x960_no_map_stride16_10ep',
    id='vitb-frozen-nomap-res480-stride16-p1-10ep-20260920', resume='never'))])
