"""P1-fixed 480x960, frozen pretrained DINOv3 ViT-B/16, no map queries."""
_base_ = ['./drivetransformer_res480_no_map_10ep.py']
model = dict(
    pretrained=None,
    img_backbone=dict(_delete_=True, type='FrozenDINOv3ViT', variant='vitb16',
        checkpoint='./ckpts/dinov3_vitb16_lvd1689m.safetensors',
        out_channels=512, forward_chunk_size=12, amp_dtype='bfloat16', freeze=True),
    img_neck=None,
)
work_dir = './adzoo/drivetransformer/work_dirs/vitb_frozen_nomap_res480_p1_10ep'
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook'),
    dict(type='WandbLoggerHook', init_kwargs=dict(project='DriveTransformer',
    name='dt_base_dinov3_vitb16_frozen_p1fix_res480x960_no_map_10ep',
    id='vitb-frozen-nomap-res480-p1-10ep-20260918', resume='never'))])
