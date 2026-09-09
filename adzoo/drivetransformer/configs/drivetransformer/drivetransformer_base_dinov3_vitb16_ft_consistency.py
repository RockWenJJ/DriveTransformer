_base_ = ['./drivetransformer_base_dinov3_vitb16_consistency.py']
model = dict(pretrained=None, img_backbone=dict(freeze=False))
# Pretrained trunk adapts slowly; the trainable adapter retains the head LR.
optimizer = dict(paramwise_cfg=dict(custom_keys={
    'img_backbone.dinov3': dict(lr_mult=0.1, decay_mult=1.0),
    'img_backbone.adapter': dict(lr_mult=1.0, decay_mult=1.0),
}))
