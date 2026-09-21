"""DT-Large / frozen DINO-B layers 4,8,12 / 480x960 / no map / S16.

Fresh adapter and DT initialization; only the DINO trunk loads pretraining.
4 GPUs x 10 samples = 40 samples/update; 58,700 updates = 2,348,000
sample exposures, matching the recent 4x20, 29,350-update training budget.
"""
_base_ = ['./drivetransformer_large.py']

num_gpus = 4
batch_size = 10
total_epochs = 10
num_iters_per_epoch = 5870
queue_length = 1
_res_aug = dict(resize_lim=(0.60, 0.70), final_dim=(480, 960),
    bot_pct_lim=(0.0, 0.0), rot_lim=(-5.4, 5.4), H=900, W=1600,
    rand_flip=True, rot3d_range=[0, 0])
data_aug_conf = _res_aug

model = dict(
    type='NoMapDriveTransformer',
    pretrained=None,
    img_backbone=dict(_delete_=True, type='FrozenDINOv3ViT',
        variant='vitb16', checkpoint='./ckpts/dinov3_vitb16_lvd1689m.safetensors',
        freeze=True, out_indices=(3, 7, 11),  # zero-based = layers 4, 8, 12
        out_channels=768, output_stride=16, forward_chunk_size=12,
        amp_dtype='bfloat16'),
    img_neck=None,
    pts_bbox_head=dict(type='NoMapHead', img_stride=16,
        ego_lcf_feat_idx=[0, 1, 2, 3, 4, 5, 6, 7, 8],
        motion_cls_valid_only=False))

data = dict(samples_per_gpu=batch_size, workers_per_gpu=4,
    train=dict(data_aug_conf=_res_aug, cache_lenth=batch_size+1,
        timestamp_frame_period_s=0.1, ego_accel_mode='xy'),
    val=dict(data_aug_conf=_res_aug, cache_lenth=batch_size+1,
        timestamp_frame_period_s=0.1, ego_accel_mode='xy'),
    test=dict(data_aug_conf=_res_aug, cache_lenth=batch_size+1,
        timestamp_frame_period_s=0.1, ego_accel_mode='xy'))
inference_contract = dict(timestamp_scale=1.0, ego_accel_mode='xy',
    fixed_dist_decoder='sin_heading', legacy_eval_dropout=False)
consistency_version = 'physical-v1-large-vitb3-nomap-s16'
load_from = None
resume_from = None
optimizer = dict(type='AdamW', lr=3e-5, weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(policy='CosineAnnealing', warmup='linear',
    warmup_iters=2000, warmup_ratio=0.1, min_lr_ratio=1e-2)
fp16 = dict(loss_scale=512.)
runner = dict(type='IterBasedRunner', max_iters=58700)
checkpoint_config = dict(interval=2000, save_last=True, max_keep_ckpts=5)
custom_hooks = []
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook'),
    dict(type='TensorboardLoggerHook'),
    dict(type='WandbLoggerHook', init_kwargs=dict(project='DriveTransformer',
        name='DTL-ViTB-Fuse3-NoMap-S16',
        id='large-vitb3-nomap-s16-res480-20260921', resume='allow'))])
work_dir = './work_dirs/large_vitb3_nomap_s16_res480_10ep'
training_contract = dict(gpus=4, samples_per_gpu=10, global_batch=40,
    optimizer_updates=58700, sample_exposures=2348000,
    backbone_layers_1based=[4, 8, 12], embed_dims=768, decoder_layers=12,
    output_stride=16, ego_accel_mode='xy', map_branch=False)
