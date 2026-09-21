_base_ = ['./drivetransformer_base_res480x960.py']
num_gpus = 4
batch_size = 20
total_epochs = 10
num_iters_per_epoch = 2935
consistency_version = 'physical-v1-res480x960'
model = dict(pretrained=dict(img='./ckpts/resnet50-19c8e357.pth'),
             pts_bbox_head=dict(motion_cls_valid_only=False))
data = dict(samples_per_gpu=20, workers_per_gpu=6,
    train=dict(cache_lenth=21, timestamp_frame_period_s=0.1, ego_accel_mode='xy'),
    val=dict(cache_lenth=21, timestamp_frame_period_s=0.1, ego_accel_mode='xy'),
    test=dict(cache_lenth=21, timestamp_frame_period_s=0.1, ego_accel_mode='xy'))
inference_contract = dict(timestamp_scale=1., ego_accel_mode='xy',
                          fixed_dist_decoder='sin_heading', legacy_eval_dropout=False)
load_from = None
resume_from = None
optimizer = dict(lr=1e-4)
runner = dict(type='IterBasedRunner', max_iters=29350)
checkpoint_config = dict(interval=1000, save_last=True, max_keep_ckpts=5)
custom_hooks = []
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook'),
    dict(type='TensorboardLoggerHook'), dict(type='WandbLoggerHook',
    init_kwargs=dict(project='DriveTransformer', name='dt_base_p1fix_res480x960_10ep',
                     resume='never'))])
work_dir = './adzoo/drivetransformer/work_dirs/res480x960_p1_10ep'
