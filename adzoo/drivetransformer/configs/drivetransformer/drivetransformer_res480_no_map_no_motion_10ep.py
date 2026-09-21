"""P1-fixed ResNet50 DT-Base at 480x960: detection and ego planning."""
_base_ = ['./drivetransformer_res480_no_map_10ep.py']
model = dict(type='NoMapNoMotionDriveTransformer',
    pts_bbox_head=dict(type='NoMapNoMotionHead', bbox_coder=dict(type='NMSFreeCoder')))
work_dir = 'work_dirs/res480_no_map_no_motion_10ep'
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook'),
    dict(type='WandbLoggerHook', init_kwargs=dict(project='DriveTransformer',
    name='DT-Res480-NoMap-NoMotion', resume='never'))])
