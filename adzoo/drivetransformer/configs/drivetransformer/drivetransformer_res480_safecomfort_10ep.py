_base_ = ['./drivetransformer_res480_no_map_10ep.py']
model = dict(pts_bbox_head=dict(type='SafeComfortHead',safety_cfg=dict(collision_weight=1.,comfort_weight=2.0254338745149288e-07,align_weight=.1,margin=.25)))
data = dict(train=dict(type='SafeComfortDataset'))
custom_hooks = [dict(type='SafeComfortRampHook',start=1468,end=4403)]
work_dir = 'work_dirs/res480_safecomfort_10ep'
log_config = dict(interval=50,hooks=[dict(type='TextLoggerHook'),dict(type='TensorboardLoggerHook'),dict(type='WandbLoggerHook',init_kwargs=dict(project='DriveTransformer',name='DT-Res480-NoMap-SafeComfort',resume='never'))])
