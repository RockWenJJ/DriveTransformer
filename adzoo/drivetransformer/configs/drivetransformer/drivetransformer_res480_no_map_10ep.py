_base_ = ['./drivetransformer_base_res480x960_p1_10ep.py']
model = dict(type='NoMapDriveTransformer', pts_bbox_head=dict(type='NoMapHead'))
work_dir = 'work_dirs/res480_no_map_10ep'
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook'),
    dict(type='WandbLoggerHook', init_kwargs=dict(project='DriveTransformer',
    name='dt_base_p1fix_res480x960_no_map_10ep', id='p1-res480x960-no-map-10ep-20260916', resume='never'))])
