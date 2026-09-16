_base_ = ['./drivetransformer_base_res480x960_p1_10ep.py']
model = dict(type='PlanningOnlyDriveTransformer', pts_bbox_head=dict(type='PlanningOnlyHead'))
work_dir = 'work_dirs/res480_planning_only_10ep'
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook'),
    dict(type='WandbLoggerHook', init_kwargs=dict(project='DriveTransformer',
    name='dt_base_p1fix_res480x960_planning_only_10ep', id='p1-res480x960-planning-only-10ep-20260916', resume='never'))])
