"""Authorized res480 map dedup + geometric clipping, P1 fixed, 20x4, 10 epochs."""
_base_ = ['./drivetransformer_res480_p1_map_datafix.py']
work_dir = './adzoo/drivetransformer/work_dirs/mapping/datafix_10ep'
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook'),
    dict(type='WandbLoggerHook', init_kwargs=dict(project='DriveTransformer',
    name='dt_base_p1fix_res480x960_map_datafix_10ep', resume='never'))])
