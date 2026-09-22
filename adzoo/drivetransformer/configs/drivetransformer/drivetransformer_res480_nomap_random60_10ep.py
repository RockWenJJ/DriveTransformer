"""DT-Res480-NoMap baseline; random 40-80-frame training windows after prefix25."""
_base_ = ['./drivetransformer_res480_no_map_10ep.py']
data = dict(
    train=dict(type='RandomShortRouteDataset', drop_prefix=25, sub_seq_lenth=-1),
    shuffler_sampler=dict(type='RandomShortRouteBatchSampler', drop_prefix=25,
                         min_length=40, max_length=80),
)
# Keep matched exposure budget (not ten guaranteed passes over eligible frames).
runner = dict(type='IterBasedRunner', max_iters=29350)
work_dir = 'work_dirs/res480_nomap_random60_10ep'
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook'),
    dict(type='WandbLoggerHook', init_kwargs=dict(project='DriveTransformer',
    name='DT-Res480-NoMap-Random60', resume='never'))])
