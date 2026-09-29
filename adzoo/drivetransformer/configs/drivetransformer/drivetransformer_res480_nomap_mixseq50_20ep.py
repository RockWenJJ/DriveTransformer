"""DT-Res480-NoMap: 50% random40..80 / 50% full post-prefix25, scratch20ep."""
_base_ = ['./drivetransformer_res480_no_map_10ep.py']
data = dict(
    train=dict(type='MixedRouteDataset', drop_prefix=25, sub_seq_lenth=-1),
    shuffler_sampler=dict(type='MixedRouteBatchSampler', random_probability=0.5,
                         drop_prefix=25, min_length=40, max_length=80),
)
runner = dict(type='IterBasedRunner', max_iters=58700)
load_from = None
resume_from = None
work_dir = 'work_dirs/res480_nomap_mixseq50_20ep'
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook'),
    dict(type='WandbLoggerHook', init_kwargs=dict(project='DriveTransformer',
         name='DT-Res480-NoMap-MixSeq50-20ep', resume='never'))])
