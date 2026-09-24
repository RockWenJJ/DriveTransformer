"""Random60 +10 original-data epochs: preserve weights and AdamW moments."""
_base_ = ['./drivetransformer_res480_nomap_random60_10ep.py']
runner = dict(type='Random60ContinuationRunner', max_iters=58700)
data = dict(shuffler_sampler=dict(type='ContinuationRandomShortRouteBatchSampler', skip_batches=29350))
lr_config = dict(_delete_=True, policy='Random60ContinueConstant', by_epoch=False,
                 start_iter=29350, additional_iters=29350, warmup=None)
work_dir = 'work_dirs/res480_random60_continue_20ep'
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook'),
    dict(type='WandbLoggerHook', init_kwargs=dict(project='DriveTransformer',
    name='DT-Res480-NoMap-Random60-20ep', resume='never'))])
