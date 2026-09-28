_base_ = ['./drivetransformer_qwen08_frozensem_10ep.py']

# Reuse the immutable native snapshot/runtime and the NoMap experiment settings.
model = dict(type='QwenLearnTokenDriveTransformer',
    img_backbone=dict(type='LearnTokenQwenBackbone', token_seed=0),
    pts_bbox_head=dict(type='QwenLearnTokenHead'))
qwen_contract = dict(name='DT-Qwen08-LearnToken', learned_language_token=True,
    readout='appended_soft_input_token', token_shape=[1, 1, 1024],
    token_init='normal_std0.02_private_seed0',
    gradient='planning_loss_to_token_through_frozen_Qwen_recomputed_VJP')
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook'),
    dict(type='TensorboardLoggerHook'), dict(type='WandbLoggerHook',
    init_kwargs=dict(project='DriveTransformer', name='DT-Qwen08-LearnToken',
                     id='qwen08-learntoken-20260928', resume='never'))])
work_dir = './adzoo/drivetransformer/work_dirs/qwen08_learntoken_10ep'
# Runs after the optimizer and only observes the first three updates.
custom_hooks = [dict(type='QwenTokenAuditHook', priority='LOW')]
