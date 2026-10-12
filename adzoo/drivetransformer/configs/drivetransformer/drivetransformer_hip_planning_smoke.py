"""Same 20 x 4 native training path; only 20 iterations and isolated outputs."""
_base_ = ['./drivetransformer_hip_planning_10ep.py']
runner = dict(type='IterBasedRunner', max_iters=20)
checkpoint_config = dict(interval=20, save_last=True, max_keep_ckpts=1)
log_config = dict(interval=1)
work_dir = './work_dirs/dt_hip_planning_smoke'
