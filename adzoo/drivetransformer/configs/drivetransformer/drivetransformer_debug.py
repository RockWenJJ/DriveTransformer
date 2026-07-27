# Debug config for single-step forward train.
# Inherit small model; keep batch_size=1 and workers=0 for IDE breakpoints.
_base_ = ['./drivetransformer_small.py']

num_gpus = 1
batch_size = 1
num_iters_per_epoch = 1
total_epochs = 1

work_dir = './work_dirs/drivetransformer_debug'

data = dict(
    samples_per_gpu=1,
    # Must be 0 so dataset / collate run in the main process (enables breakpoints).
    workers_per_gpu=0,
    train=dict(cache_lenth=2),
    val=dict(cache_lenth=2),
    test=dict(cache_lenth=2),
)

# A few iters are enough to step through one forward (+ backward) train step.
runner = dict(type='IterBasedRunner', max_iters=2)

# Disable AMP for clearer tensor values while debugging.
fp16 = None

log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
    ])

# Avoid writing checkpoints during a short debug run.
checkpoint_config = dict(interval=100000)

find_unused_parameters = True
