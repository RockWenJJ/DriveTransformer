_base_ = ['./drivetransformer_base.py']

# Aspect-ratio experiment vs base 1056x384:
# test-time: isotropic resize 0.6 -> 960x540, then drop top 60px -> 960x480.
# resize_lim lower bound is 0.60 so the resized image always covers final_dim
# (1600*0.60=960 >= 960, 900*0.60=540 >= 480).
_res_aug_conf = dict(
    resize_lim=(0.60, 0.70),
    final_dim=(480, 960),
    bot_pct_lim=(0.0, 0.0),
    rot_lim=(-5.4, 5.4),
    H=900,
    W=1600,
    rand_flip=True,
    rot3d_range=[0, 0],
)

data = dict(
    samples_per_gpu=20,
    train=dict(data_aug_conf=_res_aug_conf),
    val=dict(data_aug_conf=_res_aug_conf),
    test=dict(data_aug_conf=_res_aug_conf),
)
runner = dict(type='IterBasedRunner', max_iters=25000)
