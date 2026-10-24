_base_ = ['./drivetransformer_base.py']

# 75% input resolution: 1056x384 -> 792x288 per camera.
# resize_lim scaled so the resized image always covers final_dim
# (1600*0.50=800 >= 792, 900*0.50=450 >= 288).
_lowres_aug_conf = dict(
    resize_lim=(0.50, 0.54),
    final_dim=(288, 792),
    bot_pct_lim=(0.0, 0.0),
    rot_lim=(-5.4, 5.4),
    H=900,
    W=1600,
    rand_flip=True,
    rot3d_range=[0, 0],
)

data = dict(
    train=dict(data_aug_conf=_lowres_aug_conf),
    val=dict(data_aug_conf=_lowres_aug_conf),
    test=dict(data_aug_conf=_lowres_aug_conf),
)
