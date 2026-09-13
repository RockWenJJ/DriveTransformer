_base_ = ['./drivetransformer_base.py']

# New training: corrected physical inputs; no new architecture or loss balancing.
consistency_version = 'physical-v1'
model = dict(pts_bbox_head=dict(motion_cls_valid_only=True))
data = dict(
    train=dict(timestamp_frame_period_s=0.1, ego_accel_mode='xy'),
    val=dict(timestamp_frame_period_s=0.1, ego_accel_mode='xy'),
    test=dict(timestamp_frame_period_s=0.1, ego_accel_mode='xy'))
inference_contract = dict(timestamp_scale=1.0, ego_accel_mode='xy',
                          fixed_dist_decoder='sin_heading', legacy_eval_dropout=False)
