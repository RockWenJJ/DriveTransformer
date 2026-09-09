_base_ = ['./drivetransformer_base.py']

# Reproduce the old *asymmetric* train/open-loop vs online conventions.
# Do not use this as the corrected training baseline.
consistency_version = 'legacy-replay'
model = dict(pts_bbox_head=dict(motion_cls_valid_only=False))
data = dict(
    train=dict(timestamp_frame_period_s=0.5, ego_accel_mode='legacy_z_broadcast'),
    val=dict(timestamp_frame_period_s=0.5, ego_accel_mode='legacy_z_broadcast'),
    test=dict(timestamp_frame_period_s=0.5, ego_accel_mode='legacy_z_broadcast'))
inference_contract = dict(timestamp_scale=1.0, ego_accel_mode='xy',
                          fixed_dist_decoder='legacy_radians', legacy_eval_dropout=True)
