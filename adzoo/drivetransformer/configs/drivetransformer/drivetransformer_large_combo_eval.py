# Old Large checkpoint; only dropout-off + sin-heading online corrections.
_base_ = ['./drivetransformer_large.py']
consistency_version = 'large-combo-myubuntu22-20260915'
model = dict(pretrained=None, pts_bbox_head=dict(motion_cls_valid_only=False))
data = {split: dict(timestamp_frame_period_s=0.5, ego_accel_mode='legacy_z_broadcast') for split in ('train','val','test')}
inference_contract = dict(timestamp_scale=1.0, ego_accel_mode='xy', fixed_dist_decoder='sin_heading', legacy_eval_dropout=False)
