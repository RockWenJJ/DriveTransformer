"""Single-factor ablation: geometry-preserving training-label ROI clipping."""
_base_ = ['./drivetransformer_res480_p1_map_control.py']
data = dict(train=dict(map_clip_mode='geometric'))
work_dir = './adzoo/drivetransformer/work_dirs/mapping/clip'
