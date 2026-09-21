"""Single-factor ablation: exact same-class training-label deduplication."""
_base_ = ['./drivetransformer_res480_p1_map_control.py']
data = dict(train=dict(map_deduplicate=True))
work_dir = './adzoo/drivetransformer/work_dirs/mapping/dedup'
