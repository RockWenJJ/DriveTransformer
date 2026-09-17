"""Combined label fixes; evaluate using the unchanged historical GT protocol."""
_base_ = ['./drivetransformer_res480_p1_map_control.py']
data = dict(train=dict(map_clip_mode='geometric', map_deduplicate=True))
work_dir = './adzoo/drivetransformer/work_dirs/mapping/datafix'
