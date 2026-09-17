_base_ = ['./drivetransformer_res480_no_map_10ep.py']
model = dict(pretrained=None)
data = dict(workers_per_gpu=2)
custom_hooks = []
