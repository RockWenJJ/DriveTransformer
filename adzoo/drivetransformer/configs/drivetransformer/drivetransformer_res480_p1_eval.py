_base_ = ['./drivetransformer_base_res480x960_p1_10ep.py']
model = dict(pretrained=None)
data = dict(workers_per_gpu=2)
custom_hooks = []
