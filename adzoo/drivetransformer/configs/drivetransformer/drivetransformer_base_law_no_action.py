"""Same LAW parameters, data, schedule and cost; zero the action condition."""
_base_ = ['./drivetransformer_base_law.py']
model = dict(law_cfg=dict(use_action=False))
