_base_ = ['./drivetransformer_base.py']

# Experiment: in temporal cross attention, each task query (agent / map / ego)
# attends to the temporal memories of ALL tasks instead of only its own, so
# task queries can gather information from each other's history.
model = dict(
    pts_bbox_head=dict(
        transformer=dict(
            decoder=dict(
                transformerlayers=dict(
                    cross_task_temporal_attn=True)))))
