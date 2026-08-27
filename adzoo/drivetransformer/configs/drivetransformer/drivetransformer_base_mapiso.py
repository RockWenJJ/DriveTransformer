_base_ = ['./drivetransformer_base.py']

# Experiment: in task self attention, map queries do NOT attend to agent / ego
# queries (map queries only interact among themselves). Agent and ego queries
# keep attending to all queries, as in the baseline.
model = dict(
    pts_bbox_head=dict(
        transformer=dict(
            decoder=dict(
                transformerlayers=dict(
                    isolate_map_task_attn=True)))))
