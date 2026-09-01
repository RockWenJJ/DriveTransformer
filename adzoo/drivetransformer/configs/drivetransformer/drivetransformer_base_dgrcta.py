_base_ = ['./drivetransformer_base.py']

# Preserve each task's own temporal attention, then add independently
# normalized, zero-initialized residual paths for map->agent, agent->ego,
# and map->ego.
model = dict(
    pts_bbox_head=dict(
        transformer=dict(
            decoder=dict(
                transformerlayers=dict(
                    directed_gated_cross_task_temporal_attn=True)))))
