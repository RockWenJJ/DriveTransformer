"""DT-Base detection/motion + HiP planning without mapping, controlled 10-epoch first experiment.

Keep the established res480 P1/map-datafix recipe, detection and motion losses,
ResNet initialization, optimizer, sampler and controller parameters. Only the
planning representation, image reading, supervision and output adapter change.
Run from the repository root, as required by the original DT launch scripts.
"""
from copy import deepcopy as _deepcopy
from mmcv import Config as _Config

_base_ = ['./drivetransformer_res480_p1_map_datafix_10ep.py']
_baseline = _Config.fromfile(
    'adzoo/drivetransformer/configs/drivetransformer/'
    'drivetransformer_res480_p1_map_datafix_10ep.py')
_collect = list(_baseline.data.train.collect_keys) + ['hip_plan_targets', 'hip_plan_masks']
_pipeline = _deepcopy(_baseline.data.train.pipeline)
for _step in _pipeline:
    if _step['type'] == 'CustomFormatBundle3D':
        _step['collect_keys'] = _collect
    elif _step['type'] == 'CustomCollect3D':
        _step['keys'] = list(_step['keys']) + ['hip_plan_targets', 'hip_plan_masks']

model = dict(pts_bbox_head=dict(with_mapping=False, hip_planner=dict(
    anchor_paths='data/anchors/hip', num_attention_groups=8,
    query_chunk_size=8, checkpoint_sampling=True)))
data = dict(train=dict(with_mapping=False, map_file=None, hip_planning_targets=True, collect_keys=_collect,
                       pipeline=_pipeline),
            val=dict(with_mapping=False, map_file=None, hip_planning_targets=True),
            test=dict(with_mapping=False, map_file=None, hip_planning_targets=True))
inference_contract = dict(planning_decoder='hip_xy',
    hip_time_dt_s=0.2, hip_space_step_m=2.0,
    hip_openloop_source='selected_temp_2hz_linear_interpolation',
    hip_control_source='selected_style_5hz_with_temporal_fallback')
load_from = None
resume_from = None
work_dir = './work_dirs/dt_hip_planning_10ep'
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook'),
                                    dict(type='TensorboardLoggerHook')])
# Existing Base mixed-precision backend and 20 x 4 budget are unchanged.
runner = dict(type='IterBasedRunner', max_iters=29350)
checkpoint_config = dict(interval=1000, save_last=True, max_keep_ckpts=3)
find_unused_parameters = True
del _deepcopy, _Config, _baseline, _collect, _pipeline, _step
