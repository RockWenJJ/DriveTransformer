_base_ = ['./drivetransformer_res480_no_map_10ep.py']

# The external frozen snapshot is part of the checkpoint contract. Override paths
# together when moving to a different host; do not change its revision or hashes.
qwen_root = '/data5T/wenjj/Projects/DriveTransformer_worktrees/qwen08_frozensem_20260924'
qwen_revision = '2fc06364715b967f1860aea9cf38778875588b17'
scene_prompt = '''Analyze the ego vehicle's current front-view image for trajectory planning. Focus on:

1. Road geometry, lane boundaries, intersections, and drivable space.
2. Traffic lights, signs, and stop lines, including their relevance to the ego lane.
3. The relative positions and orientations of vehicles, pedestrians, cyclists, and obstacles.
4. Visible conditions affecting forward progress, turning, slowing down, stopping, or yielding, including occlusions and visibility limits.

Base the analysis on visible evidence. Preserve uncertainty about unclear signals, right-of-way, and motion that cannot be determined from a single image.'''

model = dict(type='QwenFrozenSemDriveTransformer', pretrained=None, img_neck=None,
    img_backbone=dict(_delete_=True, type='FrozenQwenBackbone', out_channels=512,
        model_path=qwen_root+'/weights/'+qwen_revision,
        worker_python=qwen_root+'/qwen_env/bin/python', revision=qwen_revision,
        prompt=scene_prompt, vision_chunk=6, language_chunk=1,
        memory_fraction=0.20, timeout=600),
    pts_bbox_head=dict(type='QwenFrozenSemHead', language_dim=1024, img_stride=32))

qwen_contract = dict(name='DT-Qwen08-FrozenSem', revision=qwen_revision,
    frozen_modules=['ViT', 'PatchMerger', 'language_model'],
    camera_order=['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                  'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'],
    front_camera_index=0, visual_dim=1024, language_dim=1024, dt_dim=512,
    patch_size=16, merge_size=2, output_stride=32,
    readout='last_valid_token', learned_language_token=False,
    enable_thinking=False, generate_text=False,
    language_fusion='final_decoder_planning_heads_only',
    external_weights_manifest=qwen_root+'/weights_verified.json')

log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook'),
    dict(type='TensorboardLoggerHook'), dict(type='WandbLoggerHook',
    init_kwargs=dict(project='DriveTransformer', name='DT-Qwen08-FrozenSem',
                     id='qwen08-frozensem-20260924', resume='never'))])
work_dir = './adzoo/drivetransformer/work_dirs/qwen08_frozensem_10ep'
