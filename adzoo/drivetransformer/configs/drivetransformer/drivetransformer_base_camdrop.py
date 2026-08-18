_base_ = ['./drivetransformer_base.py']

# Same train pipeline as base, plus RandomCameraBlackout after normalization.
point_cloud_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
class_names = [
    'car', 'van', 'truck', 'bicycle', 'traffic_sign', 'traffic_cone',
    'traffic_light', 'pedestrian', 'others'
]
collect_keys = [
    'lidar2img', 'cam_intrinsic', 'cam_extrinsic', 'timestamp', 'ego_pose',
    'ego_pose_inv', 'pad_shape', 'gt_traj_fut_classes', 'ego_fut_classes']

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='ResizeCropFlipImage'),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='RandomCameraBlackout', prob=0.15),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=True),
    dict(type='CustomObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='CustomObjectNameFilter', classes=class_names),
    dict(type='TrajPreprocess', pc_range=point_cloud_range, with_ego_fix_dist=True,
         ego_fut_offset_input=False, assign_class_for_ego=False),
    dict(type='CustomFormatBundle3D', class_names=class_names, with_ego=True,
         collect_keys=collect_keys),
    dict(type='CustomCollect3D',
         keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'ego_his_trajs',
               'fut_valid_flag_fix_time', 'ego_fut_trajs_fix_time',
               'ego_fut_masks_fix_time', 'fut_valid_flag_fix_dist',
               'ego_fut_trajs_fix_dist', 'ego_fut_masks_fix_dist',
               'ego_fut_cmd', 'ego_lcf_feat', 'gt_attr_labels',
               'prev_exists', 'index'] + collect_keys)
]

data = dict(train=dict(pipeline=train_pipeline))
