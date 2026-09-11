"""Future-frame pairing without annotations or temporal-memory side effects."""
import os


def find_future_pair(index, length, get_info, data_root, frame_offset,
                     frame_idx_step=1, frame_period_seconds=0.1):
    current = get_info(index)
    target_index = index + frame_offset
    meta = dict(current_index=index, future_index=target_index,
                current_frame=int(current['frame_idx']),
                scene_token=current['folder'], valid=False, reason='',
                horizon_seconds=frame_offset * frame_idx_step * frame_period_seconds)
    if target_index >= length:
        meta['reason'] = 'end_of_dataset'
        return None, meta
    target = get_info(target_index)
    meta.update(future_frame=int(target['frame_idx']), future_scene_token=target['folder'])
    if target['folder'] != current['folder']:
        meta['reason'] = 'scene_boundary'
        return None, meta
    frame_delta = int(target['frame_idx']) - int(current['frame_idx'])
    if frame_delta != frame_offset * frame_idx_step:
        meta['reason'] = 'frame_gap'
        return None, meta
    cameras = [key for key in current['sensors'] if 'CAM' in key]
    if not cameras or cameras != [key for key in target['sensors'] if 'CAM' in key]:
        meta['reason'] = 'camera_mismatch'
        return None, meta
    paths = [os.path.join(data_root, target['sensors'][key]['data_path']) for key in cameras]
    if not all(os.path.isfile(path) for path in paths):
        meta['reason'] = 'missing_image'
        return None, meta
    meta.update(valid=True, reason='valid', camera_names=cameras,
                delta_seconds=frame_delta * frame_period_seconds)
    return paths, meta
