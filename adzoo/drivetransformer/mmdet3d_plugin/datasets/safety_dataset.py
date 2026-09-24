"""Dense safety labels kept exclusively in training metadata, in current LiDAR."""
import gzip
import json
from pathlib import Path
import numpy as np
from mmcv.datasets import DATASETS
from .bench2drive_drivetransformer_dataset import B2D_DriveTransformer_Dataset


@DATASETS.register_module()
class SafeComfortDataset(B2D_DriveTransformer_Dataset):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        assert self.sample_interval_ego_fut==1 and self.future_frames_ego_fix_time==30
        assert tuple(self.data_aug_conf['rot3d_range'])==(0,0)
        self._safety_ego={}

    def ego_geometry(self,info):
        folder=info['folder']
        if folder not in self._safety_ego:
            p=Path(self.data_root)/folder/'anno'/('%05d.json.gz'%info['frame_idx'])
            with gzip.open(p,'rt') as f:raw=json.load(f)
            ego=next(b for b in raw['bounding_boxes'] if b['class']=='ego_vehicle')
            # Raw CARLA is left-handed; converted ego is x-forward,y-left.
            loc=np.asarray(ego['bbx_loc'],dtype=np.float64)*[1,-1,1]
            offset=(np.linalg.inv(info['sensors']['LIDAR_TOP']['lidar2ego']) @ np.r_[loc,1])[:3]
            dims=np.asarray(ego['extent'])*2
            assert np.allclose(info['ego_size'],dims[[1,0,2]],atol=1e-5)
            self._safety_ego[folder]=(offset,int(ego['id']))
        return self._safety_ego[folder]

    def safety_labels(self,index):
        cur=self.get_data_by_index(index);w2l=np.asarray(cur['sensors']['LIDAR_TOP']['world2lidar'])
        offset,ego_id=self.ego_geometry(cur)
        frames=[];all_ids=set()
        for step in range(31):
            idx=index+step
            if not self.is_in_same_route(index,idx):break
            f=self.get_data_by_index(idx)
            if f['frame_idx'] != cur['frame_idx']+step:break
            ids=np.asarray(f['gt_ids']).astype(np.int64)
            assert len(np.unique(ids))==len(ids)
            names=[self.NameMapping.get(str(n),str(n)) for n in f['gt_names']]
            # Exclude non-collidable light/sign trigger boxes; keep physical
            # vehicles, VRUs, cones and labeled physical 'others'.
            keep=np.array([n in ('car','van','truck','bicycle','pedestrian','traffic_cone','others') for n in names]) & (ids!=ego_id)
            boxes=np.asarray(f['gt_boxes'])[keep,:7].copy();ids=ids[keep]
            assert np.isfinite(boxes).all() and (boxes[:,3:6]>0).all()
            transform=w2l @ np.linalg.inv(f['sensors']['LIDAR_TOP']['world2lidar'])
            boxes[:,:3]=(np.c_[boxes[:,:3],np.ones(len(boxes))] @ transform.T)[:,:3]
            # Rotate width-axis headings, avoiding a pi/2 width/length swap.
            dirs=np.c_[np.cos(boxes[:,6]),np.sin(boxes[:,6]),np.zeros(len(boxes))] @ transform[:3,:3].T
            boxes[:,6]=np.arctan2(dirs[:,1],dirs[:,0])
            ego_center=(transform @ np.r_[offset,1])[:3]
            ego_yaw=np.arctan2(transform[1,0],transform[0,0])
            eg=np.r_[ego_center,f['ego_size'],ego_yaw]
            frames.append((ids,boxes,eg));all_ids.update(ids.tolist())
        ids=sorted(all_ids);lookup={v:i for i,v in enumerate(ids)}
        boxes=np.zeros((31,len(ids),7),np.float32);valid=np.zeros((31,len(ids)),bool)
        # Invalid boxes have benign sizes; masks ensure they cannot contribute.
        boxes[...,3:6]=1;ego_boxes=np.zeros((31,7),np.float32);ego_boxes[:,3:6]=cur['ego_size']
        time_valid=np.zeros(31,bool)
        for t,(actors,b,eg) in enumerate(frames):
            j=[lookup[int(i)] for i in actors];boxes[t,j]=b;valid[t,j]=True;ego_boxes[t]=eg;time_valid[t]=True
        velocity=np.linalg.inv(cur['sensors']['LIDAR_TOP']['lidar2ego'])[:3,:3] @ cur['ego_vel']
        return dict(boxes=boxes,valid=valid,ego_boxes=ego_boxes,time_valid=time_valid,
            ego_center_offset=offset.astype(np.float32),velocity=velocity[:2].astype(np.float32),
            actor_ids=np.asarray(ids),frame_idx=int(cur['frame_idx']),route=cur['folder'])

    def prepare_train_data(self,index,aug_config):
        result=super().prepare_train_data(index,aug_config)
        if result is not None:
            result['img_metas'].data[0]['safety_gt']=self.safety_labels(index)
        return result
