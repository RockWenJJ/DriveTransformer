"""NoMap with training-only safety regularizers; inference/state dict unchanged."""
import torch
from mmcv.models import HEADS
from mmcv.runner.hooks import HOOKS, Hook
from .no_map import NoMapHead
from .safety_geometry import footprint_loss,comfort_loss,alignment_loss


@HEADS.register_module()
class SafeComfortHead(NoMapHead):
    def __init__(self,safety_cfg=None,**kwargs):
        super().__init__(**kwargs)
        self.safety_cfg=dict(collision_weight=1.,comfort_weight=1e-5,align_weight=.1,margin=.25)
        self.safety_cfg.update(safety_cfg or {})
        self.safety_ramp=0.

    def loss(self,*args,**kwargs):
        losses=super().loss(*args,**kwargs)
        pred=kwargs['preds_dicts']['ego_fut_preds_fix_time'][-1,:,0].float()
        dist=kwargs['preds_dicts']['ego_fut_preds_fix_dist'][-1,:,0].float()
        b=pred.shape[0];gt=kwargs['ego_fut_gt_fix_time'].reshape(b,30,2).float()
        mask=kwargs['ego_fut_masks_fix_time'].reshape(b,30)
        dm=kwargs['ego_fut_masks_fix_dist'].reshape(b,20)
        coll=[];comfort=[];align=[];risk=[];coverage=[];acc=[];jerk=[]
        for i in range(b):
            meta=kwargs['img_metas'][i];meta=meta[0] if 0 in meta else meta
            safety=meta['safety_gt']
            c,r,v=footprint_loss(pred[i],gt[i],mask[i],safety,self.safety_cfg['margin'])
            velocity=torch.as_tensor(safety['velocity'],device=pred.device,dtype=torch.float32)
            u,a,j=comfort_loss(pred[i],gt[i],mask[i],velocity)
            coll.append(c);comfort.append(u/(1+10*r.detach()));risk.append(r);coverage.append(v);acc.append(a);jerk.append(j)
            align.append(alignment_loss(pred[i],dist[i],mask[i],dm[i]))
        mean=lambda x:torch.stack(x).mean()
        for name,vals,w in [('collision',coll,'collision_weight'),('comfort',comfort,'comfort_weight'),('align',align,'align_weight')]:
            raw=mean(vals);losses['loss_safety_'+name]=raw*self.safety_cfg[w]*self.safety_ramp
            losses['safety_'+name+'_raw']=raw.detach()
        losses.update(safety_ramp=pred.new_tensor(self.safety_ramp),safety_risk=mean(risk).detach(),
            safety_actor_coverage=mean(coverage).detach(),safety_accel_max=mean(acc),safety_jerk_max=mean(jerk))
        return losses


@HOOKS.register_module()
class SafeComfortRampHook(Hook):
    def __init__(self,start=1468,end=4403,force=None):self.start=start;self.end=end;self.force=force
    def before_train_iter(self,runner):
        model=runner.model.module if hasattr(runner.model,'module') else runner.model
        model.pts_bbox_head.safety_ramp=float(self.force if self.force is not None else min(1.,max(0.,(runner.iter-self.start)/max(1,self.end-self.start))))
