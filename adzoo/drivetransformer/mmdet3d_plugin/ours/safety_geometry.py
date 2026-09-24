"""Training-only footprint, comfort and branch alignment losses (SI units)."""
import math
import torch


def signed_sat_gap(a, b):
    """Broadcast boxes [..., x,y,width,length,yaw]; positive means separated.

    At exact coincident centers choose the positive direction instead of the
    zero subgradient of abs. max chooses one escape axis deterministically.
    """
    def axes(t):
        c, s = t[..., 4].cos(), t[..., 4].sin()
        return torch.stack((torch.stack((c,s),-1),torch.stack((-s,c),-1)),-2)
    aa, bb = axes(a), axes(b)
    ax = torch.cat((aa,bb),-2)
    d = ((a[...,:2]-b[...,:2]).unsqueeze(-2)*ax).sum(-1)
    distance = torch.where(d >= 0, d, -d)
    ra = (torch.abs((ax.unsqueeze(-2)*aa.unsqueeze(-3)).sum(-1))*a[...,None,2:4]/2).sum(-1)
    rb = (torch.abs((ax.unsqueeze(-2)*bb.unsqueeze(-3)).sum(-1))*b[...,None,2:4]/2).sum(-1)
    return (distance-ra-rb).max(-1).values


def headings(points):
    delta = points[...,1:,:]-points[...,:-1,:]
    moving = delta.square().sum(-1) > 1e-4
    # yaw=0 has width on x, length on y (the dataset LiDAR convention).
    x = torch.where(moving,delta[...,0],torch.zeros_like(delta[...,0]))
    y = torch.where(moving,delta[...,1],torch.ones_like(delta[...,1]))
    yaw = torch.atan2(y,x)-math.pi/2
    return torch.cat((torch.zeros_like(yaw[...,:1]),yaw),-1)


def footprint_loss(pred, gt, ego_mask, safety, margin=.25):
    """One sample; 30 future LiDAR positions, ground-truth actor boxes at 10Hz.

    Also test interval midpoints, only for actors valid at both ends. Missing
    actors never receive interpolated/extrapolated labels. Height uses GT road
    elevation; predicted xy cannot invent an elevation to evade a collision.
    """
    device=pred.device
    def t(k):return torch.as_tensor(safety[k],device=device,dtype=torch.float32)
    boxes=t('boxes').detach(); valid=t('valid').bool(); eg=t('ego_boxes').detach()
    time_valid=t('time_valid').bool()
    mask=torch.cat((ego_mask.new_ones(1).bool(),ego_mask.bool())) & time_valid
    p=torch.cat((pred.new_zeros(1,2),pred),0); yaw=headings(p)
    off=t('ego_center_offset')[:2]
    c,s=yaw.cos(),yaw.sin()
    offset=torch.stack((c*off[0]-s*off[1],s*off[0]+c*off[1]),-1)
    xy=p+offset
    a=torch.cat((xy,eg[:,3:5],yaw[:,None]),-1)
    if boxes.shape[1]==0:return pred.sum()*0, pred.new_zeros(()),pred.new_zeros(())
    b=torch.cat((boxes[...,:2],boxes[...,3:5],boxes[...,6:7]),-1)
    def penalties(a,b,valid,az,ah,bz,bh,mask):
        vertical=(az[:,None]-bz).abs() < (ah[:,None]+bh)/2
        eligible=valid & vertical & mask[:,None]
        gap=signed_sat_gap(a[:,None].expand(-1,b.shape[1],-1),b)
        risk=(margin-gap).clamp_min(0).square()*eligible
        return risk.max(-1).values,eligible.any(-1)
    risk,eligible=penalties(a,b,valid,eg[:,2],eg[:,5],boxes[...,2],boxes[...,5],mask)
    # Shortest-angle interpolation of orientation at the half-step.
    def midpoint(x, yaw_col):
        m=(x[1:]+x[:-1])/2
        dy=x[1:,...,yaw_col]-x[:-1,...,yaw_col]
        dy=torch.atan2(dy.sin(),dy.cos())
        return torch.cat((m[...,:yaw_col],(x[:-1,...,yaw_col]+dy/2)[...,None]),-1)
    ar=midpoint(a,4);br=midpoint(b,4)
    midrisk,midvalid=penalties(ar,br,valid[1:]&valid[:-1],(eg[1:,2]+eg[:-1,2])/2,
        (eg[1:,5]+eg[:-1,5])/2,(boxes[1:,:,2]+boxes[:-1,:,2])/2,
        (boxes[1:,:,5]+boxes[:-1,:,5])/2,mask[1:]&mask[:-1])
    # Do not optimize an unavoidable collision at the observed t=0 itself.
    values=torch.cat((risk[1:],midrisk));weights=torch.cat((mask[1:],mask[1:]&mask[:-1]))
    loss=(values*weights).sum()/weights.sum().clamp_min(1)
    return loss,(values>0).float().sum()/weights.sum().clamp_min(1),torch.cat((eligible[1:],midvalid)).float().mean()


def comfort_loss(pred,gt,mask,velocity,dt=.1,accel_limits=(2.5,3.),jerk_limit=5.):
    """Normalized excess acceleration/jerk; expert maneuvers relax thresholds."""
    def derivatives(p):
        p=torch.cat((p.new_zeros(1,2),p),0)
        v=(p[1:]-p[:-1])/dt
        a=(v-torch.cat((velocity[None],v[:-1]),0))/dt
        # Express accelerations and jerks along/across the local travel direction.
        speed=torch.linalg.vector_norm(v,dim=-1,keepdim=True)
        direction=torch.where(speed>0.1,v/speed.clamp_min(.1),v.new_tensor([0.,1.]))
        lateral=torch.stack((direction[:,1],-direction[:,0]),-1)
        projected=torch.stack(((a*lateral).sum(-1),(a*direction).sum(-1)),-1)
        jerk=(a[1:]-a[:-1])/dt
        projected_jerk=torch.stack(((jerk*lateral[1:]).sum(-1),(jerk*direction[1:]).sum(-1)),-1)
        return projected,projected_jerk
    a,j=derivatives(pred);ga,gj=derivatives(gt.detach())
    ma=mask.bool() & torch.cat((mask.new_ones(1).bool(),mask[:-1].bool()))
    mj=ma[1:]&ma[:-1]
    limits=pred.new_tensor(accel_limits)
    at=torch.maximum(limits,ga.abs()+.5);jt=torch.maximum(pred.new_tensor(jerk_limit),gj.abs()+1.)
    pa=((a.abs()-at).clamp_min(0)/at).square().sum(-1)
    pj=((j.abs()-jt).clamp_min(0)/jt).square().sum(-1)
    loss=(pa*ma).sum()/ma.sum().clamp_min(1)+(pj*mj).sum()/mj.sum().clamp_min(1)
    return loss,a.detach().abs().max(),j.detach().abs().max()


def alignment_loss(pred, distance, time_mask, distance_mask):
    """Detached timed-path teacher at true cumulative arc length, sin bearing."""
    p=torch.cat((pred.new_zeros(1,2),pred.detach()),0)
    arc=torch.cat((pred.new_zeros(1),torch.linalg.vector_norm(p[1:]-p[:-1],dim=-1).cumsum(0)))
    targets=torch.arange(1,distance.numel()+1,device=pred.device,dtype=pred.dtype)
    indices=torch.searchsorted(arc.contiguous(),targets).clamp(1,len(arc)-1)
    denom=(arc[indices]-arc[indices-1]).clamp_min(1e-6)
    ratio=((targets-arc[indices-1])/denom).clamp(0,1)
    point=p[indices-1]+ratio[:,None]*(p[indices]-p[indices-1])
    norm=torch.linalg.vector_norm(point,dim=-1)
    prefix=time_mask.bool().long().cumprod(0).bool()
    valid=distance_mask.bool() & (targets<=arc[-1]) & prefix[indices-1] & (point[:,1]>=0) & (norm>1e-4)
    target=point[:,0]/norm.clamp_min(1e-4)
    return ((distance.flatten()-target).abs()*valid).sum()/valid.sum().clamp_min(1)
