"""Adapted local-increment model inspired by the audited successful references."""
import torch
from torch import nn
from torch.nn import functional as F
from .kernel import mlp, rotation, apply, orientation_error, spans
from ..robot_layout import RobotLayout


def quat_multiply(a, b):
    aw, av = a[..., :1], a[..., 1:]
    bw, bv = b[..., :1], b[..., 1:]
    return torch.cat((aw*bw-(av*bv).sum(-1, keepdim=True), aw*bv+bw*av+torch.linalg.cross(av,bv)), -1)


def rotation_increment(vector):
    # sinc has a finite derivative at zero; no hidden normalization singularity.
    angle = vector.norm(dim=-1, keepdim=True)
    return torch.cat(((angle/2).cos(), .5*torch.sinc(angle/(2*torch.pi))*vector), -1)


class ConsistentDynamics(nn.Module):
    """12+2N local state increments; preserved name is not v3 checkpoint compatibility."""
    def __init__(self,widths=(2048,2048,2048),dt=.02,layout=None):
        super().__init__()
        self.layout=layout or RobotLayout(23);n=self.layout.dof
        from .normalization import RunningNorm
        self.in_norm=RunningNorm(self.layout.dynamics_input);self.out_norm=RunningNorm(12+2*n,eps=1e-4)
        layers=[];n=self.layout.dynamics_input
        for width in widths:
            layers.extend((nn.Linear(n,width),nn.LayerNorm(width),nn.SiLU()));n=width
        layers.append(nn.Linear(n,12+2*self.layout.dof));self.net=nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
        self.register_buffer('floor',torch.tensor([.003]*3+[.005]*3+[.02]*3+[.05]*3+[.003]*self.layout.dof+[.05]*self.layout.dof))
    def features(self,state,action,context):
        Q,DQ,LAST=spans(state)
        r=rotation(state[...,3:7])
        return torch.cat((state[...,2:3],-r[...,2,:],apply(r.transpose(-1,-2),state[...,7:10]),
                          state[...,10:13],state[...,Q],state[...,DQ]/10.,state[...,LAST],action,context),-1)
    @staticmethod
    def heading(state):
        r=rotation(state[...,3:7]);yaw=torch.atan2(r[...,1,0],r[...,0,0]);c=yaw.cos();s=yaw.sin();z=torch.zeros_like(c);o=torch.ones_like(c)
        return torch.stack((c,-s,z,s,c,z,z,z,o),-1).reshape(*yaw.shape,3,3)
    @staticmethod
    def rotation_delta(a,b):
        qa=F.normalize(a[...,3:7],dim=-1);qb=F.normalize(b[...,3:7],dim=-1)
        inv=torch.cat((qa[...,:1],-qa[...,1:]),-1);rel=quat_multiply(inv,qb)
        rel=torch.where(rel[...,:1]<0,-rel,rel);v=rel[...,1:];length=v.norm(dim=-1,keepdim=True)
        scale=torch.where(length>1e-6,2*torch.atan2(length,rel[...,:1])/length.clamp_min(1e-8),2*torch.ones_like(length))
        return v*scale
    def targets(self,a,b):
        Q,DQ,LAST=spans(a)
        return torch.cat((apply(self.heading(a).transpose(-1,-2),b[...,:3]-a[...,:3]),
                          self.rotation_delta(a,b),apply(rotation(a[...,3:7]).transpose(-1,-2),b[...,7:10]-a[...,7:10]),
                          b[...,10:13]-a[...,10:13],b[...,Q]-a[...,Q],b[...,DQ]-a[...,DQ]),-1)
    def integrate(self,s,d,action):
        Q,DQ,LAST=spans(s)
        dp,dr,dv,dw,dq,dqd=d.split((3,3,3,3,self.layout.dof,self.layout.dof),-1)
        rot=F.normalize(quat_multiply(s[...,3:7],rotation_increment(dr)),dim=-1)
        return torch.cat((s[...,:3]+apply(self.heading(s),dp),rot,
                          s[...,7:10]+apply(rotation(s[...,3:7]),dv),s[...,10:13]+dw,s[...,Q]+dq,s[...,DQ]+dqd,action),-1)
    def forward(self,s,action,context):
        d=self.net(self.in_norm(self.features(s,action,context)))*self.out_norm.std()+self.out_norm.mean
        return self.integrate(s,d,action)
    def error_groups(self,pred,truth):
        Q,DQ,LAST=spans(pred)
        # Fixed dimensional tolerances avoid suppressing XY by corpus trajectory range.
        return dict(root_position=((pred[...,:3]-truth[...,:3])/.05).square().mean(-1),
                    root_rotation=orientation_error(rotation(pred[...,3:7]),rotation(truth[...,3:7]))/.2**2,
                    root_velocity=(pred[...,7:10]-truth[...,7:10]).square().mean(-1),
                    root_angular_velocity=(pred[...,10:13]-truth[...,10:13]).square().mean(-1),
                    joint_position=((pred[...,Q]-truth[...,Q])/.1).square().mean(-1),
                    joint_velocity=((pred[...,DQ]-truth[...,DQ])/2.).square().mean(-1))

    def error(self,pred,truth):
        return (self.targets(truth,pred)/torch.maximum(self.out_norm.std(),self.floor)).square().mean(-1)


def masked_fit_loss(model,batch):
    states=batch['states'];actions=batch['actions'];mask=batch['mask'].bool()
    pred=states[:,0];total=pred.new_zeros(())
    for t in range(actions.shape[1]):
        pred=model(pred,actions[:,t],batch['contexts'])
        total=total+(model.error(pred,states[:,t+1])*mask[:,t]).sum()
    return total/states.shape[0]
