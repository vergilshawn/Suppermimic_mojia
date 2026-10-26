"""Dimension-generic differentiable kinematics, reward, and gradient primitives.

Independent runtime: no imports from the running G1 training namespace.
State is canonical named-joint order, not MuJoCo's internal joint order.
"""
from contextlib import contextmanager
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ..robot_layout import RobotLayout


def layout_of(state):
    if (state.shape[-1]-13)%3 or state.shape[-1]<=13:
        raise ValueError('Expected root13 + q,dq,last_action in canonical joint order')
    return RobotLayout((state.shape[-1]-13)//3)


def spans(state):
    n=layout_of(state).dof
    return slice(13,13+n),slice(13+n,13+2*n),slice(13+2*n,13+3*n)


def mlp(inputs,outputs,widths):
    layers=[]
    for width in widths:
        layers.extend((nn.Linear(inputs,width),nn.ELU()));inputs=width
    return nn.Sequential(*layers,nn.Linear(inputs,outputs))


def rotation(q):
    w,x,y,z=F.normalize(q,dim=-1).unbind(-1)
    return torch.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
                        2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
                        2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),-1).reshape(*q.shape[:-1],3,3)


def apply(r,v):return (r@v.unsqueeze(-1)).squeeze(-1)


def proprio(state,nominal,context,noise=False):
    q,dq,last=spans(state);n=nominal.shape[-1]
    if state.shape[-1]!=13+3*n or context.shape[-1]!=n+4:
        raise ValueError('State, nominal, and randomized context dimensions disagree')
    r=rotation(state[...,3:7])
    obs=torch.cat((state[...,q]+context[...,1:1+n]-nominal,state[...,dq],
                   state[...,10:13],-r[...,2,:],state[...,last]),-1)
    if noise:
        amplitudes=obs.new_tensor([.01]*n+[.5]*n+[.2]*3+[.05]*3+[0.]*n)
        obs=obs+(2*torch.rand_like(obs)-1)*amplitudes
    return obs


class Kinematics(nn.Module):
    """One free root and arbitrary named hinge order; fixed bodies/anchors allowed."""
    def __init__(self,model,qids=None,vids=None):
        super().__init__()
        qids=np.asarray(qids if qids is not None else np.arange(7,model.nq),dtype=int)
        vids=np.asarray(vids if vids is not None else np.arange(6,model.nv),dtype=int)
        self.layout=RobotLayout(len(qids))
        if model.nq!=7+len(qids) or model.nv!=6+len(qids):
            raise ValueError('Require one free root and N actuated hinges')
        if sorted(qids.tolist())!=list(range(7,model.nq)) or sorted(vids.tolist())!=list(range(6,model.nv)):
            raise ValueError('Joint addresses must cover all non-root coordinates exactly once')
        self.parents=model.body_parentid.tolist();self.joints=[]
        root=np.flatnonzero(model.jnt_type==0)
        if len(root)!=1 or model.jnt_bodyid[root[0]]!=1:
            raise ValueError('Require exactly one free joint on body 1')
        lookup={int(q):i for i,q in enumerate(qids)}
        axes=np.zeros((len(qids),3));anchors=np.zeros_like(axes);refs=np.zeros(len(qids))
        for body in range(model.nbody):
            ids=np.flatnonzero(model.jnt_bodyid==body)
            if body<=1:
                self.joints.append(-1);continue
            if len(ids)>1 or (len(ids) and int(model.jnt_type[ids[0]])!=3):
                raise ValueError('Only fixed bodies and single-hinge child bodies are supported')
            if not len(ids):self.joints.append(-1);continue
            joint=int(ids[0]);index=lookup[int(model.jnt_qposadr[joint])]
            if int(model.jnt_dofadr[joint])!=vids[index]:
                raise ValueError('qids and vids do not refer to the same canonical joint')
            self.joints.append(index);axes[index]=model.jnt_axis[joint]
            anchors[index]=model.jnt_pos[joint];refs[index]=model.qpos0[qids[index]]
        for name,value in (('offset',model.body_pos),('fixed_quat',model.body_quat),
                           ('axis',axes),('anchor',anchors),('reference_q',refs)):
            self.register_buffer(name,torch.tensor(value,dtype=torch.float32))

    def forward(self,state):
        if state.shape[-1]!=self.layout.state:raise ValueError('FK state dimension mismatch')
        batch=state.shape[:-1];n=self.layout.dof
        eye=torch.eye(3,dtype=state.dtype,device=state.device).expand(*batch,3,3)
        zeros=state.new_zeros(*batch,3)
        positions,matrices,velocities,omegas=[zeros],[eye],[zeros],[zeros]
        fixed=rotation(self.fixed_quat.to(state))
        for body in range(1,len(self.parents)):
            if body==1:
                pos,mat=state[...,:3],rotation(state[...,3:7])
                vel,omega=state[...,7:10],apply(mat,state[...,10:13])
            else:
                parent=self.parents[body]
                displacement=apply(matrices[parent],self.offset[body].to(state).expand(*batch,3))
                unrotated=positions[parent]+displacement
                mat=matrices[parent]@fixed[body];omega=omegas[parent]
                index=self.joints[body]
                if index>=0:
                    axis=self.axis[index].to(state).expand(*batch,3)
                    anchor=self.anchor[index].to(state).expand(*batch,3)
                    joint_world=unrotated+apply(mat,anchor)
                    world_axis=apply(mat,axis)
                    angle=(state[...,13+index]-self.reference_q[index])/2
                    quat=torch.cat((angle.cos()[...,None],axis*angle.sin()[...,None]),-1)
                    mat=mat@rotation(quat)
                    pos=joint_world-apply(mat,anchor)
                    joint_omega=world_axis*state[...,13+n+index:14+n+index]
                    vel=(velocities[parent]+torch.linalg.cross(omegas[parent],pos-positions[parent])
                         +torch.linalg.cross(joint_omega,pos-joint_world))
                    omega=omega+joint_omega
                else:
                    pos=unrotated
                    vel=velocities[parent]+torch.linalg.cross(omegas[parent],displacement)
            positions.append(pos);matrices.append(mat);velocities.append(vel);omegas.append(omega)
        return tuple(torch.stack(items,-3 if i==1 else -2) for i,items in
                     enumerate((positions[1:],matrices[1:],velocities[1:],omegas[1:])))


def orientation_error(a,b):
    relative=a@b.transpose(-1,-2)
    cosine=((relative.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1)
    safe=cosine.clamp(-1+1e-6,1-1e-6)
    return torch.where(cosine>.9999,2*(1-cosine),torch.acos(safe).square())


class Task(nn.Module):
    def __init__(self,model,nominal,tracked,qids=None,vids=None,dt=.02):
        super().__init__();self.layout=RobotLayout(len(nominal));self.dt=float(dt)
        qids=np.asarray(qids if qids is not None else np.arange(7,model.nq),dtype=int)
        self.fk=Kinematics(model,qids,vids);self.tracked=list(tracked)
        self.register_buffer('nominal',torch.tensor(nominal,dtype=torch.float32))
        joint_for_q={int(q):i for i,q in enumerate(model.jnt_qposadr)}
        ranges=torch.tensor(model.jnt_range[[joint_for_q[int(q)] for q in qids]],dtype=torch.float32)
        middle,half=ranges.mean(-1),(ranges[:,1]-ranges[:,0])*.45
        self.register_buffer('lower',middle-half);self.register_buffer('upper',middle+half)

    def reward(self,state,next_state,action,reference,reference_kinematics=None):
        q,_,last=spans(state);p,r,v,w=self.fk(next_state)
        rp,rr,rv,rw=self.fk(reference) if reference_kinematics is None else reference_kinematics
        t=self.tracked
        delta=r[...,0,:,:] @ rr[...,0,:,:].transpose(-1,-2)
        yaw=torch.atan2(delta[...,1,0],delta[...,0,0]);c=yaw.cos();sn=yaw.sin();z=torch.zeros_like(c);one=torch.ones_like(c)
        heading=torch.stack((c,-sn,z,sn,c,z,z,z,one),-1).reshape(*yaw.shape,3,3)
        anchor=torch.cat((p[...,0,:2],rp[...,0,2:3]),-1)
        aligned_p=anchor[...,None,:]+apply(heading[...,None,:,:],rp-rp[...,0:1,:])
        aligned_r=heading[...,None,:,:] @ rr
        terms=(torch.exp(-(p[...,t,:]-aligned_p[...,t,:]).square().sum(-1).mean(-1)/.3**2)
               +torch.exp(-orientation_error(r[...,t,:,:],aligned_r[...,t,:,:]).mean(-1)/.4**2)
               +torch.exp(-(v[...,t,:]-rv[...,t,:]).square().sum(-1).mean(-1))
               +torch.exp(-(w[...,t,:]-rw[...,t,:]).square().sum(-1).mean(-1)/3.14**2)
               +.5*torch.exp(-(p[...,0,:]-rp[...,0,:]).square().sum(-1)/.3**2)
               +.5*torch.exp(-orientation_error(r[...,0,:,:],rr[...,0,:,:])/.4**2))
        limits=(F.relu(self.lower-next_state[...,q])+F.relu(next_state[...,q]-self.upper)).sum(-1)
        smooth=(action-state[...,last]).square().sum(-1)
        return self.dt*(terms-2*limits-.1*smooth)

    def termination(self,state,reference,soft):
        tilt=(rotation(state[...,3:7])[...,2,2]-rotation(reference[...,3:7])[...,2,2]).abs()
        height=(state[...,2]-reference[...,2]).abs()
        if soft:return 1-(1-torch.sigmoid((tilt-.8)/.05))*(1-torch.sigmoid((height-.25)/.02))
        return (tilt>.8)|(height>.25)


@contextmanager
def frozen(*modules):
    parameters=[p for module in modules for p in module.parameters()]
    flags=[p.requires_grad for p in parameters]
    for p in parameters:p.requires_grad_(False);p.grad=None
    try:yield
    finally:
        for p,flag in zip(parameters,flags):p.requires_grad_(flag)


def lambda_returns(rewards,terminations,values,gamma=.99,lam=.95):
    result=values[-1].detach();targets=[]
    for t in reversed(range(len(rewards))):
        result=rewards[t].detach()+gamma*(1-terminations[t].detach())*((1-lam)*values[t+1].detach()+lam*result)
        targets.append(result)
    return torch.stack(targets[::-1])


def step(loss,optimizer,parameters,element_clip=False):
    parameters=[p for p in parameters if p.requires_grad]
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
    optimizer.zero_grad(set_to_none=True);loss.backward()
    norm=nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
    if element_clip:
        for p in parameters:
            if p.grad is not None:p.grad.clamp_(-.1,.1)
    optimizer.step()
    if any(not p.isfinite().all() for p in parameters):raise FloatingPointError('Nonfinite parameters')
    return float(norm)
