"""G1 differentiable dynamics, analytical full-body reward and BPTT kernels."""
import torch
from torch import nn
from torch.nn import functional as F

from .optim import frozen, lambda_returns

N = 29
DYNAMIC, STATE, OBS, CONTEXT = 71, 100, 93, 33
Q, DQ, LAST = slice(13, 42), slice(42, 71), slice(71, 100)


def mlp(inputs, outputs, widths):
    layers = []
    for width in widths:
        layers.extend((nn.Linear(inputs, width), nn.ELU()))
        inputs = width
    return nn.Sequential(*layers, nn.Linear(inputs, outputs))


def rotation(q):
    w, x, y, z = F.normalize(q, dim=-1).unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(*q.shape[:-1], 3, 3)


def apply(r, v):
    return (r @ v.unsqueeze(-1)).squeeze(-1)


def proprio(state, nominal, context, noise=False):
    r = rotation(state[..., 3:7])
    gravity = -r[..., 2, :]
    obs = torch.cat((state[..., Q] + context[..., 1:30] - nominal, state[..., DQ],
                     state[..., 10:13], gravity, state[..., LAST]), -1)
    if noise:
        amplitudes = obs.new_tensor([.01]*29 + [.5]*29 + [.2]*3 + [.05]*3 + [0.]*29)
        obs = obs + (2*torch.rand_like(obs)-1)*amplitudes
    return obs


def goal_features(states):
    r = rotation(states[..., 3:7])
    return torch.cat((states[..., Q], states[..., DQ], states[..., 2:3],
                      r[..., :, :2].flatten(-2), apply(r.transpose(-1,-2), states[..., 7:10]),
                      states[..., 10:13]), -1)


class Actor(nn.Module):
    def __init__(self, history=10, future=10, encoder=(2048,1024,512,256), decoder=(2048,1024,512,256)):
        super().__init__()
        self.encoder = mlp(future*71, 32, encoder)
        self.decoder = mlp(history*OBS + 32, N, decoder)
        # Train every layer from scratch; there is no teacher or reference-action bypass.
        nn.init.normal_(self.decoder[-1].weight, std=.001)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, history, future):
        latent = self.encoder(future).tanh()
        return self.decoder(torch.cat((history.flatten(-2), latent), -1))


class Dynamics(nn.Module):
    def __init__(self, widths=(2048,2048,2048)):
        super().__init__()
        self.net = mlp(STATE+N+CONTEXT, DYNAMIC, widths)
        self.register_buffer("center", torch.zeros(STATE))
        self.register_buffer("scale", torch.ones(STATE))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    @torch.no_grad()
    def calibrate(self, states):
        self.center.copy_(states.mean(0))
        self.scale.copy_(states.std(0, unbiased=False).clamp_min(.1))

    def forward(self, state, action, context):
        delta = self.net(torch.cat(((state-self.center)/self.scale, action, context), -1))
        dynamic = state[..., :DYNAMIC] + delta*self.scale[:DYNAMIC]
        return torch.cat((dynamic[..., :3], F.normalize(dynamic[..., 3:7], dim=-1),
                          dynamic[..., 7:], action), -1)

    def error(self, predicted, truth):
        sign = torch.where((predicted[...,3:7]*truth[...,3:7]).sum(-1,keepdim=True)<0,-1.,1.)
        aligned = torch.cat((truth[...,:3], truth[...,3:7]*sign, truth[...,7:DYNAMIC]),-1)
        return ((predicted[...,:DYNAMIC]-aligned)/self.scale[:DYNAMIC]).square().mean(-1)


class Kinematics(nn.Module):
    """Body-origin positions, rotations and velocities; exact for this hinge tree.

    Reject unsupported trees rather than silently approximating MuJoCo kinematics.
    """
    def __init__(self, model):
        super().__init__()
        import numpy as np
        if model.nq != 36 or model.nv != 35 or abs(model.jnt_pos).max() != 0 or abs(model.qpos0[7:]).max() != 0:
            raise ValueError("Expected one free root and 29 zero-anchor, zero-reference hinges")
        self.parents = model.body_parentid.tolist()
        self.joints = []
        for body in range(model.nbody):
            ids = np.flatnonzero(model.jnt_bodyid == body)
            if body > 1 and (len(ids) > 1 or (len(ids) and int(model.jnt_type[ids[0]]) != 3)):
                raise ValueError("Unsupported joint tree")
            self.joints.append(int(ids[0])-1 if len(ids) and body > 1 else -1)
        for name, value in (("offset", model.body_pos), ("fixed_quat", model.body_quat),
                            ("axis", model.jnt_axis[1:])):
            self.register_buffer(name, torch.tensor(value, dtype=torch.float32))

    def forward(self, state):
        batch = state.shape[:-1]
        eye = torch.eye(3, dtype=state.dtype, device=state.device).expand(*batch,3,3)
        zeros = torch.zeros(*batch,3,dtype=state.dtype,device=state.device)
        positions, matrices, velocities, omegas = [zeros], [eye], [zeros], [zeros]
        fixed = rotation(self.fixed_quat.to(state))
        for b in range(1, len(self.parents)):
            if b == 1:
                pos, mat = state[...,:3], rotation(state[...,3:7])
                vel, omega = state[...,7:10], apply(mat,state[...,10:13])
            else:
                p = self.parents[b]
                displacement = apply(matrices[p],self.offset[b].to(state).expand(*batch,3))
                pos = positions[p] + displacement
                mat = matrices[p] @ fixed[b]
                vel = velocities[p] + torch.linalg.cross(omegas[p],displacement)
                omega = omegas[p]
                j = self.joints[b]
                if j >= 0:
                    axis = self.axis[j].to(state).expand(*batch,3)
                    world_axis = apply(mat,axis)
                    angle = state[...,13+j]/2
                    quat = torch.cat((angle.cos()[...,None],axis*angle.sin()[...,None]),-1)
                    mat = mat @ rotation(quat)
                    omega = omega + world_axis*state[...,42+j:43+j]
            positions.append(pos); matrices.append(mat); velocities.append(vel); omegas.append(omega)
        return tuple(torch.stack(items,-3 if i==1 else -2) for i,items in
                     enumerate((positions[1:],matrices[1:],velocities[1:],omegas[1:])))


def orientation_error(a,b):
    """Squared geodesic angle; finite derivative at identity (no acos singularity)."""
    relative = a @ b.transpose(-1,-2)
    cos = ((relative.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1)
    # acos^2 approaches 2*(1-cos) at identity.
    safe = cos.clamp(-1+1e-6,1-1e-6)
    return torch.where(cos > .9999, 2*(1-cos), torch.acos(safe).square())


class Task(nn.Module):
    def __init__(self, model, nominal, tracked):
        super().__init__()
        self.fk = Kinematics(model)
        self.tracked = tracked
        self.register_buffer("nominal",torch.tensor(nominal,dtype=torch.float32))
        # BeyondMimic-style soft position limits, nominal factor 0.9 from mjlab.
        ranges = torch.tensor(model.jnt_range[1:],dtype=torch.float32)
        mid, half = ranges.mean(-1), (ranges[:,1]-ranges[:,0])*.45
        self.register_buffer("lower",mid-half)
        self.register_buffer("upper",mid+half)

    def reward(self, state, next_state, action, reference, reference_kinematics=None):
        p,r,v,w = self.fk(next_state)
        rp,rr,rv,rw = self.fk(reference) if reference_kinematics is None else reference_kinematics
        t = self.tracked
        delta=r[...,0,:,:] @ rr[...,0,:,:].transpose(-1,-2)
        yaw=torch.atan2(delta[...,1,0],delta[...,0,0]);c=yaw.cos();sn=yaw.sin();z=torch.zeros_like(c);one=torch.ones_like(c)
        heading=torch.stack((c,-sn,z,sn,c,z,z,z,one),-1).reshape(*yaw.shape,3,3)
        anchor=torch.cat((p[...,0,:2],rp[...,0,2:3]),-1)
        aligned_p=anchor[...,None,:]+apply(heading[...,None,:,:],rp-rp[...,0:1,:])
        aligned_r=heading[...,None,:,:] @ rr
        terms = (torch.exp(-(p[...,t,:]-aligned_p[...,t,:]).square().sum(-1).mean(-1)/.3**2)
                 +torch.exp(-orientation_error(r[...,t,:,:],aligned_r[...,t,:,:]).mean(-1)/.4**2)
                 +torch.exp(-(v[...,t,:]-rv[...,t,:]).square().sum(-1).mean(-1))
                 +torch.exp(-(w[...,t,:]-rw[...,t,:]).square().sum(-1).mean(-1)/3.14**2)
                 +.5*torch.exp(-(p[...,0,:]-rp[...,0,:]).square().sum(-1)/.3**2)
                 +.5*torch.exp(-orientation_error(r[...,0,:,:],rr[...,0,:,:])/.4**2))
        limits = (F.relu(self.lower-next_state[...,Q])+F.relu(next_state[...,Q]-self.upper)).sum(-1)
        smooth = (action-state[...,LAST]).square().sum(-1)
        return .02*(terms-2*limits-.1*smooth)

    def termination(self, state, reference, soft):
        tilt = (rotation(state[...,3:7])[...,2,2]-rotation(reference[...,3:7])[...,2,2]).abs()
        height = (state[...,2]-reference[...,2]).abs()
        if soft:
            return 1-(1-torch.sigmoid((tilt-.8)/.05))*(1-torch.sigmoid((height-.25)/.02))
        return (tilt>.8)|(height>.25)


def fit_loss(model, states, actions, context):
    state, loss = states[:,0], states.new_zeros(())
    for t in range(actions.shape[1]):
        state = model(state,actions[:,t],context)
        loss = loss+model.error(state,states[:,t+1]).mean()
    return loss/actions.shape[1]


def critic_input(state, future, context):
    return torch.cat((state,future,context),-1)


def imagine(model,actor,critic,task,initial,history,references,futures,context,noise=True,bootstrap=True):
    state, history = initial.detach(), history.detach()
    survival = state.new_ones(state.shape[0]); total = state.new_zeros(state.shape[0])
    rewards,terminals,values,inputs = [],[],[],[]
    horizon = futures.shape[1]-1
    # References do not depend on actions. Evaluate all reference kinematics in
    # one batch, instead of launching the same Python FK tree at every step.
    reference_kinematics=task.fk(references[:,1:])
    for t in range(horizon):
        inputs.append(critic_input(state,futures[:,t],context))
        values.append(critic(inputs[-1]).squeeze(-1))
        action = actor(history,futures[:,t])
        nxt = model(state,action,context)
        reward = task.reward(state,nxt,action,references[:,t+1],
                             tuple(x[:,t] for x in reference_kinematics))
        terminal = task.termination(nxt,references[:,t+1],soft=True)
        total = total+.99**t*survival*reward
        survival = survival*(1-terminal)
        rewards.append(reward);terminals.append(terminal)
        state = nxt
        obs = proprio(state,task.nominal,context,noise)
        history = torch.cat((history[:,1:],obs[:,None]),1)
    final = critic(critic_input(state,futures[:,-1],context)).squeeze(-1)
    values.append(final)
    if bootstrap:
        total = total+.99**horizon*survival*final
    targets = lambda_returns(rewards,terminals,values)
    return -total.mean(), torch.stack(inputs,1).detach(), targets.transpose(0,1).detach()


def step(loss, optimizer, parameters, element_clip=False):
    parameters = [p for p in parameters if p.requires_grad]
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
    if element_clip:
        for p in parameters:
            if p.grad is not None:
                p.grad.clamp_(-.1,.1)
    optimizer.step()
    if any(not p.isfinite().all() for p in parameters):
        raise FloatingPointError("Nonfinite parameters")
    return float(norm)
