"""Motion/velocity interfaces for the revised SuperMimic training branch.

Paper contracts: Appendix B/C/G encoder widths, shared 32-D tanh latent,
downstream stop-gradient on decoder *weights*, per-interface critics, commands,
and Table A.6 reward.  The following are explicit local choices, because the
paper does not publish their exact formulas: motion features use relative root
orientation and heading-frame relative root xyz (2N+16/frame); velocity history is
the flattened ten observations; command scales are (2, 1.32, 3.35); and critics
include controller history and reference/command-process metadata.  These choices
are not author configuration claims. Relative root goals require a root pose in
the reference coordinate system; deployment needs a corresponding pose observer.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.func import functional_call
from torch.nn import functional as F

from .kernel import apply, mlp, rotation, spans
from ..robot_layout import RobotLayout
from .normalization import RunningNorm

MOTION, VELOCITY = 0, 1
LATENT = 32
GOAL_METADATA = 5
COMMAND_LOW = (-1.5, -1.32, -3.35)
COMMAND_HIGH = (2.0, 1.32, 3.35)
COMMAND_SCALE = (2.0, 1.32, 3.35)


def heading_rotation(state):
    """World-from-yaw-frame rotation, avoiding a yaw angle wrap discontinuity."""
    mat = rotation(state[..., 3:7])
    xy = F.normalize(mat[..., :2, 0], dim=-1, eps=1e-8)
    # Heading is undefined for an exactly vertical root x-axis.  Pick world x
    # there; this is a disclosed numerical convention, not a physical reset.
    fallback = torch.zeros_like(xy)
    fallback[..., 0] = 1
    xy = torch.where((mat[..., :2, 0].square().sum(-1) < 1e-12)[..., None], fallback, xy)
    c, s = xy.unbind(-1)
    zero, one = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack((c, -s, zero, s, c, zero, zero, zero, one), -1).reshape(*c.shape, 3, 3)


def motion_goal_features(state, future_states):
    """2N+16/frame; relative XY and heading errors are observable to the actor.

    Relative displacement is expressed in the *current robot's* heading frame,
    recomputed at each imagined step and differentiated through current state.
    Orientation is reference root orientation relative to the current root.
    Translating/rotating robot and reference together in XY/yaw leaves the
    features unchanged, unlike absolute-reference orientation with no actor yaw.
    """
    Q,DQ,LAST=spans(state)
    relative = future_states[..., :3] - state[..., None, :3]
    heading = heading_rotation(state).transpose(-1, -2)[..., None, :, :]
    reference_rotation = rotation(future_states[..., 3:7])
    relative_rotation = rotation(state[..., 3:7]).transpose(-1, -2)[..., None, :, :] @ reference_rotation
    return torch.cat((future_states[..., Q], future_states[..., DQ], future_states[..., 2:3],
                      relative_rotation[..., :, :2].flatten(-2),
                      apply(reference_rotation.transpose(-1, -2), future_states[..., 7:10]),
                      future_states[..., 10:13], apply(heading, relative)), -1)


def normalize_command(command):
    """Local max-absolute range normalization; no clipping of reference data."""
    return command / command.new_tensor(COMMAND_SCALE)


def _interfaces(interface, state):
    if interface is None:
        return torch.zeros(state.shape[0], dtype=torch.long, device=state.device)
    interface = torch.as_tensor(interface, dtype=torch.long, device=state.device)
    if interface.ndim == 0:
        interface = interface.expand(state.shape[0])
    if interface.shape != state.shape[:1]:
        raise ValueError("interface must have one ID per batch row")
    if bool(((interface != MOTION) & (interface != VELOCITY)).any()):
        raise ValueError("Only motion=0 and velocity=1 interfaces are implemented")
    return interface


class MotionVelocityActor(nn.Module):
    def __init__(self, history=10, future=10, encoder=(2048, 1024, 512, 256),
                 decoder=(2048, 1024, 512, 256), velocity_encoder=(256, 256), layout=None):
        super().__init__()
        self.history, self.future = int(history), int(future)
        self.layout=layout or RobotLayout(23,history,future)
        GOAL_DIM,LATENT,OBS,N=self.layout.goal,self.layout.latent,self.layout.observation,self.layout.dof
        self.obs_norm=RunningNorm(OBS)
        self.goal_norm=RunningNorm(self.future*GOAL_DIM)
        self.motion_encoder = mlp(self.future * GOAL_DIM, LATENT, encoder)
        self.velocity_encoder = mlp(self.history * OBS + 3, LATENT, velocity_encoder)
        self.decoder = mlp(self.history * OBS + LATENT, N, decoder)
        nn.init.normal_(self.decoder[-1].weight, std=.001)
        nn.init.zeros_(self.decoder[-1].bias)

    @torch.no_grad()
    def update_normalizers(self,history,future,state):
        self.obs_norm.update(history[:,-1])
        self.goal_norm.update(motion_goal_features(state,future).flatten(-2))

    def motion_encode(self, state, future_states):
        goal = motion_goal_features(state, future_states)
        return self.motion_encoder(self.goal_norm(goal.flatten(-2))).tanh()

    def velocity_encode(self, history, command):
        return self.velocity_encoder(torch.cat((history.flatten(-2), normalize_command(command)), -1)).tanh()

    def decode(self, history, latent, freeze_weights=False):
        inputs = torch.cat((self.obs_norm(history).flatten(-2), latent), -1)
        if freeze_weights:
            # Detaching the output, using no_grad, or freezing the whole actor
            # here would also kill the downstream encoder/input Jacobian.
            weights = {name: value.detach() for name, value in self.decoder.named_parameters()}
            return functional_call(self.decoder, weights, (inputs,))
        return self.decoder(inputs)

    def forward(self, history, future_states, state, interface=None, command=None,
                freeze_velocity_decoder=True):
        ids = _interfaces(interface, state)
        actions = state.new_zeros(state.shape[0], self.layout.dof)
        for ident in (MOTION, VELOCITY):
            mask = ids == ident
            if not bool(mask.any()):
                continue
            if ident == MOTION:
                latent = self.motion_encode(state[mask], future_states[mask])
            else:
                if command is None:
                    raise ValueError("Velocity actions require commands")
                latent = self.velocity_encode(history[mask], command[mask])
            actions[mask] = self.decode(history[mask], latent,
                                        freeze_weights=ident == VELOCITY and freeze_velocity_decoder)
        return actions


class IndependentCritics(nn.Module):
    """Separate parameters and task inputs; retain state/history input gradients.

    History augments the paper's state/goal/context inputs to represent this
    implementation's history-dependent controller. Five metadata features encode
    clip identity, phase, remaining duration, reference-command source and elapsed
    resampling period. Their scaling is owned by the collector. These features
    make the task's exogenous goal process observable to the value function and
    are explicit local corrections, not published author input-layout claims.
    """
    def __init__(self, history=10, future=10, widths=(256, 128), layout=None):
        super().__init__()
        self.history, self.future = int(history), int(future)
        self.layout=layout or RobotLayout(23,history,future)
        STATE,CONTEXT,OBS,GOAL_DIM=self.layout.state,self.layout.context,self.layout.observation,self.layout.goal
        common = STATE + CONTEXT + self.history * OBS + GOAL_METADATA
        self.motion_norm=RunningNorm(common+self.future*GOAL_DIM)
        self.velocity_norm=RunningNorm(common+3)
        self.motion = mlp(common + self.future * GOAL_DIM, 1, widths)
        self.velocity = mlp(common + 3, 1, widths)
        for network in (self.motion, self.velocity):
            nn.init.zeros_(network[-1].weight)
            nn.init.zeros_(network[-1].bias)

    def input_features(self, history, future_states, state, context, command=None, goal_metadata=None, update=False):
        if goal_metadata is None:
            goal_metadata = state.new_zeros(state.shape[0], GOAL_METADATA)
        if goal_metadata.shape != (state.shape[0], GOAL_METADATA):
            raise ValueError("Critic goal metadata must be [batch,5]")
        common = torch.cat((state, context, history.flatten(-2), goal_metadata), -1)
        motion = torch.cat((common, motion_goal_features(state, future_states).flatten(-2)), -1)
        velocity = None if command is None else torch.cat((common, normalize_command(command)), -1)
        if update:
            self.motion_norm.update(motion)
            if velocity is not None:self.velocity_norm.update(velocity)
        return self.motion_norm(motion), None if velocity is None else self.velocity_norm(velocity)

    def forward(self, history, future_states, state, context, interface=None, command=None, goal_metadata=None):
        ids = _interfaces(interface, state)
        values = state.new_zeros(state.shape[0])
        for ident, network in ((MOTION, self.motion), (VELOCITY, self.velocity)):
            mask = ids == ident
            if not bool(mask.any()):
                continue
            row_command = None if command is None else command[mask]
            row_metadata = None if goal_metadata is None else goal_metadata[mask]
            features = self.input_features(history[mask], future_states[mask], state[mask],
                                            context[mask], row_command, row_metadata)[ident]
            if features is None:
                raise ValueError("Velocity critic requires commands")
            values[mask] = network(features).squeeze(-1)
        return values


def reference_velocity_command(states, frames=None, window=25):
    """Centered 25-frame means in each reference frame's heading coordinates.

    `states` is one clip [T,13+3N]. Returns [T,3], or frames.shape+[3]. At clip
    boundaries the centered window is cropped, an explicit local convention.
    """
    if states.ndim != 2 or (states.shape[-1]-13)%3 != 0 or not len(states):
        raise ValueError("Expected one nonempty [T,13+3N] reference clip")
    if window <= 0 or window % 2 != 1:
        raise ValueError("Centered command window must have positive odd length")
    local_v = apply(heading_rotation(states).transpose(-1, -2), states[:, 7:10])
    world_w = apply(rotation(states[:, 3:7]), states[:, 10:13])
    raw = torch.cat((local_v[:, :2], world_w[:, 2:3]), -1)
    at = torch.arange(len(states), device=states.device) if frames is None else torch.as_tensor(frames, device=states.device, dtype=torch.long)
    at = at.clamp(0, len(states)-1)
    lower, upper = (at-window//2).clamp_min(0), (at+window//2+1).clamp_max(len(states))
    sums = torch.cat((raw.new_zeros(1, 3), raw.cumsum(0)), 0)
    return (sums[upper]-sums[lower]) / (upper-lower)[..., None]


class VelocityCommandSampler:
    """Appendix G per-environment command source and held random command.

    Call reset at a physics reset and call step once after each completed
    control step with that next step's reference commands. Sources are sampled
    at reset and after 250 controls. Reference commands refresh every call;
    random commands remain constant until resampling. No global RNG is owned
    unless an explicit torch.Generator is supplied.
    """
    def __init__(self, nworlds, device="cpu", *, dtype=torch.float32, period=250,
                 mode="rl", generator=None):
        if mode not in ("rl", "distillation") or period <= 0:
            raise ValueError("Expected rl/distillation mode and a positive period")
        self.nworlds, self.device = int(nworlds), torch.device(device)
        self.dtype, self.period, self.mode = dtype, int(period), mode
        self.generator = generator
        self.reference_mask = torch.ones(self.nworlds, dtype=torch.bool, device=self.device)
        self.held_commands = torch.zeros(self.nworlds, 3, dtype=dtype, device=self.device)
        self.commands = self.held_commands.clone()
        self.elapsed = torch.zeros(self.nworlds, dtype=torch.long, device=self.device)

    def _resample(self, ids):
        count = len(ids)
        if not count:
            return
        rand = lambda *shape: torch.rand(*shape, device=self.device, dtype=self.dtype, generator=self.generator)
        self.reference_mask[ids] = True if self.mode == "distillation" else rand(count) < .5
        low, high = self.commands.new_tensor(COMMAND_LOW), self.commands.new_tensor(COMMAND_HIGH)
        sampled = low + (high-low)*rand(count, 3)
        sampled[rand(count) < .1] = 0
        self.held_commands[ids] = sampled
        self.elapsed[ids] = 0

    def refresh(self, reference_commands):
        if reference_commands.shape != self.commands.shape:
            raise ValueError("Reference commands must be [nworlds,3]")
        self.commands.copy_(torch.where(self.reference_mask[:, None], reference_commands, self.held_commands))
        return self.commands

    def reset(self, ids=None, reference_commands=None):
        ids = (torch.arange(self.nworlds, device=self.device) if ids is None else
               torch.as_tensor(ids, device=self.device, dtype=torch.long).reshape(-1))
        self._resample(ids)
        if reference_commands is not None:
            if reference_commands.shape == self.commands.shape:
                reference_commands = reference_commands[ids]
            elif reference_commands.shape != (len(ids), 3):
                raise ValueError("Reset reference commands must be [nworlds,3] or [len(ids),3]")
            self.commands[ids] = torch.where(self.reference_mask[ids, None],
                                             reference_commands, self.held_commands[ids])
        else:
            self.commands[ids] = self.held_commands[ids]
        return self.commands

    def step(self, reference_commands):
        self.elapsed += 1
        self._resample(torch.nonzero(self.elapsed >= self.period, as_tuple=False).flatten())
        return self.refresh(reference_commands)

    def state_dict(self):
        result = {"nworlds": self.nworlds, "period": self.period, "mode": self.mode}
        for name in ("reference_mask", "held_commands", "commands", "elapsed"):
            result[name] = getattr(self, name).detach().cpu().clone()
        if self.generator is not None:
            result["generator_state"] = self.generator.get_state().cpu()
        return result

    def load_state_dict(self, saved):
        if (saved["nworlds"], saved["period"], saved["mode"]) != (self.nworlds, self.period, self.mode):
            raise ValueError("Command sampler configuration does not match checkpoint")
        for name in ("reference_mask", "held_commands", "commands", "elapsed"):
            target, source = getattr(self, name), saved[name]
            if target.shape != source.shape:
                raise ValueError(f"Command sampler tensor shape mismatch: {name}")
            target.copy_(source.to(target))
        if "generator_state" in saved:
            if self.generator is None:
                raise ValueError("Checkpoint needs its explicit command generator")
            self.generator.set_state(saved["generator_state"].cpu())


class InterfaceTasks(nn.Module):
    def __init__(self, base_task, mode="rl", torso_index=None):
        super().__init__()
        if mode not in ("rl", "distillation"):
            raise ValueError("Velocity mode must be rl or distillation")
        self.base = base_task
        self.mode = mode
        # FK excludes MuJoCo world body. 15 is named torso_link for this G1;
        # callers with another model must pass model.body('torso_link').id-1.
        if torso_index is None:raise ValueError("A named torso FK index is required")
        self.torso_index = int(torso_index)

    @property
    def nominal(self):
        return self.base.nominal

    @property
    def base_task(self):
        return self.base

    @property
    def fk(self):
        return self.base.fk

    def velocity_reward(self, state, next_state, action, command):
        Q,DQ,LAST=spans(state)
        heading = heading_rotation(next_state).transpose(-1, -2)
        local_v = apply(heading, next_state[..., 7:10])
        world_w = apply(rotation(next_state[..., 3:7]), next_state[..., 10:13])
        _, matrices, _, omegas = self.fk(next_state)
        torso_z = apply(heading, matrices[..., self.torso_index, :, 2])
        torso_w = apply(heading, omegas[..., self.torso_index, :])
        track_v = 2*torch.exp(-(local_v[..., :2]-command[..., :2]).square().sum(-1)/.5**2)
        # Table A.6 explicitly uses 0.5 (not 0.5**2) for yaw-rate error.
        track_yaw = 2*torch.exp(-(world_w[..., 2]-command[..., 2]).square()/.5)
        smooth = (action-state[..., LAST]).square().sum(-1)
        limits = (F.relu(self.base.lower-next_state[..., Q]) +
                  F.relu(next_state[..., Q]-self.base.upper)).sum(-1)
        sideways = torso_z[..., 1].square()
        roll_pitch = torso_w[..., :2].square().sum(-1)
        return self.base.dt*(track_v+track_yaw-.3*smooth-2*limits-4*sideways-.05*roll_pitch)

    def reward(self, state, next_state, action, reference, interface=None, command=None,
               reference_kinematics=None):
        ids = _interfaces(interface, state)
        if self.mode == "distillation":
            return self.base.reward(state, next_state, action, reference, reference_kinematics)
        result = state.new_zeros(state.shape[0])
        for ident in (MOTION, VELOCITY):
            mask = ids == ident
            if not bool(mask.any()):
                continue
            if ident == MOTION:
                fk = None if reference_kinematics is None else tuple(x[mask] for x in reference_kinematics)
                result[mask] = self.base.reward(state[mask], next_state[mask], action[mask], reference[mask], fk)
            else:
                if command is None:
                    raise ValueError("Velocity reward requires commands")
                result[mask] = self.velocity_reward(state[mask], next_state[mask], action[mask], command[mask])
        return result

    def termination(self, state, reference, soft, interface=None):
        motion = self.base.termination(state, reference, soft)
        if self.mode == "distillation":
            return motion
        gravity_z = -rotation(state[..., 3:7])[..., 2, 2]
        if soft:
            fall = 1-(1-torch.sigmoid((.35-state[..., 2])/.02))*(1-torch.sigmoid((1+gravity_z-.7)/.05))
        else:
            fall = (state[..., 2] < .35) | (1+gravity_z > .7)
        return torch.where(_interfaces(interface, state) == VELOCITY, fall, motion)


def latent_alignment_loss(actor, history, future_states, state, command,
                          reference_mask, interface=None, weight=.1):
    """RL reference-command samples only; teacher motion encoder is detached."""
    mask = torch.as_tensor(reference_mask, device=state.device, dtype=torch.bool)
    if interface is not None:
        mask = mask & (_interfaces(interface, state) == VELOCITY)
    velocity = actor.velocity_encode(history, command)
    motion = actor.motion_encode(state, future_states).detach()
    error = (velocity-motion).square().mean(-1)
    return weight*(error*mask).sum()/mask.sum().clamp_min(1)
