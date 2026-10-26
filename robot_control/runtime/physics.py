"""Batched G1 MuJoCo Warp physics with exact per-world COM randomization.

This module only advances physics. The learner owns references, interface
commands, termination/truncation, histories and episode/replay bookkeeping.
Importing it does not require the optional Warp packages; constructing it does.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import math

import torch


def _kernels(wp):
    # Define lazily so CPU-only installations can still import training helpers.
    @wp.kernel(module="unique", enable_backward=False)
    def pd(qpos: wp.array2d[float], qvel: wp.array2d[float],
           action: wp.array2d[float], context: wp.array2d[float],
           nominal: wp.array[float], scale: wp.array[float],
           kp: wp.array[float], kd: wp.array[float], effort: wp.array[float],
           ctrl: wp.array2d[float], saturated: wp.array2d[int]):
        w, j = wp.tid()
        target = nominal[j] + scale[j] * action[w, j]
        torque = kp[j] * (target - qpos[w, 7+j] - context[w, 1+j]) - kd[j] * qvel[w, 6+j]
        ctrl[w, j] = wp.clamp(torque, -effort[j], effort[j])
        if wp.abs(torque) >= 0.95 * effort[j]:
            saturated[w, j] = 1

    @wp.kernel(module="unique", enable_backward=False)
    def contact_summary(nacon: wp.array[int], capacity: int,
                        world: wp.array[int], distance: wp.array[float],
                        counts: wp.array[int], minimum: wp.array[float]):
        i = wp.tid()
        if i < nacon[0] and i < capacity:
            w = world[i]
            if w >= 0 and w < counts.shape[0]:
                wp.atomic_add(counts, w, 1)
                wp.atomic_min(minimum, w, distance[i])

    @wp.kernel(module="unique", enable_backward=False)
    def capacities(nacon: wp.array[int], nefc: wp.array[int],
                   naconmax: int, njmax: int, overflow: wp.array[int]):
        w = wp.tid()
        if nacon[0] > naconmax or nefc[w] > njmax:
            overflow[w] = 1

    return pd, contact_summary, capacities


class WarpPhysics:
    """Persistent independent worlds, using the same clipped PD as G1Tracking.

    ``reset(indices, states, contexts)`` takes selected world IDs and tensors
    shaped (K,100), (K,33). State ordering is exactly ``kernel.STATE``. Context
    is friction, 29 encoder calibration offsets, torso COM displacement xyz.
    ``step(actions)`` accepts (N,29) and returns newly allocated (N,100) states
    and per-world diagnostic tensors. Returned states do not alias simulator
    memory and are safe to retain as transition endpoints.

    All calls run on the active PyTorch CUDA stream. NumPy/CPU inputs are not
    silently copied. Contact capacities are checked before returning any data;
    overflow raises instead of training on silently discarded constraints.
    """
    def __init__(self, env, num_envs=4096, device="cuda:0", *, nconmax=64,
                 njmax=256, use_cuda_graph=True):
        import mujoco_warp as mjw
        import warp as wp

        self.wp, self.mjw = wp, mjw
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("WarpPhysics requires an accessible CUDA device")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.num_envs = int(num_envs)
        if self.num_envs < 1 or nconmax < 1 or njmax < 1:
            raise ValueError("World and contact/constraint capacities must be positive")
        self.substeps = round(float(env.control["control_dt"]) / float(env.control["physics_dt"]))
        if not math.isclose(self.substeps*float(env.control["physics_dt"]),
                            float(env.control["control_dt"]), abs_tol=1e-10):
            raise ValueError("Control interval must contain an integer number of physics steps")
        if (env.model.nq, env.model.nv, env.model.nu) != (36,35,29):
            raise ValueError("WarpPhysics currently implements the G1 29 DOF state convention")
        wp.init()
        self._warp_stream = wp.Stream(str(self.device))
        self._torch_stream = wp.stream_to_torch(self._warp_stream)
        self._pd, self._contact_summary, self._capacities = _kernels(wp)
        self.cpu_model = copy.copy(env.model)
        self.contract = dict(env.contract)
        # The CPU environment may have been randomized previously. Restore the
        # canonical inertial offsets before constructing independent GPU worlds.
        self.cpu_model.body_ipos[:] = env.nominal_com
        self.torso = env.torso
        self._const_graph = self._step_graph = None
        self.context = torch.zeros(self.num_envs,33,device=self.device)
        self.context[:,0] = 1.
        self._actions = torch.zeros(self.num_envs,29,device=self.device)
        self._last_action = torch.zeros_like(self._actions)
        self._saturated = torch.zeros(self.num_envs,29,device=self.device,dtype=torch.int32)
        self._contact_count = torch.zeros(self.num_envs,device=self.device,dtype=torch.int32)
        self._min_distance = torch.zeros(self.num_envs,device=self.device)
        self._overflow = torch.zeros(self.num_envs,device=self.device,dtype=torch.int32)
        self._mask = torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
        self._nominal_com = torch.as_tensor(env.nominal_com,device=self.device,dtype=torch.float32)
        with self._scope():
            self.model = mjw.put_model(self.cpu_model)
            self.data = mjw.make_data(self.cpu_model,nworld=self.num_envs,nconmax=nconmax,njmax=njmax)
            # set_const_0 writes these arrays per world. Merely expanding
            # body_ipos leaves stale or racing inverse inertial weights.
            fields = ("body_ipos", "geom_friction", "dof_invweight0", "body_invweight0",
                      "tendon_invweight0", "tendon_length0", "cam_pos0", "cam_poscom0",
                      "cam_mat0", "light_pos0", "light_poscom0", "light_dir0",
                      "actuator_acc0", "actuator_biasprm")
            for name in fields:
                old = getattr(self.model,name)
                if old.shape[0] == 1 and self.num_envs != 1:
                    values = wp.to_torch(old).expand(self.num_envs,*wp.to_torch(old).shape[1:]).clone()
                    setattr(self.model,name,wp.from_torch(values,dtype=old.dtype))
            old = self.model.stat.meaninertia
            self.model.stat.meaninertia = wp.from_torch(wp.to_torch(old).expand(self.num_envs).clone())
            self.qpos = wp.to_torch(self.data.qpos)
            self.qvel = wp.to_torch(self.data.qvel)
            self._body_ipos = wp.to_torch(self.model.body_ipos)
            self._friction = wp.to_torch(self.model.geom_friction)
            self._wp_context = wp.from_torch(self.context)
            self._wp_actions = wp.from_torch(self._actions)
            self._wp_saturated = wp.from_torch(self._saturated)
            self._wp_contact_count = wp.from_torch(self._contact_count)
            self._wp_min_distance = wp.from_torch(self._min_distance)
            self._wp_overflow = wp.from_torch(self._overflow)
            self._wp_mask = wp.from_torch(self._mask)
            self._control = [wp.from_torch(torch.as_tensor(v,device=self.device,dtype=torch.float32))
                             for v in (env.nominal,env.scale,env.kp,env.kd,env.effort)]
            initial = env.references[0,0].to(self.device).expand(self.num_envs,-1).clone()
            indices = torch.arange(self.num_envs,device=self.device)
            self.reset(indices,initial,self.context.clone())
            if use_cuda_graph:
                # Compile first, then capture. set_const_0 explicitly supports
                # graph capture in MuJoCo Warp 3.8.1 and restores d.qpos itself.
                mjw.set_const_0(self.model,self.data)
                with wp.ScopedCapture() as capture:
                    mjw.set_const_0(self.model,self.data)
                self._const_graph = capture.graph
                self._advance()
                with wp.ScopedCapture() as capture:
                    self._advance()
                self._step_graph = capture.graph
                self.reset(indices,initial,self.context.clone())

    @contextmanager
    def _scope(self):
        wp = self.wp
        previous = torch.cuda.current_stream(self.device)
        self._torch_stream.wait_stream(previous)
        with torch.cuda.device(self.device), torch.cuda.stream(self._torch_stream), \
                wp.ScopedDevice(str(self.device)), wp.ScopedStream(self._warp_stream):
            yield
        previous.wait_stream(self._torch_stream)

    def _validate(self,value,shape,name):
        if value.device != self.device or value.dtype != torch.float32 or tuple(value.shape) != tuple(shape):
            raise ValueError(f"{name} must be float32 CUDA tensor on {self.device}, shape {tuple(shape)}")
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"Nonfinite {name}")

    @torch.no_grad()
    def reset(self, indices, states, contexts):
        if indices.device != self.device or indices.ndim != 1 or indices.dtype != torch.long:
            raise ValueError("indices must be a 1-D long tensor on the physics device")
        n = indices.numel()
        self._validate(states,(n,100),"states")
        self._validate(contexts,(n,33),"contexts")
        if not n:
            return
        if bool(((indices < 0) | (indices >= self.num_envs)).any()) or len(indices.unique()) != n:
            raise ValueError("Reset indices must be unique and in range")
        if bool((contexts[:,0] < 0).any()):
            raise ValueError("Negative friction")
        with self._scope():
            self._mask.zero_();self._mask[indices] = True
            self.mjw.reset_data(self.model,self.data,self._wp_mask)
            self.context[indices] = contexts
            self._overflow[indices] = 0
            self.qpos[indices,:7] = states[:,:7]
            self.qpos[indices,7:] = states[:,13:42]
            self.qvel[indices,:6] = states[:,7:13]
            self.qvel[indices,6:] = states[:,42:71]
            self._last_action[indices] = states[:,71:100]
            self._friction[indices,:,0] = contexts[:,0,None]
            self._body_ipos[indices] = self._nominal_com
            self._body_ipos[indices,self.torso] += contexts[:,30:33]
            if self._const_graph is None:
                self.mjw.set_const_0(self.model,self.data)
            else:
                self.wp.capture_launch(self._const_graph)
            self.mjw.forward(self.model,self.data)
            self._assert_capacity()

    def state(self):
        """Return a snapshot, including the actually executed previous action."""
        return torch.cat((self.qpos[:,:7],self.qvel[:,:6],self.qpos[:,7:],
                          self.qvel[:,6:],self._last_action),dim=-1)

    def state_dict(self):
        """Serialize integration state, not just qpos/qvel.

        CUDA graphs contain executable kernels only and are reconstructed by
        the constructor. All their persistent physical inputs are saved here.
        As with MuJoCo Warp generally, bitwise determinism across GPU/library
        versions is not promised; the contract is checked before restoration.
        """
        names = ("qpos", "qvel", "act", "qacc_warmstart", "ctrl", "qfrc_applied",
                 "xfrc_applied", "time", "eq_active", "mocap_pos", "mocap_quat", "qacc")
        with self._scope():
            return dict(schema=1,num_envs=self.num_envs,contract=self.contract,
                        mujoco_warp=self.mjw.__version__,warp=self.wp.__version__,
                        physics_dt=float(self.cpu_model.opt.timestep),substeps=self.substeps,
                        context=self.context.detach().cpu().clone(),
                        previous_action=self._last_action.detach().cpu().clone(),
                        integration={name:self.wp.to_torch(getattr(self.data,name)).detach().cpu().clone()
                                     for name in names})

    @torch.no_grad()
    def load_state_dict(self,snapshot):
        if snapshot["schema"] != 1 or snapshot["num_envs"] != self.num_envs or \
                snapshot["contract"] != self.contract or snapshot["substeps"] != self.substeps or \
                snapshot["physics_dt"] != float(self.cpu_model.opt.timestep) or \
                snapshot["mujoco_warp"] != self.mjw.__version__ or snapshot["warp"] != self.wp.__version__:
            raise ValueError("Incompatible Warp physics snapshot")
        integration = snapshot["integration"]
        context = snapshot["context"].to(self.device)
        previous = snapshot["previous_action"].to(self.device)
        qpos,qvel = integration["qpos"].to(self.device),integration["qvel"].to(self.device)
        states = torch.cat((qpos[:,:7],qvel[:,:6],qpos[:,7:],qvel[:,6:],previous),-1)
        self.reset(torch.arange(self.num_envs,device=self.device),states,context)
        with self._scope():
            for name,value in integration.items():
                target = self.wp.to_torch(getattr(self.data,name))
                if target.shape != value.shape or target.dtype != value.dtype:
                    raise ValueError(f"Incompatible integration field {name}")
                target.copy_(value.to(self.device))
            # Rebuild derived contacts/kinematics; forward does not advance time
            # or overwrite qacc_warmstart. Restore qacc as recorded afterwards.
            self.mjw.forward(self.model,self.data)
            self.wp.to_torch(self.data.qacc).copy_(integration["qacc"].to(self.device))
            self._assert_capacity()

    def _advance(self):
        wp,mjw = self.wp,self.mjw
        self._wp_saturated.zero_();self._wp_overflow.zero_()
        for _ in range(self.substeps):
            wp.launch(self._pd,dim=(self.num_envs,29),inputs=[self.data.qpos,self.data.qvel,
                      self._wp_actions,self._wp_context,*self._control],
                      outputs=[self.data.ctrl,self._wp_saturated])
            mjw.step(self.model,self.data)
            wp.launch(self._capacities,dim=self.num_envs,inputs=[self.data.nacon,self.data.nefc,
                      self.data.naconmax,self.data.njmax],outputs=[self._wp_overflow])
        mjw.forward(self.model,self.data)
        self._wp_contact_count.zero_();self._wp_min_distance.zero_()
        wp.launch(self._contact_summary,dim=self.data.naconmax,
                  inputs=[self.data.nacon,self.data.naconmax,self.data.contact.worldid,self.data.contact.dist],
                  outputs=[self._wp_contact_count,self._wp_min_distance])
        wp.launch(self._capacities,dim=self.num_envs,inputs=[self.data.nacon,self.data.nefc,
                  self.data.naconmax,self.data.njmax],outputs=[self._wp_overflow])

    def _assert_capacity(self):
        if bool((self.wp.to_torch(self.data.nacon) > self.data.naconmax).any()) or \
                bool((self.wp.to_torch(self.data.nefc) > self.data.njmax).any()) or \
                bool(self._overflow.any()):
            raise RuntimeError("Warp contact/constraint capacity exceeded; increase nconmax/njmax before collecting")

    @torch.no_grad()
    def step(self,actions):
        self._validate(actions,(self.num_envs,29),"actions")
        with self._scope():
            self._actions.copy_(actions)
            if self._step_graph is None:
                self._advance()
            else:
                self.wp.capture_launch(self._step_graph)
            self._assert_capacity()
            self._last_action.copy_(actions)
            state = self.state()
            if not torch.isfinite(state).all() or bool((state[:,7:71].abs() > 1e5).any()):
                raise FloatingPointError("Nonfinite or divergent Warp physical transition")
            diagnostics = dict(torque_saturated=self._saturated.any(dim=-1).clone(),
                               torque_saturated_joints=self._saturated.sum(dim=-1),
                               min_contact_distance_m=self._min_distance.clone(),
                               contact_count=self._contact_count.clone(),
                               constraint_count=self.wp.to_torch(self.data.nefc).clone())
            return state,diagnostics


def main():
    """Bounded physics-only verification, never starts a training job."""
    import argparse
    import json
    from pathlib import Path
    import time
    from .g1_env import G1Tracking, ROOT

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--assets",type=Path,default=ROOT/"data/g1")
    parser.add_argument("--worlds",type=int,default=4096)
    parser.add_argument("--steps",type=int,default=8)
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    env = G1Tracking(args.assets)
    torch.cuda.reset_peak_memory_stats()
    free_before,total = torch.cuda.mem_get_info()
    started = time.monotonic()
    physics = WarpPhysics(env,args.worlds)
    torch.cuda.synchronize()
    creation_s = time.monotonic()-started
    context = torch.zeros(args.worlds,33,device="cuda")
    context[:,0].uniform_(.3,1.2)
    context[:,1:30].uniform_(-.01,.01)
    context[:,30].uniform_(-.025,.025)
    context[:,31:33].uniform_(-.05,.05)
    initial = env.references[0,0].to("cuda").expand(args.worlds,-1).clone()
    physics.reset(torch.arange(args.worlds,device="cuda"),initial,context)
    torch.cuda.synchronize()
    free_after_creation,_ = torch.cuda.mem_get_info()
    action = torch.zeros(args.worlds,29,device="cuda")
    started = time.monotonic()
    for _ in range(args.steps):
        state,diagnostics = physics.step(action)
    torch.cuda.synchronize()
    elapsed = time.monotonic()-started
    free_after_steps,_ = torch.cuda.mem_get_info()
    result = dict(status="passed",kind="bounded_physics_only",worlds=args.worlds,
                  control_steps=args.steps,physics_substeps=physics.substeps*args.steps,
                  device=torch.cuda.get_device_name(),finite=bool(state.isfinite().all()),
                  control_dt=env.control["control_dt"],physics_dt=env.control["physics_dt"],
                  creation_s=creation_s,steps_s=elapsed,
                  physical_transitions_per_s=args.steps*args.worlds/elapsed,
                  incremental_cuda_memory_creation_bytes=free_before-free_after_creation,
                  incremental_cuda_memory_after_steps_bytes=free_before-free_after_steps,
                  total_device_bytes=total,torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  max_constraint_count=int(diagnostics["constraint_count"].max()),
                  max_contact_count=int(diagnostics["contact_count"].max()),
                  min_contact_distance_m=float(diagnostics["min_contact_distance_m"].min()),
                  context_ranges=dict(friction=[float(context[:,0].min()),float(context[:,0].max())],
                                      com_x=[float(context[:,30].min()),float(context[:,30].max())]),
                  note="Physics only; policy inference, rewards and replay transfers are excluded.")
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
