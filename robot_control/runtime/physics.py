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


class CPUVectorPhysics:
    """Independent serial MuJoCo worlds for bounded CPU correctness tests.

    Uses named qpos/dof/actuator maps and the same PD substeps as WarpPhysics.
    It is not a throughput substitute for GPU training.
    """
    def __init__(self, env, num_envs=1, device="cpu", **unused):
        import mujoco
        import numpy as np
        self.mujoco, self.np = mujoco, np
        self.device = torch.device(device)
        if self.device.type != "cpu":
            raise ValueError("CPUVectorPhysics only accepts CPU tensors")
        self.num_envs = int(num_envs)
        if self.num_envs < 1 or self.num_envs > 64:
            raise ValueError("CPU verification supports 1..64 worlds; select Warp for scale")
        self.layout = env.layout
        self.contract = dict(env.contract)
        self.substeps = round(env.control["control_dt"] / env.control["physics_dt"])
        if not math.isclose(self.substeps * env.control["physics_dt"], env.control["control_dt"], abs_tol=1e-10):
            raise ValueError("Control interval must be an integer multiple of physics interval")
        if env.control.get("actuator_delay_physics_steps", 0) != 0:
            raise ValueError("Delayed control requires a different state contract")
        self.models = [copy.copy(env.model) for _ in range(self.num_envs)]
        self.datas = [mujoco.MjData(model) for model in self.models]
        self.qids, self.vids, self.aids = env.qids.copy(), env.vids.copy(), env.actuator_ids.copy()
        self.torso, self.nominal_com = env.torso, env.nominal_com.copy()
        self.nominal, self.scale, self.kp, self.kd, self.effort = (
            x.copy() for x in (env.nominal, env.scale, env.kp, env.kd, env.effort))
        self.context = torch.zeros(self.num_envs, self.layout.context)
        self.context[:, 0] = 1.
        self._last_action = torch.zeros(self.num_envs, self.layout.dof)
        self.reset(torch.arange(self.num_envs), env.references[0, 0].expand(self.num_envs, -1), self.context.clone())

    def _validate(self, value, shape, name):
        if value.device.type != "cpu" or value.dtype != torch.float32 or tuple(value.shape) != tuple(shape):
            raise ValueError(f"{name} requires CPU float32, shape {shape}")
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"Nonfinite {name}")

    @torch.no_grad()
    def reset(self, indices, states, contexts):
        if indices.dtype != torch.long or indices.device.type != "cpu" or indices.ndim != 1:
            raise ValueError("Indices must be one-dimensional CPU long tensor")
        count = indices.numel()
        self._validate(states, (count, self.layout.state), "states")
        self._validate(contexts, (count, self.layout.context), "contexts")
        if len(indices.unique()) != count or bool(((indices < 0) | (indices >= self.num_envs)).any()):
            raise ValueError("Reset indices must be unique and in range")
        if bool((contexts[:, 0] < 0).any()):
            raise ValueError("Negative friction")
        mj, n = self.mujoco, self.layout.dof
        contexts = contexts.clone()
        self.context[indices] = contexts
        for row, index in enumerate(indices.tolist()):
            model, data = self.models[index], self.datas[index]
            state, context = states[row].numpy(), contexts[row].numpy()
            mj.mj_resetData(model, data)
            model.geom_friction[:, 0] = context[0]
            model.body_ipos[:] = self.nominal_com
            model.body_ipos[self.torso] += context[n + 1:n + 4]
            mj.mj_setConst(model, data)
            data.qpos[:7], data.qpos[self.qids] = state[:7], state[13:13 + n]
            data.qvel[:6], data.qvel[self.vids] = state[7:13], state[13 + n:13 + 2 * n]
            self._last_action[index] = states[row, 13 + 2 * n:]
            mj.mj_forward(model, data)

    def state(self):
        return torch.tensor(self.np.stack([self.np.concatenate((data.qpos[:7], data.qvel[:6],
            data.qpos[self.qids], data.qvel[self.vids], self._last_action[i].numpy()))
            for i, data in enumerate(self.datas)]), dtype=torch.float32)

    @torch.no_grad()
    def step(self, actions):
        self._validate(actions, (self.num_envs, self.layout.dof), "actions")
        mj, np = self.mujoco, self.np
        saturated = np.zeros((self.num_envs, self.layout.dof), dtype=bool)
        distances, contacts, constraints = [], [], []
        for i, (model, data) in enumerate(zip(self.models, self.datas)):
            target = self.nominal + self.scale * actions[i].numpy()
            offset = self.context[i, 1:1 + self.layout.dof].numpy()
            for _ in range(self.substeps):
                torque = self.kp * (target - data.qpos[self.qids] - offset) - self.kd * data.qvel[self.vids]
                saturated[i] |= np.abs(torque) >= .95 * self.effort
                data.ctrl[self.aids] = np.clip(torque, -self.effort, self.effort)
                mj.mj_step(model, data)
            mj.mj_forward(model, data)
            contacts.append(data.ncon)
            constraints.append(data.nefc)
            distances.append(min([0.] + [c.dist for c in data.contact]))
        self._last_action.copy_(actions)
        states = self.state()
        if not states.isfinite().all() or bool((states[:, 7:13 + 2 * self.layout.dof].abs() > 1e5).any()):
            raise FloatingPointError("Nonfinite or divergent CPU transition")
        return states, dict(torque_saturated=torch.tensor(saturated.any(axis=1)),
                            torque_saturated_joints=torch.tensor(saturated.sum(axis=1)),
                            min_contact_distance_m=torch.tensor(distances, dtype=torch.float32),
                            contact_count=torch.tensor(contacts), constraint_count=torch.tensor(constraints))

    def state_dict(self):
        spec = self.mujoco.mjtState.mjSTATE_INTEGRATION
        values = []
        for model, data in zip(self.models, self.datas):
            state = self.np.empty(self.mujoco.mj_stateSize(model, spec))
            self.mujoco.mj_getState(model, data, state, spec)
            values.append(torch.from_numpy(state))
        return dict(schema=1, backend="CPUVectorPhysics", contract=self.contract, num_envs=self.num_envs,
                    context=self.context.clone(), previous_action=self._last_action.clone(),
                    integration=torch.stack(values),
                    qacc=torch.tensor(self.np.stack([d.qacc.copy() for d in self.datas])))

    @torch.no_grad()
    def load_state_dict(self, snapshot):
        if snapshot.get("schema") != 1 or snapshot.get("backend") != "CPUVectorPhysics" or snapshot["contract"] != self.contract or snapshot["num_envs"] != self.num_envs:
            raise ValueError("Incompatible CPU physics checkpoint")
        self.reset(torch.arange(self.num_envs), self.state(), snapshot["context"])
        self._last_action.copy_(snapshot["previous_action"])
        for i, (model, data) in enumerate(zip(self.models, self.datas)):
            self.mujoco.mj_setState(model, data, snapshot["integration"][i].numpy(), self.mujoco.mjtState.mjSTATE_INTEGRATION)
            self.mujoco.mj_forward(model, data)
            data.qacc[:] = snapshot["qacc"][i].numpy()


def _kernels(wp):
    # Define lazily so CPU-only installations can still import training helpers.
    @wp.kernel(module="unique", enable_backward=False)
    def pd(qpos: wp.array2d[float], qvel: wp.array2d[float],
           action: wp.array2d[float], context: wp.array2d[float],
           nominal: wp.array[float], scale: wp.array[float],
           kp: wp.array[float], kd: wp.array[float], effort: wp.array[float],
           qids: wp.array[int], vids: wp.array[int], aids: wp.array[int],
           ctrl: wp.array2d[float], saturated: wp.array2d[int]):
        w, j = wp.tid()
        target = nominal[j] + scale[j] * action[w, j]
        torque = kp[j] * (target - qpos[w, qids[j]] - context[w, 1+j]) - kd[j] * qvel[w, vids[j]]
        ctrl[w, aids[j]] = wp.clamp(torque, -effort[j], effort[j])
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
    """Persistent independent worlds, using the same clipped PD as MojiaTracking.

    ``reset(indices, states, contexts)`` takes selected world IDs and tensors
    shaped (K,82), (K,27). State ordering is exactly ``kernel.STATE``. Context
    is friction, 23 encoder calibration offsets, torso COM displacement xyz.
    ``step(actions)`` accepts (N,23) and returns newly allocated (N,82) states
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
        if (env.model.nq, env.model.nv, env.model.nu) != (30,29,23):
            raise ValueError("WarpPhysics currently implements the Mojia 23 DOF state convention")
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
        self.context = torch.zeros(self.num_envs,27,device=self.device)
        self.context[:,0] = 1.
        self._actions = torch.zeros(self.num_envs,23,device=self.device)
        self._last_action = torch.zeros_like(self._actions)
        self._saturated = torch.zeros(self.num_envs,23,device=self.device,dtype=torch.int32)
        self._contact_count = torch.zeros(self.num_envs,device=self.device,dtype=torch.int32)
        self._min_distance = torch.zeros(self.num_envs,device=self.device)
        self._overflow = torch.zeros(self.num_envs,device=self.device,dtype=torch.int32)
        self._mask = torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
        self._qids = torch.as_tensor(env.qids, device=self.device, dtype=torch.long)
        self._vids = torch.as_tensor(env.vids, device=self.device, dtype=torch.long)
        self._addresses = [wp.from_torch(torch.as_tensor(v, device=self.device, dtype=torch.int32))
                           for v in (env.qids, env.vids, env.actuator_ids)]
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
        self._validate(states,(n,82),"states")
        self._validate(contexts,(n,27),"contexts")
        if not n:
            return
        if bool(((indices < 0) | (indices >= self.num_envs)).any()) or len(indices.unique()) != n:
            raise ValueError("Reset indices must be unique and in range")
        if bool((contexts[:,0] < 0).any()):
            raise ValueError("Negative friction")
        contexts = contexts.clone()
        with self._scope():
            self._mask.zero_();self._mask[indices] = True
            self.mjw.reset_data(self.model,self.data,self._wp_mask)
            self.context[indices] = contexts
            self._overflow[indices] = 0
            self.qpos[indices,:7] = states[:,:7]
            self.qpos[indices[:,None],self._qids[None,:]] = states[:,13:36]
            self.qvel[indices,:6] = states[:,7:13]
            self.qvel[indices[:,None],self._vids[None,:]] = states[:,36:59]
            self._last_action[indices] = states[:,59:82]
            self._friction[indices,:,0] = contexts[:,0,None]
            self._body_ipos[indices] = self._nominal_com
            self._body_ipos[indices,self.torso] += contexts[:,24:27]
            if self._const_graph is None:
                self.mjw.set_const_0(self.model,self.data)
            else:
                self.wp.capture_launch(self._const_graph)
            self.mjw.forward(self.model,self.data)
            self._assert_capacity()

    def state(self):
        """Return a snapshot, including the actually executed previous action."""
        return torch.cat((self.qpos[:,:7],self.qvel[:,:6],self.qpos[:,self._qids],
                          self.qvel[:,self._vids],self._last_action),dim=-1)

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
        states = torch.cat((qpos[:,:7],qvel[:,:6],qpos[:,self._qids],qvel[:,self._vids],previous),-1)
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
            wp.launch(self._pd,dim=(self.num_envs,23),inputs=[self.data.qpos,self.data.qvel,
                      self._wp_actions,self._wp_context,*self._control,*self._addresses],
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
        self._validate(actions,(self.num_envs,23),"actions")
        with self._scope():
            self._actions.copy_(actions)
            if self._step_graph is None:
                self._advance()
            else:
                self.wp.capture_launch(self._step_graph)
            self._assert_capacity()
            self._last_action.copy_(actions)
            state = self.state()
            if not torch.isfinite(state).all() or bool((state[:,7:59].abs() > 1e5).any()):
                raise FloatingPointError("Nonfinite or divergent Warp physical transition")
            diagnostics = dict(torque_saturated=self._saturated.any(dim=-1).clone(),
                               torque_saturated_joints=self._saturated.sum(dim=-1),
                               min_contact_distance_m=self._min_distance.clone(),
                               contact_count=self._contact_count.clone(),
                               constraint_count=self.wp.to_torch(self.data.nefc).clone())
            return state,diagnostics

