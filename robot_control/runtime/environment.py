"""Isolated 23-DOF Mojia reference environment; never imports old teacher weights.

The supported baseline is torque-clipped, zero-delay PD. The learned physical
representation has 82 dimensions. Exact numerical simulator restoration also
uses the separately saved integration/warmstart state; this is not a proof of
Markov sufficiency for real delayed actuators, which are rejected here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch

from ..robot_layout import RobotLayout

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT
AUDIT = ROOT / "configs/robot.json"

def sha(path):
    path = Path(path)
    if not path.is_absolute():path = ROOT/path
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def audited_control(audit=AUDIT):
    audit = Path(audit)
    record = json.loads(audit.read_text())
    joints = record["joints"]
    urdf = ROOT / record["candidate_asset"]["path"]
    if sha(urdf) != record["candidate_asset"]["sha256"]:
        raise ValueError("Audited candidate URDF has changed; repeat the contract audit")
    fields = {"nominal": "nominal_rad", "stiffness": "kp", "damping": "kd",
              "effort_limit": "torque_limit_sim_nm", "armature": "armature",
              "velocity_limit": "urdf_velocity_rad_s", "lower": "lower_rad", "upper": "upper_rad"}
    return dict(schema=1, robot="mojia_v3_8", joint_names=[j["name"] for j in joints],
                **{k: [j[v] for j in joints] for k, v in fields.items()},
                action_scale=[record["nominal_action_scale_rad"]] * len(joints),
                physics_dt=record["physics_dt_s"], control_dt=record["control_dt_s"],
                actuator_delay_physics_steps=0, actuator_model="torque_clipped_zero_delay_pd",
                hardware_validated=False, root_body=record.get('root_body','base_link'),
                torso_body=record.get('torso_body','lumbar_yaw'), tracking_points=record.get('tracking_points',[]), urdf=str(urdf.relative_to(ROOT)), urdf_sha256=sha(urdf),
                mesh_sha256={str(p.relative_to(ROOT)): sha(p) for p in sorted((urdf.parent.parent / "meshes").rglob("*")) if p.is_file()},
                audit_sha256=sha(audit), velocity_basis="URDF conservative limit; not hardware certified")


def build_model(control):
    if control.get("actuator_delay_physics_steps", 0) != 0 or control.get("actuator_model") != "torque_clipped_zero_delay_pd":
        raise ValueError("Only zero-delay PD is implemented; delayed actuators require explicit queue state")
    urdf = Path(control["urdf"])
    if not urdf.is_absolute():urdf = ROOT/urdf
    if sha(urdf) != control["urdf_sha256"]:
        raise ValueError("Robot asset digest mismatch")
    for filename, expected in control.get("mesh_sha256", {}).items():
        if sha(filename) != expected:
            raise ValueError(f"Robot mesh asset digest mismatch: {filename}")
    source = ET.parse(urdf).getroot()
    compiler = source.find("mujoco/compiler")
    if compiler is None:
        compiler = ET.SubElement(ET.SubElement(source, "mujoco"), "compiler")
    meshdir = urdf.parent.parent / "meshes"
    compiler.set("meshdir", str(meshdir))
    compiler.set("fusestatic", "false")
    for link in source.findall("link"):
        for kind in ("visual", "collision"):
            for i, geom in enumerate(link.findall(kind)):
                geom.set("name", f"{link.get('name')}_{kind}_{i}")
    initial = mujoco.MjModel.from_xml_string(ET.tostring(source, encoding="unicode"))
    with tempfile.TemporaryDirectory(prefix="mojia_isolated_") as tmp:
        canonical = Path(tmp) / "robot.xml"
        mujoco.mj_saveLastXML(str(canonical), initial)
        root = ET.parse(canonical).getroot()
    root.find("compiler").set("meshdir", str(meshdir))
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.attrib.update(timestep=str(control["physics_dt"]), gravity="0 0 -9.81",
                         solver="Newton", iterations="100", integrator="implicitfast")
    world = root.find("worldbody")
    base = world.find("body[@name='base_link']")
    if base is None:
        raise ValueError("Expected a named base_link root")
    ET.SubElement(base, "freejoint", name="floating_base")
    ET.SubElement(world, "geom", name="floor", type="plane", size="20 20 .1",
                  friction="1 .005 .0001")
    actuator = ET.SubElement(root, "actuator")
    for i, name in enumerate(control["joint_names"]):
        joint = root.find(f".//joint[@name='{name}']")
        if joint is None:
            raise ValueError(f"Missing named joint: {name}")
        joint.attrib.update(armature=str(control["armature"][i]), damping="0", frictionloss="0")
        limit = control["effort_limit"][i]
        ET.SubElement(actuator, "motor", name=name + "_motor", joint=name,
                      gear="1", ctrllimited="true", ctrlrange=f"{-limit} {limit}")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    if (model.nq, model.nv, model.nu) != (30, 29, 23):
        raise ValueError("Expected one free root and 23 independently actuated hinges")
    if model.joint("floating_base").qposadr[0] != 0 or model.joint("floating_base").dofadr[0] != 0:
        raise ValueError("Floating root addresses must start at zero")
    return model


def named_addresses(model, names):
    qids = np.array([model.joint(n).qposadr[0] for n in names], dtype=np.int64)
    vids = np.array([model.joint(n).dofadr[0] for n in names], dtype=np.int64)
    aids = np.array([model.actuator(n + "_motor").id for n in names], dtype=np.int64)
    if len(set(qids)) != 23 or len(set(vids)) != 23 or len(set(aids)) != 23:
        raise ValueError("Joint and actuator address maps must be bijective")
    return qids, vids, aids


class ReferenceBank:
    def __init__(self, references, dof=23):
        self.dof = dof
        self.lengths = torch.tensor([len(r) for r in references], dtype=torch.long)
        self.offsets = torch.cat((torch.zeros(1, dtype=torch.long), self.lengths.cumsum(0)[:-1]))
        self.states = torch.cat(references)
        self._devices = {}

    def to(self, device):
        device = torch.device(device)
        if device == self.states.device:
            return self
        if str(device) not in self._devices:
            bank = object.__new__(ReferenceBank)
            bank.dof = self.dof
            for name in ("states", "lengths", "offsets"):
                setattr(bank, name, getattr(self, name).to(device))
            bank._devices = {}
            self._devices[str(device)] = bank
        return self._devices[str(device)]

    def __getitem__(self, index):
        clip, frame, *extra = index
        device = self.states.device
        clip = torch.as_tensor(clip, dtype=torch.long, device=device)
        if isinstance(frame, slice):
            if frame.stop is None:
                raise ValueError("Reference slices require an explicit stop")
            frame = torch.arange(frame.start or 0, frame.stop, frame.step or 1, device=device)
        frame = torch.as_tensor(frame, dtype=torch.long, device=device)
        clip, frame = torch.broadcast_tensors(clip, frame)
        if bool(((clip < 0) | (clip >= len(self.lengths))).any()):
            raise IndexError("Invalid clip index")
        length = self.lengths[clip]
        state = self.states[self.offsets[clip] + frame.clamp_min(0).minimum(length - 1)]
        result = state.clone()
        tail = (frame >= length)[..., None]
        result[..., 7:13] = torch.where(tail, 0., state[..., 7:13])
        a, b = 13 + self.dof, 13 + 2 * self.dof
        result[..., a:b] = torch.where(tail, 0., state[..., a:b])
        return result[(..., *extra)] if extra else result


class MojiaTracking:
    def __init__(self, assets=None, history=10, future=10, *, synthetic_only=False, audit=AUDIT):
        from .kernel import Task
        self.layout = RobotLayout(23, history=history, future=future)
        self.history, self.future = history, future
        self.assets = Path(assets).resolve() if assets else None
        if self.assets is None and not synthetic_only:
            raise ValueError("Prepared Mojia assets required; synthetic_only=True is reserved for smoke tests")
        manifest = None
        if self.assets:
            manifest = json.loads((self.assets / "manifest.json").read_text())
            if manifest.get("synthetic_only") and not synthetic_only:
                raise ValueError("Synthetic standing references may only be used with explicit smoke_test")
            for relative, expected in manifest["files"].items():
                path = (self.assets / relative).resolve()
                if not path.is_relative_to(self.assets) or sha(path) != expected:
                    raise ValueError(f"Prepared data digest/path mismatch: {relative}")
            for filename, expected in manifest.get("asset_dependencies", {}).items():
                if sha(filename) != expected:
                    raise ValueError(f"Target mesh dependency changed: {filename}")
            self.control = json.loads((self.assets / "control.json").read_text())
        else:
            self.control = audited_control(audit)
        self.model = build_model(self.control)
        self.data = mujoco.MjData(self.model)
        self.qids, self.vids, self.actuator_ids = named_addresses(self.model, self.control["joint_names"])
        self.nominal, self.kp, self.kd, self.effort, self.scale = (
            np.asarray(self.control[key], dtype=np.float64)
            for key in ("nominal", "stiffness", "damping", "effort_limit", "action_scale"))
        self.nominal_com = self.model.body_ipos.copy()
        if self.model.body(1).name != self.control.get('root_body','base_link'):
            raise ValueError('Free-root body contract mismatch')
        self.torso = self.model.body(self.control.get('torso_body','lumbar_yaw')).id
        compiled_bodies = [self.model.body(i).name for i in range(1,self.model.nbody)]
        if self.assets is None: self.control['body_names']=compiled_bodies
        if self.control.get('body_names') != compiled_bodies:
            raise ValueError('Robot control body-name order differs from compiled model')
        points=self.control.get('tracking_points')
        if not points: raise ValueError('Explicit tracking-point names and local offsets required')
        if len({point['name'] for point in points}) != len(points): raise ValueError('Duplicate tracking point names')
        self.tracking_points=points
        self.task = Task(self.model, self.nominal, [self.model.body(point["body"]).id - 1 for point in points],
                         qids=self.qids, vids=self.vids, dt=self.control["control_dt"],
                         tracked_offsets=[point["offset_local_m"] for point in points])
        refs, self.clip_names = [], []
        if manifest:
            for entry in manifest["clips"]:
                with np.load(self.assets / entry["file"], allow_pickle=False) as z:
                    if z["joint_names"].tolist() != self.control["joint_names"]:
                        raise ValueError("Prepared references must use the canonical named joint order")
                    if z['body_names'].tolist()!=compiled_bodies:
                        raise ValueError('Prepared motion body-name order differs from model')
                    state = z["states"]
                    if state.ndim != 2 or state.shape[1] != self.layout.state or len(state) < 2 or not np.isfinite(state).all():
                        raise ValueError("Invalid prepared state shape or values")
                    if not np.isclose(float(z["fps"].reshape(-1)[0]), 1 / self.control["control_dt"]):
                        raise ValueError("Reference control rate mismatch")
                    refs.append(torch.from_numpy(state.astype(np.float32)))
                    self.clip_names.append(Path(entry["file"]).stem)
        else:
            state = torch.zeros(128, self.layout.state)
            state[:, 2] = .87
            state[:, 3] = 1.
            state[:, 13:36] = torch.tensor(self.nominal)
            refs.append(state)
            self.clip_names.append("SYNTHETIC_STANDING_SMOKE_ONLY")
        if not refs:
            raise ValueError("At least one reference clip is required")
        self.lengths = [len(r) for r in refs]
        self.references = ReferenceBank(refs)
        self.contract = dict(robot="mojia_v3_8", schema=1, layout=self.layout.to_dict(),
                             joint_names=self.control["joint_names"], qpos_addresses=self.qids.tolist(),
                             qvel_addresses=self.vids.tolist(), actuator_ids=self.actuator_ids.tolist(),
                             urdf_sha256=self.control["urdf_sha256"], audit_sha256=self.control["audit_sha256"],
                             mesh_bundle_sha256=hashlib.sha256(json.dumps(
                                 manifest.get("asset_dependencies", {}) if manifest else self.control["mesh_sha256"],
                                 sort_keys=True).encode()).hexdigest(),
                             control_sha256=hashlib.sha256(json.dumps(self.control, sort_keys=True).encode()).hexdigest(),
                             manifest_sha256=sha(self.assets / "manifest.json") if manifest else None,
                             synthetic_only=bool(manifest.get("synthetic_only")) if manifest else True,
                             physics_dt=self.control["physics_dt"], control_dt=self.control["control_dt"],
                             actuator_model="torque_clipped_zero_delay_pd", mujoco=mujoco.__version__,
                             hardware_validated=False)

        self.contract.update(root_body=self.control['root_body'],torso_body=self.control['torso_body'],
                             tracking_points=self.tracking_points,future_frames='t+1 through t+future',
                             state_units='metres, wxyz, world body-origin m/s, root-local rad/s, joint rad and rad/s')
        self.contract["dynamics_representation"]="independent local pose and velocity increments; normalized 58-D output"

    def state(self, last):
        return torch.tensor(np.concatenate((self.data.qpos[:7], self.data.qvel[:6],
                                             self.data.qpos[self.qids], self.data.qvel[self.vids],
                                             np.asarray(last))), dtype=torch.float32)

    def reset(self, clip, frame, context):
        state = self.references[clip, frame].cpu().numpy()
        context = np.asarray(context, dtype=np.float64)
        if context.shape != (27,) or not np.isfinite(context).all() or context[0] < 0:
            raise ValueError("Expected finite friction + 23 calibration + COM xyz context")
        mujoco.mj_resetData(self.model, self.data)
        self.model.geom_friction[:, 0] = context[0]
        self.model.body_ipos[:] = self.nominal_com
        self.model.body_ipos[self.torso] += context[24:27]
        mujoco.mj_setConst(self.model, self.data)
        self.data.qpos[:7] = state[:7]
        self.data.qpos[self.qids] = state[13:36]
        self.data.qvel[:6] = state[7:13]
        self.data.qvel[self.vids] = state[36:59]
        mujoco.mj_forward(self.model, self.data)
