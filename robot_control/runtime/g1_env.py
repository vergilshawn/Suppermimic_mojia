"""Serial CPU MuJoCo reference tracker for a controlled G1 method experiment."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch

from .kernel import CONTEXT, LAST, STATE, Task, goal_features, proprio, rotation
from .references import ReferenceBank, FutureBank

ROOT = Path(__file__).resolve().parents[2]
TRACKED = ("pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
           "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
           "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
           "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class G1Tracking:
    def __init__(self, assets=ROOT/"data/g1", history=10, future=10, *, robot_only=False):
        self.assets = Path(assets).resolve()
        if robot_only:
            manifest={"clips":[],"files":{}}
        elif not (self.assets/"manifest.json").is_file():
            raise ValueError("Prepared G1 manifest required. Raw BONES CSV directories must first be converted with robot_control.runtime.prepare_lafan")
        else:manifest = json.loads((self.assets/"manifest.json").read_text())
        clips=manifest["clips"]
        # Packed states avoid global padded future tensors. Bound the CPU corpus
        # before loading; full BONES needs a streamed dataset backend.
        estimated_bytes=sum(int(e.get("frames",0)) for e in clips)*STATE*4*3
        if estimated_bytes>1024**3:
            raise ValueError("Reference subset exceeds the eager CPU cache budget; use a bounded subset or implement a streamed/batched backend before full-corpus training")
        for relative, expected in manifest["files"].items():
            if sha(self.assets/relative) != expected:
                raise ValueError(f"Asset hash mismatch: {relative}")
        self.control = json.loads((self.assets/"control.json").read_text())
        c = self.control
        tree = ET.parse(self.assets/"robot/g1.xml")
        root = tree.getroot()
        root.find("compiler").set("meshdir",str(self.assets/"robot/assets"))
        option = root.find("option")
        if option is None:
            option = ET.SubElement(root,"option")
        option.attrib.update(timestep=str(c["physics_dt"]),gravity="0 0 -9.81",integrator="implicitfast",
                             solver="Newton",iterations="100")
        ET.SubElement(root.find("worldbody"),"geom",name="floor",type="plane",size="20 20 .1",friction="1 .005 .0001")
        actuator = ET.SubElement(root,"actuator")
        for i,name in enumerate(c["joint_names"]):
            joint = root.find(f".//joint[@name='{name}']")
            joint.set("armature",str(c["armature"][i]))
            joint.set("damping","0")
            joint.set("frictionloss","0")
            limit = c["effort_limit"][i]
            ET.SubElement(actuator,"motor",name=name+"_motor",joint=name,gear="1",ctrllimited="true",ctrlrange=f"{-limit} {limit}")
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root,encoding="unicode"))
        self.data = mujoco.MjData(self.model)
        if [self.model.joint(i).name for i in range(1,30)] != c["joint_names"]:
            raise ValueError("Compiled joint order mismatch")
        self.history, self.future = history,future
        self.nominal = np.array(c["nominal"])
        self.kp,self.kd,self.effort,self.scale = (np.array(c[k]) for k in ("stiffness","damping","effort_limit","action_scale"))
        self.torso = self.model.body("torso_link").id
        self.nominal_com = self.model.body_ipos.copy()
        self.task = Task(self.model,self.nominal,[self.model.body(n).id-1 for n in TRACKED])
        references, self.lengths = [],[]
        self.clip_names = []
        for entry in manifest["clips"]:
            with np.load(self.assets/entry["file"],allow_pickle=False) as z:
                if z["joint_names"].tolist()!=c["joint_names"] or z["body_names"].tolist()!=c["body_names"]:
                    raise ValueError("Named motion order mismatch")
                states = torch.zeros(len(z["joint_pos"]),STATE)
                states[:,:3] = torch.from_numpy(z["body_pos_w"][:,0])
                states[:,3:7] = torch.from_numpy(z["body_quat_w"][:,0])
                states[:,7:10] = torch.from_numpy(z["body_lin_vel_w"][:,0])
                states[:,10:13] = (rotation(states[:,3:7]).transpose(-1,-2) @ torch.from_numpy(z["body_ang_vel_w"][:,0])[...,None]).squeeze(-1)
                states[:,13:42] = torch.from_numpy(z["joint_pos"])
                states[:,42:71] = torch.from_numpy(z["joint_vel"])
            references.append(states)
            self.lengths.append(len(states))
            self.clip_names.append(Path(entry["file"]).stem)
        if not references and robot_only:
            state=torch.zeros(2,STATE);state[:,2]=.8;state[:,3]=1;state[:,13:42]=torch.tensor(self.nominal);references=[state];self.lengths=[2];self.clip_names=["ROBOT_TEMPLATE_ONLY"]
        elif not references:
            raise ValueError("At least one reference is required")
        self.references = ReferenceBank(references)
        self.futures = FutureBank(self.references,future)
        self.contract = dict(robot="G1 29DOF",manifest_sha256=None if robot_only else sha(self.assets/"manifest.json"),
                             control_sha256=sha(self.assets/"control.json"),mujoco=mujoco.__version__,
                             backend="serial CPU MuJoCo",history=history,future=future,
                             tracked_bodies=list(TRACKED),context="friction,29 calibration offsets,torso COM xyz",
                             reward="paper Table A.2 structure; 14 mjlab tracked bodies; soft joint limits factor 0.9",
                             goal="10 future q,dq,root_z,rotation6d,root-frame v,w",
                             action="q_setpoint=nominal+per_joint_scale*action; explicit torque-clipped PD",
                             dynamics_state="root_xyz,wxyz,world_origin_v,body_w,29q,29dq,29previous_action")

    def state(self,last):
        return torch.tensor(np.concatenate((self.data.qpos[:7],self.data.qvel[:6],
                                            self.data.qpos[7:],self.data.qvel[6:],last)),dtype=torch.float32)

    def reset(self,clip,frame,context):
        mujoco.mj_resetData(self.model,self.data)
        state = self.references[clip,frame].numpy()
        self.data.qpos[:7]=state[:7];self.data.qpos[7:]=state[13:42]
        self.data.qvel[:6]=state[7:13];self.data.qvel[6:]=state[42:71]
        self.model.geom_friction[:,0]=float(context[0])
        self.model.body_ipos[:]=self.nominal_com
        self.model.body_ipos[self.torso]+=context[30:33].numpy()
        mujoco.mj_setConst(self.model,self.data)
        # mj_setConst can recompute derived constants; restore the chosen dynamic state.
        self.data.qpos[:7]=state[:7];self.data.qpos[7:]=state[13:42]
        self.data.qvel[:6]=state[7:13];self.data.qvel[6:]=state[42:71]
        mujoco.mj_forward(self.model,self.data)

    @torch.no_grad()
    def rollout(self,actor,seed,clip=0,start=0,duration=None,randomize=False,action_noise=0.,sensor_noise=False,policy_std=None,record_history=True):
        device=next(actor.parameters()).device
        rng=np.random.default_rng(seed)
        context=torch.zeros(CONTEXT);context[0]=1.
        if randomize:
            context[0]=float(rng.uniform(.3,1.2))
            context[1:30]=torch.tensor(rng.uniform(-.01,.01,29))
            context[30:33]=torch.tensor(rng.uniform([-.025,-.05,-.05],[.025,.05,.05]))
        self.reset(clip,start,context)
        state=self.state(np.zeros(29))
        obs=proprio(state,self.task.nominal,context,sensor_noise)
        history=obs.repeat(self.history,1)
        states, histories, observations, actions, terminals, logprobs = [state],[history],[obs],[],[],[]
        planned=self.lengths[clip]-start-1
        if duration is not None:
            planned=min(planned,round(duration/.02))
        if planned < 1:
            raise ValueError("Reset must leave at least one physical transition")
        total=0.; minimum_depth=0.; saturated_steps=0
        for t in range(planned):
            frame=start+t
            mean=actor(history[None].to(device),self.futures[clip,frame:frame+1].to(device))[0].cpu()
            if policy_std is not None:
                distribution=torch.distributions.Normal(mean,policy_std.cpu())
                action=distribution.sample()
                logprobs.append(distribution.log_prob(action).sum())
            else:
                action=mean+torch.tensor(rng.normal(0,action_noise,29),dtype=torch.float32)
            if not action.isfinite().all():
                raise FloatingPointError("Nonfinite action")
            target=self.nominal+self.scale*action.numpy()
            saturated=False
            for _ in range(4):
                measured=self.data.qpos[7:]+context[1:30].numpy()
                torque=self.kp*(target-measured)-self.kd*self.data.qvel[6:]
                saturated |= bool((abs(torque)>=.95*self.effort).any())
                self.data.ctrl[:]=np.clip(torque,-self.effort,self.effort)
                mujoco.mj_step(self.model,self.data)
            mujoco.mj_forward(self.model,self.data)
            nxt=self.state(action.numpy())
            if not nxt.isfinite().all() or nxt[7:71].abs().max()>1e5:
                raise FloatingPointError("Invalid physical transition")
            terminal=bool(self.task.termination(nxt,self.references[clip,frame+1],soft=False))
            saturated_steps+=int(saturated)
            for contact in self.data.contact:
                minimum_depth=min(minimum_depth,float(contact.dist))
            obs=proprio(nxt,self.task.nominal,context,sensor_noise)
            history=torch.cat((history[1:],obs[None]))
            actions.append(action);terminals.append(terminal);states.append(nxt);observations.append(obs)
            if record_history:
                histories.append(history)
            state=nxt
            if terminal:
                break
        states=torch.stack(states);actions=torch.stack(actions)
        frames=torch.arange(start,start+len(actions)+1)
        rewards=self.task.reward(states[:-1],states[1:],actions,self.references[clip,frames[1:]])
        total=float(rewards.double().sum())
        report=dict(seed=seed,clip=self.clip_names[clip],start_frame=start,steps=len(actions),
                    planned_steps=planned,terminated=terminal,truncated=not terminal,
                    tracking_return=total,normalized_return=total/(5*.02*planned),
                    completed=not terminal,min_contact_distance_m=minimum_depth,
                    torque_saturated_control_steps=saturated_steps,randomized=randomize,
                    action_noise=action_noise,sensor_noise=sensor_noise)
        result=dict(states=states,observations=torch.stack(observations),actions=actions,rewards=rewards,
                    terminals=torch.tensor(terminals,dtype=torch.float32),context=context,clip=clip,
                    frames=frames,report=report)
        if record_history:
            result["histories"]=torch.stack(histories)
        if logprobs:
            result["logprobs"]=torch.stack(logprobs)
        return result


class Replay:
    """Episode-safe model windows and independent imagination starts.

    Compact noisy observations reconstruct exactly the policy's rolling history.
    Transition FIFO capacity is independent of episode length and termination.
    """
    def __init__(self,capacity=128,capacity_transitions=None,history=10):
        self.episodes=[];self.capacity=capacity
        self.capacity_transitions=capacity_transitions;self.history=history
        self._packed={}

    def __len__(self):
        return sum(len(e["actions"]) for e in self.episodes)

    def _history(self,e,start):
        sequence=torch.cat((e["prefix"],e["observations"][:start+1]))
        return sequence[-self.history:]

    def add(self,episode):
        if len(episode["states"])!=len(episode["actions"])+1 or episode["terminals"][:-1].any():
            raise ValueError("Invalid episode boundaries")
        e={k:v for k,v in episode.items() if k not in ("histories","report")}
        if "observations" not in e:
            e["observations"]=episode["histories"][:,-1].clone()
        e["prefix"]=(episode["histories"][0,:-1].clone() if "histories" in episode else
                     e["observations"][0].repeat(self.history-1,1))
        self.episodes.append(e)
        self._packed.clear()
        if self.capacity_transitions is None:
            self.episodes=self.episodes[-self.capacity:]
        else:
            while len(self)>self.capacity_transitions:
                excess=len(self)-self.capacity_transitions;e=self.episodes[0]
                if excess>=len(e["actions"]):
                    self.episodes.pop(0);continue
                prefix=self._history(e,excess)[:-1].clone()
                for key in ("states","observations","actions","terminals","frames","rewards","logprobs"):
                    if key in e:e[key]=e[key][excess:].clone()
                e["prefix"]=prefix

    def _pack(self,device):
        device=torch.device(device);key=str(device)
        if key not in self._packed:
            lengths=torch.tensor([len(e["actions"]) for e in self.episodes])
            offsets=lambda n:torch.cat((torch.zeros(1,dtype=torch.long),n.cumsum(0)[:-1])).to(device)
            pack=dict(state_offset=offsets(lengths+1),action_offset=offsets(lengths),
                      observation_offset=offsets(lengths+self.history))
            for name in ("states","actions","frames"):
                pack[name]=torch.cat([e[name] for e in self.episodes]).to(device)
            pack["observations"]=torch.cat([torch.cat((e["prefix"],e["observations"])) for e in self.episodes]).to(device)
            pack["context"]=torch.stack([e["context"] for e in self.episodes]).to(device)
            pack["clip"]=torch.tensor([e["clip"] for e in self.episodes],device=device)
            self._packed[key]=pack
        return self._packed[key]

    def sample(self,batch,horizon,device):
        counts=torch.tensor([max(0,len(e["actions"])-horizon+1) for e in self.episodes])
        if counts.sum()==0:
            raise ValueError(f"No {horizon}-step physical windows; increase collection or inspect early terminations")
        # Keep draws on CPU for reproducible serialized CPU RNG, then gather the
        # whole batch on its training device. No Python loop over 4096 seeds.
        ids=torch.multinomial(counts.float(),batch,replacement=True)
        starts=(torch.rand(batch)*counts[ids]).long().to(device);ids=ids.to(device)
        p=self._pack(device);state_index=p["state_offset"][ids,None]+starts[:,None]+torch.arange(horizon+1,device=device)
        action_index=p["action_offset"][ids,None]+starts[:,None]+torch.arange(horizon,device=device)
        history_index=p["observation_offset"][ids,None]+starts[:,None]+torch.arange(self.history,device=device)
        return (p["states"][state_index],p["actions"][action_index],p["context"][ids],
                p["observations"][history_index],p["frames"][state_index],p["clip"][ids])

    def sample_starts(self,batch,device):
        # Every logged transition origin is eligible, including short failures
        # and the final 31 origins of each episode. No real H=32 window needed.
        states,_,context,history,frames,clips=self.sample(batch,1,device)
        return states[:,0],context,history,frames[:,0],clips

    def state_dict(self):
        return dict(episodes=self.episodes,capacity=self.capacity,
                    capacity_transitions=self.capacity_transitions,history=self.history)

    def load_state_dict(self,state):
        if (state["capacity"],state["capacity_transitions"],state["history"])!=(self.capacity,self.capacity_transitions,self.history):
            raise ValueError("Replay definition changed")
        self.episodes=state["episodes"]
        self._packed.clear()
