"""Raw LAFAN BVH -> robot_retargeter link scaling and Mink IK -> Mojia CSV.
External projects are read-only. Outputs retain original clip timing and provenance.
"""
import argparse, hashlib, json, pickle, re, sys, os
from pathlib import Path
import numpy as np
import mujoco
import mink
import yaml
from scipy.spatial.transform import Rotation as R
RETARGET = Path(os.environ.get('ROBOT_RETARGETER','third_party/robot_retargeter')).resolve()
GMR = Path(os.environ.get('GMR_ROOT','third_party/GMR')).resolve()
sys.path.insert(0, str(RETARGET / 'scripts'))
sys.path.insert(0, str(GMR / 'general_motion_retargeting/utils'))
from lafan_vendor.extract import read_bvh
from lafan_vendor.utils import quat_fk
from smpl_replay import compute_robot_link_lengths, compute_link_geometry_from_positions, compute_link_scale_factors, apply_link_scales_to_positions, compute_leg_displacement_scale, append_robot_foot_keypoints
from robot_retarget import RobotRetarget
class LimitedRetarget(RobotRetarget):
    def setup_retarget_configuration(self):
        audit=json.loads((Path(__file__).resolve().parents[1]/'configs/robot.json').read_text())
        self.audit_joints=audit['joints']
        for j in self.audit_joints:
            i=self.model.joint(j['name']).id
            self.model.jnt_range[i]=[j['lower_rad'],j['upper_rad']]
            self.model.jnt_limited[i]=True
        super().setup_retarget_configuration()
    def update_targets(self,frame_idx):
        if frame_idx:
            for j in self.audit_joints:
                joint=self.model.joint(j['name']);previous=self.configuration.q[joint.qposadr[0]]
                delta=.95*j['urdf_velocity_rad_s']/self.fps
                self.model.jnt_range[joint.id]=[max(j['lower_rad'],previous-delta),min(j['upper_rad'],previous+delta)]
            self.ik_limits=[mink.ConfigurationLimit(self.model)]
        super().update_targets(frame_idx)

LINKS = {'left_hip': ('hips_mean','LeftUpLeg'), 'left_thigh': ('LeftUpLeg','LeftLeg'), 'left_calf': ('LeftLeg','LeftFoot'), 'right_hip': ('hips_mean','RightUpLeg'), 'right_thigh': ('RightUpLeg','RightLeg'), 'right_calf': ('RightLeg','RightFoot'), 'neck': ('hips_mean','shoulder_mean'), 'head': ('shoulder_mean','Head'), 'left_shoulder': ('shoulder_mean','LeftArm'), 'left_arm': ('LeftArm','LeftForeArm'), 'left_fore_arm': ('LeftForeArm','LeftHand'), 'right_shoulder': ('shoulder_mean','RightArm'), 'right_arm': ('RightArm','RightForeArm'), 'right_fore_arm': ('RightForeArm','RightHand')}
AXES = R.from_euler('z',90,degrees=True) * R.from_euler('x',90,degrees=True)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(path, limit=None):
    a=read_bvh(str(path));q,p=quat_fk(a.quats,a.pos,a.parents)
    p=AXES.apply(p.reshape(-1,3)).reshape(p.shape)/100
    q=(AXES*R.from_quat(q.reshape(-1,4)[:,[1,2,3,0]])).as_quat(scalar_first=True).reshape(q.shape)
    names=list(a.bones);slots={n:i for i,n in enumerate(names)}
    for name, pair, ori in [('hips_mean',('LeftUpLeg','RightUpLeg'),'Hips'),('shoulder_mean',('LeftArm','RightArm'),'Spine2')]:
        p=np.concatenate((p, p[:,[slots[n] for n in pair]].mean(1)[:,None]),1)
        q=np.concatenate((q,q[:,slots[ori]][:,None]),1);slots[name]=len(names);names.append(name)
    return p[:limit],q[:limit],slots

def run(source,out,limit=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    dest=out/(source.stem+'.csv')
    if dest.exists(): raise FileExistsError(dest)
    cfgpath=RETARGET/'config/robot/mojia_v3_8.yaml';cfg=yaml.safe_load(cfgpath.read_text());xml=RETARGET/cfg['robot_xml_path']
    model=mujoco.MjModel.from_xml_path(str(xml));data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    fps=1/float(re.search(r'Frame Time:\s*([\d.]+)',source.read_text()).group(1))
    p,q,slots=load(source,limit);tp,tq,ts=load(GMR/'ik_config_manager/TPOSE.bvh',1)
    lengths=compute_robot_link_lengths(cfgpath,xml)
    sl,vec,resolved=compute_link_geometry_from_positions(LINKS,slots,p)
    scales,_=compute_link_scale_factors(lengths,sl)
    scale,_,_=compute_leg_displacement_scale(lengths,vec,0.)
    root=p[:,slots['hips_mean']].copy();root[:,:2]-=root[0,:2];root*=scale
    p+= (root-p[:,slots['hips_mean']])[:,None]
    p,q=apply_link_scales_to_positions(p,resolved,slots,scales,vec,0.,('LeftUpLeg','LeftLeg','LeftFoot'),('RightUpLeg','RightLeg','RightFoot'),quaternions=q)
    names=['hips_mean',*cfg['robot_links']];positions=[];quats=[]
    for name in names:
        bone='hips_mean' if name=='hips_mean' else LINKS[name][1]
        body='hips_sphere' if name=='hips_mean' else cfg['robot_links'][name][1]
        # Calibrate BVH local bone axes against the supplied T-pose, not against frame zero of each clip.
        ref=R.from_quat(tq[0,ts[bone]],scalar_first=True)
        target=R.from_quat(data.xquat[model.body(body).id],scalar_first=True)
        orient=R.from_quat(q[:,slots[bone]],scalar_first=True)*ref.inv()*target
        positions.append(p[:,slots[bone]]);quats.append(orient.as_quat(scalar_first=True))
    positions=np.stack(positions,1);quats=np.stack(quats,1)
    positions,quats,extra=append_robot_foot_keypoints(positions,quats,cfg['robot_links'],cfgpath,xml);names+=extra
    contacts=cfg['contact_links'];source_contacts=['LeftFoot','LeftToe','RightFoot','RightToe','LeftHand','RightHand']
    raw,_,rawslots=load(source,limit)
    contact_pos=np.stack([raw[:,rawslots[n]] for n in source_contacts],1)
    speed=np.linalg.norm(np.gradient(contact_pos,1/fps,axis=0),axis=-1)
    states=(contact_pos[...,2]<.08)&(speed<.5)
    # Match source floor contacts to target soles while retaining jumps and root trajectory.
    index={n:i for i,n in enumerate(names)}
    target_ids=[index.get(n,index['left_fore_arm' if n.startswith('left') else 'right_fore_arm']) for n in contacts]
    correction=np.zeros(len(p))
    for t in range(len(p)):
        active=states[t]
        if active.any(): correction[t]=np.median(positions[t,np.asarray(target_ids)[active],2])
    positions[:,:,2]-=correction[:,None]
    payload=dict(keypoint_names=names,positions=positions,quaternions=quats,contact_names=contacts,contact_states=states,fps=fps)
    kp=out/(source.stem+'.keypoints.pkl');kp.write_bytes(pickle.dumps(payload))
    solver=LimitedRetarget(str(xml),str(kp),cfg['ik_match_table'],contact_body_names=contacts,contact_position_cost=15.)
    solver.retarget();solver.save_results_as_csv(str(dest))
    joint_names=[model.joint(i).name for i in range(model.njnt) if model.jnt_type[i]==mujoco.mjtJoint.mjJNT_HINGE]
    (out/'joint_order.json').write_text(json.dumps(joint_names,indent=2))
    record=dict(source=str(source),source_sha256=sha(source),output_sha256=sha(dest),frames=len(p),fps=fps,raw_human_bvh=True,root_displacement_scale=scale,axes='Y-up to Z-up then +90deg yaw; TPOSE calibrated body orientations',target_xml=str(xml),target_xml_sha256=sha(xml),retargeter_sha256=sha(RETARGET/'scripts/robot_retarget.py'),physical_tracking_validated=False,limits_from_training_audit=True,per_frame_velocity_bounds=True)
    (out/(source.stem+'.json')).write_text(json.dumps(record,indent=2));return record
