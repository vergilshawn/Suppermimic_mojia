"""Prepare all 40 local G1 LAFAN1 motions; preserve every clip and source hash.

Local LAFAN retargeting README: 30 Hz, metres, root XYZW, joint radians.
This is a kinematic target library, never world-model transition supervision.
"""
import argparse
import json
from pathlib import Path
import shutil

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import torch

from .g1_env import G1Tracking, ROOT, sha
from .kernel import STATE, rotation


def read_csv(path):
    values=np.loadtxt(path,delimiter=",")
    if values.ndim!=2 or values.shape[1]!=36 or len(values)<4 or not np.isfinite(values).all():
        raise ValueError(f"Invalid G1 LAFAN motion: {path}")
    norm=np.linalg.norm(values[:,3:7],axis=1)
    if np.max(abs(norm-1))>.01:
        raise ValueError("Source quaternion is not normalized XYZW")
    times=np.arange(len(values))/30.;target=np.minimum(np.arange(0,times[-1]+1e-9,.02),times[-1])
    pose=np.column_stack([np.interp(target,times,v) for v in values[:,np.r_[0:3,7:36]].T])
    rot=Slerp(times,Rotation.from_quat(values[:,3:7]))(target)
    lo=np.maximum(np.arange(len(target))-1,0);hi=np.minimum(np.arange(len(target))+1,len(target)-1)
    world_w=(rot[hi]*rot[lo].inv()).as_rotvec()/((hi-lo)*.02)[:,None]
    states=np.zeros((len(target),STATE),np.float32)
    states[:,:3]=pose[:,:3];states[:,3:7]=rot.as_quat()[:,[3,0,1,2]]
    states[:,7:10]=np.gradient(pose[:,:3],.02,axis=0,edge_order=2)
    states[:,10:13]=rot.inv().apply(world_w)
    states[:,13:42]=pose[:,3:];states[:,42:71]=np.gradient(pose[:,3:],.02,axis=0,edge_order=2)
    return states,len(values)


def prepare(source,out,assets):
    source=Path(source).resolve();out=Path(out).resolve()
    files=sorted(source.glob("*.csv"))
    if len(files)!=40:
        raise ValueError(f"Expected complete 40-motion LAFAN1 benchmark, found {len(files)}")
    out.mkdir(parents=True,exist_ok=False);(out/"motions").mkdir()
    env=G1Tracking(assets,robot_only=True);torch.set_num_threads(2)
    expected=[side+"_"+joint+"_joint" for side in ("left","right") for joint in
              ("hip_pitch","hip_roll","hip_yaw","knee","ankle_pitch","ankle_roll")]
    expected += ["waist_yaw_joint","waist_roll_joint","waist_pitch_joint"]
    expected += [side+"_"+joint+"_joint" for side in ("left","right") for joint in
                 ("shoulder_pitch","shoulder_roll","shoulder_yaw","elbow","wrist_roll","wrist_pitch","wrist_yaw")]
    if env.control["joint_names"]!=expected:
        raise ValueError("Robot joint order differs from the local LAFAN1 README")
    shutil.copytree(Path(assets)/"robot",out/"robot")
    for name in ("control.json","MJLAB_LICENSE"):
        shutil.copy2(Path(assets)/name,out/name)
    # Preserve available source documentation/licenses without inventing provenance.
    documents={}
    canonical=source.parent/"LAFAN-For_g1_h1_h2/LAFAN1_Retargeting_Dataset"
    for parent in (source,source.parent,canonical):
        for name in ("README.md","LICENSE","LICENSE.md","LICENSE.txt"):
            p=parent/name
            if p.is_file():
                target=out/(parent.name+"_"+name);shutil.copy2(p,target)
                documents[str(p)]=sha(p)
    clips=[]
    with torch.no_grad():
        for path in files:
            state,n=read_csv(path);s=torch.from_numpy(state);parts=[]
            for chunk in s.split(2048):
                parts.append(tuple(x.numpy() for x in env.task.fk(chunk)))
            pos,mat,vel,omega=(np.concatenate([part[i] for part in parts]) for i in range(4))
            quat=Rotation.from_matrix(mat.reshape(-1,3,3)).as_quat().reshape(len(s),-1,4)[...,[3,0,1,2]]
            worst=0.
            for t in np.unique(np.linspace(0,len(s)-1,11,dtype=int)):
                d=mujoco.MjData(env.model);d.qpos[:7]=state[t,:7];d.qpos[7:]=state[t,13:42]
                mujoco.mj_forward(env.model,d)
                worst=max(worst,float(np.max(abs(d.xpos[1:]-pos[t]))))
            if worst>2e-4:raise ValueError(f"FK verification failed: {path}")
            dest=out/"motions"/(path.stem+".npz")
            np.savez_compressed(dest,fps=np.array([50.]),joint_names=np.array(env.control["joint_names"]),
                                body_names=np.array(env.control["body_names"]),joint_pos=state[:,13:42],
                                joint_vel=state[:,42:71],body_pos_w=pos,body_quat_w=quat.astype(np.float32),
                                body_lin_vel_w=vel,body_ang_vel_w=omega)
            violation=np.maximum(env.model.jnt_range[1:,0]-state[:,13:42],0)+np.maximum(state[:,13:42]-env.model.jnt_range[1:,1],0)
            clips.append(dict(file=str(dest.relative_to(out)),frames=len(s),source=str(path),source_sha256=sha(path),
                              source_frames=n,duration_s=(len(s)-1)*.02,fk_sample_max_error_m=worst,
                              max_joint_limit_violation_rad=float(violation.max()),minimum_body_origin_z_m=float(pos[:,:,2].min())))
            print(json.dumps(dict(clip=path.stem,frames=len(s))),flush=True)
    manifest=dict(schema=1,robot="G1 29DOF",source=str(source),source_documents=documents,
                  source_fps=30,target_fps=50,source_layout="root XYZ metres, quaternion XYZW, 29 joint radians in G1 MuJoCo DFS order",
                  joint_order_basis="local LAFAN retargeting README and G1 XML order; original SuperMimic dataset hash unavailable",
                  motion_scope="all 40 local G1 LAFAN1 motions; no reference filtering, clipping or height shifts",
                  official_supermimic_code=False,clips=clips,
                  source_duration_h=sum(e["source_frames"] for e in clips)/30/3600,
                  files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob("*")) if p.is_file()})
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(dict(output=str(out),clips=len(clips),source_duration_h=manifest["source_duration_h"])))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--out",type=Path,default=ROOT/"data/g1")
    p.add_argument("--assets",type=Path,default=ROOT/"assets/robots/g1")
    a=p.parse_args(argv);prepare(a.source,a.out,a.assets)

if __name__ == "__main__":
    main()
