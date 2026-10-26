"""Full-frame kinematic data audit; contact proximity is not a force label."""
import json
import hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor


def _intervals(mask, dt):
    import numpy as np
    edges=np.diff(np.r_[False,mask,False].astype(int))
    return [dict(start_s=round(a*dt,3),end_s=round(b*dt,3),frames=int(b-a))
            for a,b in zip(np.where(edges==1)[0],np.where(edges==-1)[0])]


def _clip(args):
    import numpy as np
    import mujoco
    from .runtime.environment import build_model, named_addresses
    from .runtime.prepare import validate_states
    folder,entry,control=args
    path=Path(folder)/entry['file'];dt=control['control_dt'];model=build_model(control);data=mujoco.MjData(model)
    qids,vids,_=named_addresses(model,control['joint_names']);floor=model.geom('floor').id
    with np.load(path,allow_pickle=False) as f:z={k:f[k] for k in f.files}
    s=z['states'];n=len(s)
    result=dict(clip=path.stem,frames=n,duration_s=(n-1)*dt,**validate_states(s,control))
    result['all_numeric_fields_finite']=all(np.isfinite(v).all() for v in z.values() if np.issubdtype(v.dtype,np.number))
    result['joint_order_matches']=z['joint_names'].tolist()==control['joint_names']
    result['quaternion_norm_max_error']=float(np.max(abs(np.linalg.norm(s[:,3:7],axis=-1)-1)))
    result['state_joint_array_max_error']=float(np.max(abs(s[:,13:36]-z['joint_pos'])))
    result['state_velocity_array_max_error']=float(np.max(abs(s[:,36:59]-z['joint_vel'])))
    qdot=np.gradient(s[:,13:36].astype(float),dt,axis=0,edge_order=1)
    rootdot=np.gradient(s[:,:3].astype(float),dt,axis=0,edge_order=1)
    result['joint_velocity_difference_max_rad_s']=float(np.max(abs(s[:,36:59]-qdot)))
    result['root_velocity_difference_max_m_s']=float(np.max(abs(s[:,7:10]-rootdot)))
    result['joint_acceleration_peak_rad_s2']=float(np.max(abs(np.gradient(s[:,36:59].astype(float),dt,axis=0,edge_order=1))))
    lift=z.get('root_height_correction_m',np.zeros(n));lift_v=np.gradient(lift.astype(float),dt,edge_order=1)
    result['floor_lift']=dict(max_m=float(lift.max()),worst_time_s=float(lift.argmax()*dt),p95_m=float(np.percentile(lift,95)),frames_over_5cm=int((lift>.05).sum()),frames_over_10cm=int((lift>.1).sum()),peak_added_vertical_speed_m_s=float(abs(lift_v).max()),over_10cm_intervals=_intervals(lift>.1,dt))
    collidable=[i for i in range(model.ngeom) if model.geom_contype[i] or model.geom_conaffinity[i]]
    groups={}
    for side in ['left','right']:
        for kind,body in [('foot',side+'_ankle_roll'),('forearm',side+'_elbow_yaw')]:
            groups[side+'_'+kind]=[i for i in collidable if model.geom_bodyid[i]==model.body(body).id]
    near={k:np.zeros(n,bool) for k in groups};slide={k:np.zeros(n) for k in groups}
    self_depth=np.zeros(n);floor_depth=np.zeros(n);pairs={};fkmax=0.;fromto=np.zeros(6)
    for t,state in enumerate(s):
        data.qpos[:7]=state[:7];data.qpos[qids]=state[13:36]
        data.qvel[:6]=state[7:13];data.qvel[vids]=state[36:59]
        mujoco.mj_forward(model,data)
        fkmax=max(fkmax,float(abs(data.xpos[1:]-z['body_pos_w'][t]).max()))
        framepairs={}
        for contact in data.contact:
            a,b=int(contact.geom1),int(contact.geom2);depth=max(0.,-float(contact.dist))
            if floor in (a,b):floor_depth[t]=max(floor_depth[t],depth)
            elif depth>0:
                self_depth[t]=max(self_depth[t],depth)
                key=' / '.join(sorted([model.geom(a).name,model.geom(b).name]))
                framepairs[key]=max(framepairs.get(key,0),depth)
        for key,depth in framepairs.items():
            rec=pairs.setdefault(key,dict(frames=0,max_depth_m=0.,worst_time_s=0.))
            rec['frames']+=1
            if depth>rec['max_depth_m']:rec.update(max_depth_m=depth,worst_time_s=t*dt)
        for key,geoms in groups.items():
            distance=.051;point=None;body=None
            for geom in geoms:
                dist=mujoco.mj_geomDistance(model,data,floor,geom,.05,fromto)
                if dist<distance:distance=dist;point=fromto[3:].copy();body=int(model.geom_bodyid[geom])
            if distance<=.025:
                near[key][t]=True
                # Cached velocity is at body-frame origin, not COM.
                v=z['body_lin_vel_w'][t,body-1]+np.cross(z['body_ang_vel_w'][t,body-1],point-data.xpos[body])
                slide[key][t]=np.linalg.norm(v[:2])
    result['fk_body_position_max_error_m']=fkmax
    result['self_collision']=dict(max_depth_m=float(self_depth.max()),frames_over_1cm=int((self_depth>.01).sum()),frames_over_2cm=int((self_depth>.02).sum()),over_2cm_intervals=_intervals(self_depth>.02,dt),worst_time_s=float(self_depth.argmax()*dt),pairs=sorted([dict(pair=k,**v) for k,v in pairs.items()],key=lambda r:r['max_depth_m'],reverse=True))
    result['floor_penetration']=dict(max_depth_m=float(floor_depth.max()),frames_over_2mm=int((floor_depth>.002).sum()))
    result['near_floor_motion']={}
    for k in groups:
        v=slide[k][near[k]];bad=near[k]&(slide[k]>.2)
        result['near_floor_motion'][k]=dict(near_floor_frames=int(near[k].sum()),horizontal_speed_mean_m_s=float(v.mean()) if len(v) else None,horizontal_speed_p95_m_s=float(np.percentile(v,95)) if len(v) else None,frames_over_02m_s=int(bad.sum()),over_02m_s_intervals=_intervals(bad,dt))
    result['review_flags']=[]
    if result['floor_lift']['frames_over_10cm']:result['review_flags'].append('floor_lift_over_10cm')
    if result['self_collision']['frames_over_2cm']:result['review_flags'].append('self_penetration_over_2cm')
    if any(v['frames_over_02m_s'] for k,v in result['near_floor_motion'].items() if k.endswith('foot')):result['review_flags'].append('near_floor_foot_motion_over_02m_s')
    result['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def audit(config,out,workers=4):
    import os
    import html
    import numpy as np
    from .runtime.train import make_environment
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    c=json.loads(Path(config).read_text());env=make_environment(c);folder=env.assets
    manifest=json.loads((folder/'manifest.json').read_text())
    expected=json.loads(Path('configs/motions.json').read_text())
    names=[Path(e['file']).stem for e in manifest['clips']]
    os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
    args=[(str(folder),e,env.control) for e in manifest['clips']]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results=[]
        for r in pool.map(_clip,args):
            results.append(r);print(json.dumps(dict(checked=len(results),total=len(args),clip=r['clip'],flags=r['review_flags'])),flush=True)
    summary=dict(clips=len(results),frames=sum(r['frames'] for r in results),expected_clip_set_matches=set(names)==set(expected),duplicate_clips=len(names)-len(set(names)),all_manifest_hashes_checked=True,all_frames_checked=True,kinematically_eligible_clips=sum(r['kinematically_eligible'] for r in results),clips_with_self_penetration_over_2cm=sum(bool(r['self_collision']['frames_over_2cm']) for r in results),clips_with_floor_lift_over_10cm=sum(bool(r['floor_lift']['frames_over_10cm']) for r in results),clips_with_near_floor_foot_motion=sum('near_floor_foot_motion_over_02m_s' in r['review_flags'] for r in results),max_self_penetration_m=max(r['self_collision']['max_depth_m'] for r in results),max_floor_lift_m=max(r['floor_lift']['max_m'] for r in results),max_fk_position_error_m=max(r['fk_body_position_max_error_m'] for r in results))
    report=dict(config=str(config),assets=str(folder),contract=env.contract,summary=summary,scope='Every frame; MuJoCo FK/collision queries without stepping, trained policy, contact-force validation or torque feasibility. Near-floor <=2.5cm is a geometric proxy, not a planted-foot or load label. Forearm proximity uses the current model collidable forearm geometry; it does not imply an independently modeled hand.',review_thresholds='1/2cm self penetration, 10cm floor lift and 0.2m/s near-floor horizontal speed are review flags, not paper acceptance criteria or automatic filters.',motions=results)
    out.mkdir(parents=True);(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    rows=''.join('<tr><td>'+html.escape(r['clip'])+'</td><td>'+str(round(r['duration_s'],1))+'</td><td>'+str(round(r['floor_lift']['max_m']*100,1))+'</td><td>'+str(round(r['self_collision']['max_depth_m']*100,1))+'</td><td>'+str(r['self_collision']['frames_over_2cm'])+'</td><td>'+str(sum(v['frames_over_02m_s'] for k,v in r['near_floor_motion'].items() if k.endswith('foot')))+'</td></tr>' for r in results)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Mojia 数据全量检查</title><style>body{font:16px sans-serif;margin:30px}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:8px}th{position:sticky;top:0;background:#ddd}</style><h1>Mojia 数据全量检查</h1><p>全帧运动学检查；近地脚部运动不是受力滑动判定。未运行策略或验证力矩可行性。阈值仅用于人工复核，没有自动过滤数据。</p><a href="report.json">详细报告：时间段与碰撞部件</a><pre>'+html.escape(json.dumps(summary,indent=2))+'</pre><table><tr><th>动作</th><th>时长 s</th><th>最大抬升 cm</th><th>最大自穿透 cm</th><th>自穿透&gt;2cm 帧数</th><th>近地足部速度&gt;0.2m/s 足帧数</th></tr>'+rows+'</table>')
    print(json.dumps(summary,indent=2))
    return report
