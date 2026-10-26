"""CPU MuJoCo worlds with batched actor/FK; no CUDA inference or simulator changes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import mujoco
import torch
from robot_control.runtime.train import make_environment
from robot_control.runtime.interfaces import MotionVelocityActor
from robot_control.runtime.kernel import proprio
from .evaluation_queue import atomic_json


def load(checkpoint):
    cp=torch.load(checkpoint,map_location='cpu',weights_only=True)
    c=cp['config'];env=make_environment(c)
    if cp['schema']!=5 or cp['contract']!=env.contract:raise ValueError('Evaluation contract mismatch')
    actor=MotionVelocityActor(10,10,c['encoder_widths'],c['decoder_widths'],c['velocity_encoder_widths'])
    actor.load_state_dict(cp['actor']);actor.eval()
    return cp,env,actor


@torch.inference_mode()
def evaluate(env,actor,clips=None,max_steps=None,physics_workers=4):
    clips=list(range(len(env.lengths))) if clips is None else list(clips)
    if not clips or len(set(clips))!=len(clips):raise ValueError('Nonempty unique clips required')
    n=len(clips);ctx=torch.zeros(n,33);ctx[:,0]=1
    states=[];data=[]
    # reset applies the identical nominal constants before any world is stepped.
    for clip in clips:
        env.reset(clip,0,ctx[0]);d=mujoco.MjData(env.model)
        d.qpos[:]=env.data.qpos;d.qvel[:]=env.data.qvel
        mujoco.mj_forward(env.model,d);data.append(d)
        states.append(env.state(np.zeros(29)))
    state=torch.stack(states)
    history=proprio(state,env.task.nominal,ctx,False)[:,None].expand(-1,10,-1).clone()
    alive=np.ones(n,dtype=bool);failed=np.zeros(n,dtype=bool)
    steps=np.zeros(n,dtype=int);returns=np.zeros(n);qerr=np.zeros(n);perr=np.zeros(n)
    lengths=np.array([env.lengths[c]-1 for c in clips])
    budget=int(lengths.max()) if max_steps is None else min(max_steps,int(lengths.max()))
    def advance(item):
        index,action=item;d=data[index];target=env.nominal+env.scale*action
        for _ in range(4):
            torque=env.kp*(target-d.qpos[7:])-env.kd*d.qvel[6:]
            d.ctrl[:]=np.clip(torque,-env.effort,env.effort);mujoco.mj_step(env.model,d)
        return np.concatenate((d.qpos[:7],d.qvel[:6],d.qpos[7:],d.qvel[6:],action))
    with ThreadPoolExecutor(max_workers=physics_workers) as pool:
        for f in range(budget):
            ids=np.flatnonzero(alive)
            if not len(ids):break
            future=torch.stack([env.references[clips[i],torch.arange(f+1,f+11)] for i in ids])
            action=actor(history[ids],future,state[ids])
            nxt=torch.tensor(np.stack(list(pool.map(advance,zip(ids,action.numpy())))),dtype=torch.float32)
            ref=torch.stack([env.references[clips[i],f+1] for i in ids])
            returns[ids]+=env.task.reward(state[ids],nxt,action,ref).numpy()
            qerr[ids]+=(nxt[:,13:42]-ref[:,13:42]).square().mean(-1).numpy()
            perr[ids]+=(nxt[:,:3]-ref[:,:3]).square().sum(-1).numpy()
            fail=env.task.termination(nxt,ref,False).numpy()
            state[ids]=nxt
            history[ids]=torch.cat((history[ids,1:],proprio(nxt,env.task.nominal,ctx[ids],False)[:,None]),1)
            steps[ids]+=1;failed[ids]|=fail
            alive[ids]=~fail & (steps[ids]<lengths[ids])
    motions=[dict(clip=env.clip_names[c],steps=int(steps[i]),completed=bool(not failed[i] and steps[i]==lengths[i]),
                  terminated=bool(failed[i]),budget_limited=bool(alive[i]),normalized_return=float(returns[i]/(.1*lengths[i])),
                  joint_rmse_rad=float(np.sqrt(qerr[i]/max(1,steps[i]))),
                  root_rms_distance_m=float(np.sqrt(perr[i]/max(1,steps[i])))) for i,c in enumerate(clips)]
    return dict(motions=motions,completion_rate=sum(m['completed'] for m in motions)/n,
                normalized_return=sum(m['normalized_return'] for m in motions)/n,
                full_library=clips==list(range(len(env.lengths))) and max_steps is None,
                device='cpu',physics='MuJoCo nominal; same clipped PD and reward as v3')


def parent_alive(pid):
    if pid is None:return True
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--parent-pid',type=int);p.add_argument('--once',action='store_true')
    p.add_argument('--device',default='cpu')
    args=p.parse_args();torch.set_num_threads(2)
    try:os.nice(10)
    except OSError:pass
    folder=args.run/'evaluation';folder.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(folder/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    backend=folder/'backend.json'
    if backend.exists() and json.loads(backend.read_text())['device']!=args.device:
        raise ValueError('Use a separate evaluation folder for a different physics backend')
    atomic_json(dict(device=args.device),backend)
    while True:
        jobs=[j for j in sorted(folder.glob('*.job.json')) if not j.with_suffix('.result.json').exists() and not j.with_suffix('.error.json').exists() and not j.with_suffix('.skipped.json').exists()]
        if jobs:
            # Coalesce stale pending jobs. No unbounded queue or concurrent models.
            for old in jobs[:-1]:atomic_json({'reason':'superseded by newer pending checkpoint'},old.with_suffix('.skipped.json'))
            job=jobs[-1];began=time.monotonic()
            try:
                meta=json.loads(job.read_text());cp_path=folder/meta['checkpoint']
                if hashlib.sha256(cp_path.read_bytes()).hexdigest()!=meta['sha256']:raise ValueError('Checkpoint hash mismatch')
                cp,env,actor=load(cp_path)
                if args.device.startswith('cuda'):
                    from .gpu_evaluation import evaluate as gpu_evaluate
                    result=gpu_evaluate(env,actor,device=args.device)
                else:result=evaluate(env,actor)
                record=dict(iteration=meta['iteration'],checkpoint=meta,seconds=time.monotonic()-began,evaluation=result)
                atomic_json(record,job.with_suffix('.result.json'))
                best=folder/'best.json'
                if result['full_library'] and (not best.exists() or result['normalized_return']>json.loads(best.read_text())['evaluation']['normalized_return']):atomic_json(record,best)
                print(json.dumps({'iteration':meta['iteration'],'return':result['normalized_return'],'seconds':record['seconds']}),flush=True)
                del cp,env,actor
                if args.device.startswith('cuda'):torch.cuda.empty_cache()
            except Exception as exc:
                atomic_json(dict(error=repr(exc)),job.with_suffix('.error.json'));print(repr(exc),flush=True)
        elif args.once or not parent_alive(args.parent_pid):break
        else:time.sleep(5)


if __name__=='__main__':main()
