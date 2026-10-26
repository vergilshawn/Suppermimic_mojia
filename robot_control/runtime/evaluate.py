"""Offline CPU physics evaluation with explicit metrics and independent model tests.

No evaluation is silently inserted into the learner's wall-time. Full evaluation
is blocked while the protected G1 job runs. --smoke permits only a smoke learner,
one clip and at most 16 controls, and cannot certify policy performance.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from .kernel import proprio, rotation, spans
from .interfaces import heading_rotation
from .physics import CPUVectorPhysics
from ..contact_metrics import FootContactMetrics
from .train import (SCHEMA,MIGRATION,PROTECTED_REPORT,make_environment,make_networks,
                    source_hashes,atomic_json)


def evaluation_guard(smoke,config,max_steps,clips):
    if smoke:
        if not config.get('smoke_test') or max_steps is None or max_steps>16 or max_steps<1 or len(clips)>1:
            raise ValueError('Protected-run evaluation exception requires smoke checkpoint, one clip, <=16 controls')
        return
    if PROTECTED_REPORT.exists() and json.loads(PROTECTED_REPORT.read_text()).get('status')=='running':
        raise RuntimeError('Full offline evaluation is blocked while the protected G1 job is running')


@torch.no_grad()
def physical_evaluation(env,actor,tasks,clips=None,max_steps=None,interface='motion',command=(0.,0.,0.)):
    if interface not in ('motion','velocity'):raise ValueError('Unknown evaluation interface')
    actor.cpu();tasks.cpu();physics=CPUVectorPhysics(env,1,device='cpu')
    clips=list(range(len(env.lengths))) if clips is None else list(clips)
    results=[];q,dq,last=spans(env.references[0,0])
    for clip in clips:
        if clip<0 or clip>=len(env.lengths):raise ValueError('Invalid clip')
        context=torch.zeros(1,env.layout.context);context[:,0]=1
        state=env.references[clip,0][None].clone();state[:,last]=0
        physics.reset(torch.tensor([0]),state,context)
        contact_metrics=FootContactMetrics(physics.models[0],control_dt_s=env.control['control_dt'])
        history=proprio(state,tasks.nominal,context)[:,None].expand(-1,env.layout.history,-1).clone()
        ret=0.;fail=False;saturated=0;root_error=[];joint_error=[];body_error=[];yaw_error=[];limit_steps=0
        budget=env.lengths[clip]-1 if max_steps is None else min(max_steps,env.lengths[clip]-1)
        if budget<1:raise ValueError('Clip/evaluation budget must contain a transition')
        ids=torch.tensor([0 if interface=='motion' else 1]);cmd=torch.tensor(command,dtype=torch.float32)[None]
        for frame in range(budget):
            future=env.references[clip,torch.arange(frame,frame+env.layout.future)][None]
            action=actor(history,future,state,ids,cmd)
            nxt,diag=physics.step(action);ref=env.references[clip,frame+1][None]
            # CPUVectorPhysics already mj_forward's the endpoint. These queries
            # only read solver contacts/forces and do not advance or refit physics.
            contact_metrics.sample(physics.datas[0])
            ret+=float(tasks.reward(state,nxt,action,ref,ids,cmd));fail=bool(tasks.termination(nxt,ref,False,ids))
            saturated+=int(diag['torque_saturated'].sum())
            root_error.append(float((nxt[:,:3]-ref[:,:3]).square().sum()))
            joint_error.append(float((nxt[:,q]-ref[:,q]).square().mean()))
            p=tasks.fk(nxt)[0];rp=tasks.fk(ref)[0]
            body_error.append(float((p[:,tasks.base.tracked]-rp[:,tasks.base.tracked]).square().sum(-1).mean()))
            heading=heading_rotation(nxt);target=heading_rotation(ref)
            cosine=(heading[...,:2,0]*target[...,:2,0]).sum(-1).clamp(-1,1)
            yaw_error.append(float(torch.acos(cosine).square()))
            limit_steps+=int(((nxt[:,q]<tasks.base.lower)|(nxt[:,q]>tasks.base.upper)).any())
            state=nxt;history=torch.cat((history[:,1:],proprio(state,tasks.nominal,context)[:,None]),1)
            if fail:break
        steps=frame+1;completed=not fail and steps==env.lengths[clip]-1
        results.append(dict(clip=env.clip_names[clip],interface=interface,steps=steps,completed=completed,fall=fail,
                            budget_limited=not fail and steps<env.lengths[clip]-1,
                            normalized_return=ret/(5*env.control['control_dt']*(env.lengths[clip]-1)) if interface=='motion' else None,
                            episode_reward=ret,root_position_rmse_m=float(np.sqrt(np.mean(root_error))),
                            joint_rmse_rad=float(np.sqrt(np.mean(joint_error))),root_yaw_rmse_rad=float(np.sqrt(np.mean(yaw_error))),
                            tracked_body_position_rmse_m=float(np.sqrt(np.mean(body_error))),
                            torque_saturated_control_fraction=saturated/steps,soft_limit_control_fraction=limit_steps/steps,
                            physical_foot_contacts=contact_metrics.summary()))
    return dict(motions=results,interface=interface,completion_rate=sum(r['completed'] for r in results)/len(results),
                failures=sum(r['fall'] for r in results),
                normalized_return=float(np.mean([r['normalized_return'] for r in results])) if interface=='motion' else None,
                evaluation_is_full=not any(r['budget_limited'] for r in results),
                completion_definition='Reference-length execution without height/tilt termination; tracking errors are separate metrics',
                foot_slip_contact_force_metrics_available=True,
                foot_contact_metric_scope='Actual MuJoCo contacts sampled at control-step endpoints; per-motion left/right load, loaded contact-point slip and penetration. Sample maxima exclude substep peaks; no impulse integration.')


from .validation import independent_model_validation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--smoke',action='store_true');p.add_argument('--max-steps',type=int)
    p.add_argument('--clips',type=int,nargs='+');p.add_argument('--interface',choices=('motion','velocity'),default='motion')
    p.add_argument('--command',type=float,nargs=3,default=(0.,0.,0.));p.add_argument('--model-validation',action='store_true')
    p.add_argument('--validation-samples',type=int,default=8)
    args=p.parse_args();torch.set_num_threads(1);began=time.monotonic()
    if args.checkpoint.stat().st_size>512*1024*1024:raise ValueError('Use small learner.pt; loading full replay checkpoints is unnecessary')
    cp=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if cp.get('schema')!=SCHEMA or cp['source_sha256']!=source_hashes():raise ValueError('Checkpoint/source schema mismatch')
    c=cp['config'];clips=args.clips or ([0] if args.smoke else [])
    evaluation_guard(args.smoke,c,args.max_steps,clips)
    if args.model_validation and args.smoke:raise ValueError('Independent model validation is a separate post-training workload, not the protected-run smoke exception')
    if args.interface not in cp['trained_interfaces']:raise ValueError('Requested interface was never trained')
    env=make_environment(c)
    if cp['contract']!=env.contract:raise ValueError('Robot/data/physics contract differs from checkpoint')
    actor,critics,model,tasks=make_networks(env,c,'cpu')
    actor.load_state_dict(cp['actor']);model.load_state_dict(cp['model'])
    result=physical_evaluation(env,actor,tasks,clips or None,args.max_steps,args.interface,args.command)
    if args.model_validation:result['world_model']=independent_model_validation(env,model,actor,args.validation_samples)
    result.update(checkpoint=str(args.checkpoint),iteration=cp['iteration'],seconds=time.monotonic()-began,
                  smoke_test=c['smoke_test'],paper_reproduced=False,hardware_deployable=False)
    out=args.out.resolve()
    if not out.is_relative_to(MIGRATION) or out.exists():raise ValueError('Evaluation output must be a fresh isolated path')
    out.parent.mkdir(parents=True,exist_ok=True);atomic_json(result,out)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
