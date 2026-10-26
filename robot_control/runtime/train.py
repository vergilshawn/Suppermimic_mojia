"""Revised G1 learner: GPU physics, host replay and motion/velocity interfaces.

Formal training is user-started. --preflight checks the recipe without learning.
The UTD unit is explicit: WM optimizer steps / vector control steps. The paper
publishes 0.4 without its implementation counting unit; equivalence is not claimed.
"""
import argparse
import json
import math
from pathlib import Path
import random
import signal
import time
import numpy as np
import torch

from .g1_env import G1Tracking, ROOT, sha
from .kernel import frozen, proprio, step, lambda_returns
from .interfaces import MotionVelocityActor, IndependentCritics, InterfaceTasks, latent_alignment_loss
from .dynamics import ConsistentDynamics, masked_fit_loss
from .replay import ReplayV3
from .collector import VectorCollector
from .training_utils import atomic_save, cosine_multiplier, rng_state, restore_rng


def make_environment(c):
    env=G1Tracking(ROOT/c['assets'])
    env.contract.update(backend='MuJoCo Warp v3; matching CPU evaluation',
                        goal='10 future q,dq,z,relative root rotation6D,root-frame v,w,heading-relative xyz (74/frame)',
                        learned_dynamics='159 local features -> 70 independent local state increments; running-normalized SiLU LayerNorm',
                        critic='per-interface, physical state + history + task goal + context + goal-process metadata')
    return env


def make_networks(env,c,device):
    actor=MotionVelocityActor(10,10,c['encoder_widths'],c['decoder_widths'],c['velocity_encoder_widths']).to(device)
    critics=IndependentCritics(10,10,c['critic_widths']).to(device)
    model=ConsistentDynamics(c['dynamics_widths']).to(device)
    tasks=InterfaceTasks(env.task,mode=c['velocity_mode'],torso_index=env.model.body('torso_link').id-1).to(device)
    return actor,critics,model,tasks


def validate_config(c):
    if c.get('schema')!=5:raise ValueError('Use the isolated schema 5 recipe; old checkpoints cannot be resumed')
    if c['interfaces'] != 'motion':raise ValueError('Unknown interface recipe')
    if c['velocity_mode'] not in ('rl','distillation'):raise ValueError('Unknown velocity objective')
    for k in ('simulator_environments','replay_transitions','fit_steps','wm_batch_size','actor_batch_size','iterations'):
        if c[k]<1:raise ValueError(f'{k} must be positive')
    if c['interfaces']=='motion_velocity' and (c['simulator_environments']%2 or c['actor_batch_size']%2):
        raise ValueError('Joint recipe requires even worlds/batch for exact half interfaces')
    if c['fit_window']!=8 or c['horizon']!=32:raise ValueError('k=8,H=32 required')
    if c['update_to_data_ratio']<=0 or c['utd_unit']!='outer_updates_per_vector_control_step':
        raise ValueError('Declare the implemented UTD unit explicitly')
    period=1/c['update_to_data_ratio']
    if period<1:raise ValueError('UTD greater than one is not supported in this recipe')
    if c['replay_transitions']<c['simulator_environments']:
        raise ValueError('Replay needs at least one transition per world')
    return period


def mixed_starts(replay,batch,c,device):
    if c['interfaces']=='motion':return replay.sample_starts(batch,device,interface=0)
    parts=[replay.sample_starts(batch//2,device,interface=i) for i in (0,1)]
    return {k:torch.cat([p[k] for p in parts]) for k in parts[0]}


def imagine_v3(model,actor,critics,tasks,collector,start,c):
    state=start['states'].detach();history=start['history'].detach();context=start['contexts']
    ids=start['interfaces'];clips=start['clips'];frame=start['frames']
    command=start['commands'].clone();ref_source=start['reference_commands'].bool().clone()
    elapsed=start.get('command_elapsed',torch.zeros_like(frame)).clone()
    survival=state.new_ones(len(state));total=state.new_zeros(len(state))
    rewards=[];dones=[];values=[];features={0:[],1:[]};masks=[];alignment=state.new_zeros(())
    lengths=collector.lengths[clips];reference_task=(ids==0) | (tasks.mode=='distillation')
    for t in range(c['horizon']):
        future=collector.future(clips,frame+t)
        with torch.no_grad():
            fm,fv=critics.input_features(history,future,state,context,command,
                                            goal_metadata=collector.goal_metadata(clips,frame+t,ref_source,elapsed))
            value=state.new_zeros(len(state))
            for ident,net,inp in ((0,critics.motion,fm),(1,critics.velocity,fv)):
                mask=ids==ident
                if mask.any():
                    features[ident].append(inp[mask].detach());value[mask]=net(inp[mask]).squeeze(-1)
        values.append(value)
        valid=~reference_task | (frame+t<lengths-1)
        masks.append(valid)
        action=actor(history,future,state,ids,command)
        if c['interfaces']=='motion_velocity' and tasks.mode=='rl':
            alignment=alignment+latent_alignment_loss(actor,history,future,state,command,
                                                       ref_source & valid,ids,weight=.1)/c['horizon']
        nxt=model(state,action,context)
        ref=collector.references[clips,frame+t+1]
        reward=tasks.reward(state,nxt,action,ref,ids,command)
        terminal=tasks.termination(nxt,ref,True,ids)
        at_end=reference_task & (frame+t+1>=lengths-1)
        terminal=torch.where(at_end,torch.ones_like(terminal),terminal)
        total=total+.99**t*survival*reward
        survival=survival*(1-terminal)
        rewards.append(reward);dones.append(terminal)
        state=nxt;obs=proprio(state,tasks.nominal,context,True)
        history=torch.cat((history[:,1:],obs[:,None]),1)
        elapsed+=1
        resample=elapsed>=250
        if tasks.mode=='rl':
            reference_draw=torch.rand(len(state),device=state.device)<.5
            ref_source=torch.where(resample,reference_draw,ref_source)
            low=command.new_tensor([-1.5,-1.32,-3.35]);high=command.new_tensor([2.,1.32,3.35])
            draw=low+(high-low)*torch.rand_like(command)
            draw=torch.where((torch.rand(len(state),device=state.device)<.1)[:,None],0.,draw)
            command=torch.where(resample[:,None],draw,command)
        elapsed=torch.where(resample,0,elapsed)
        command=torch.where(ref_source[:,None],collector.ref_commands(clips,frame+t+1),command)
    final=critics(history,collector.future(clips,frame+c['horizon']),state,context,ids,command,
                  goal_metadata=collector.goal_metadata(clips,frame+c['horizon'],ref_source,elapsed))
    values.append(final)
    total=total+.99**c['horizon']*survival*final
    targets=lambda_returns(rewards,dones,values).transpose(0,1).detach()
    mask=torch.stack(masks,1)
    data={}
    for ident in (0,1):
        rows=ids==ident
        if rows.any():data[ident]=(torch.stack(features[ident],1).flatten(0,1),targets[rows].flatten(),mask[rows].flatten())
    return -total.mean()+alignment,data,dict(alignment_loss=float(alignment.detach()))


def fit_critics(critics,optimizer,data,epochs=5,minibatches=4):
    losses=[]
    for _ in range(epochs):
        chunks={i:torch.randperm(len(v[1]),device=v[1].device).chunk(minibatches) for i,v in data.items()}
        for j in range(minibatches):
            loss=next(iter(data.values()))[0].new_zeros(())
            for ident,(features,targets,mask) in data.items():
                idx=chunks[ident][j];net=critics.motion if ident==0 else critics.velocity
                err=(net(features[idx]).squeeze(-1)-targets[idx]).square()
                loss=loss+(err*mask[idx]).sum()/mask[idx].sum().clamp_min(1)
            loss=loss/len(data);step(loss,optimizer,critics.parameters());losses.append(float(loss.detach()))
    return sum(losses)/len(losses)


@torch.no_grad()
def model_validation(model,replay,device,batch=64):
    result={}
    for h in (1,8,32):
        b=replay.sample_windows(batch,h,device);p=b['states'][:,0]
        for t in range(h):p=model(p,b['actions'][:,t],b['contexts'])
        # Complete horizons only; shortened windows count separately.
        valid=b['mask'].all(-1);count=int(valid.sum())
        result[str(h)]={'full_windows':count,'short_windows':batch-count}
        if count:
            truth=b['states'][:,-1];groups=model.error_groups(p,truth)
            result[str(h)].update(root_position_rmse_m=float((p[valid,:3]-truth[valid,:3]).square().mean().sqrt()),
                joint_rmse_rad=float((p[valid,13:42]-truth[valid,13:42]).square().mean().sqrt()),
                normalized_groups={k:float(v[valid].mean()) for k,v in groups.items()})
    return result


@torch.no_grad()
def physical_evaluation(env,actor,tasks,device,clips=None):
    # Independent CPU MuJoCo physical episodes; no training observations/noise/DR.
    from .interfaces import MOTION
    results=[]
    nominal=tasks.nominal.to(device)
    for clip in range(len(env.lengths)) if clips is None else clips:
        ctx=torch.zeros(1,33,device=device);ctx[:,0]=1
        env.reset(clip,0,ctx[0].cpu());s=env.state(np.zeros(29)).to(device)[None]
        history=proprio(s,nominal,ctx,False)[:,None].expand(-1,10,-1).clone();ret=0.;fail=False
        for f in range(env.lengths[clip]-1):
            future=env.references[clip,torch.arange(f+1,f+11)].to(device)[None]
            a=actor(history,future,s)[0]
            target=env.nominal+env.scale*a.cpu().numpy()
            import mujoco
            for _ in range(4):
                torque=env.kp*(target-env.data.qpos[7:])-env.kd*env.data.qvel[6:]
                env.data.ctrl[:]=np.clip(torque,-env.effort,env.effort);mujoco.mj_step(env.model,env.data)
            nxt=env.state(a.cpu().numpy()).to(device)[None]
            ref=env.references[clip,f+1].to(device)[None]
            ret+=float(tasks.base.reward(s,nxt,a[None],ref))
            fail=bool(tasks.base.termination(nxt,ref,False))
            s=nxt;history=torch.cat((history[:,1:],proprio(s,nominal,ctx,False)[:,None]),1)
            if fail:break
        results.append(dict(clip=env.clip_names[clip],steps=f+1,completed=not fail,normalized_return=ret/(.1*(env.lengths[clip]-1))))
    return dict(motions=results,completion_rate=sum(r['completed'] for r in results)/len(results),
                normalized_return=sum(r['normalized_return'] for r in results)/len(results))


def run(c,out,device,seed=42,resume=None,max_iterations=None):
    collection_steps=validate_config(c);torch.set_num_threads(c.get('threads',2))
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_float32_matmul_precision(c.get('matmul_precision','high'))
    if not torch.cuda.is_available() or not str(device).startswith('cuda'):
        raise RuntimeError('v3 requires an accessible CUDA GPU for real MuJoCo Warp physics')
    from .physics import WarpPhysics
    env=make_environment(c);actor,critics,model,tasks=make_networks(env,c,device)
    physics=WarpPhysics(env,c['simulator_environments'],device=device)
    collector=VectorCollector(env,physics,actor,critics,tasks,c,device)
    collector.dynamics=model
    replay=ReplayV3(c['replay_transitions'],c['simulator_environments'],seed=seed)
    opts=[torch.optim.Adam(model.parameters(),lr=c['learning_rate']),
          torch.optim.Adam(actor.parameters(),lr=c['learning_rate'],betas=(.9,.95)),
          torch.optim.Adam(critics.parameters(),lr=c['learning_rate'])]
    sources={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'robot_control/runtime').glob('*.py'))}
    start=0;physical_steps=0;best=-float('inf');stop=False
    def halt(signum,frame):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGINT,halt)
    if resume:
        cp=torch.load(resume,map_location='cpu',weights_only=True)
        if cp['schema']!=5 or cp['config']!=c or cp['source_sha256']!=sources or cp['contract']!=env.contract:
            raise ValueError('v3 resume needs identical schema/config/source/physics-data contract; old checkpoints are baselines only')
        for module,key in ((actor,'actor'),(model,'model'),(critics,'critics')):module.load_state_dict(cp[key])
        for opt,state in zip(opts,cp['optimizers']):opt.load_state_dict(state)
        replay.load_state_dict(cp['replay']);collector.load_state_dict(cp['collector']);restore_rng(cp['rng'])
        start=cp['iteration']+1;physical_steps=cp['physical_steps'];best=cp['best']
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps(c,indent=2)+'\n')
    report=dict(schema=5,status='running',config=c,source_sha256=sources,contract=env.contract,
                paper_reproduced=False,filtering_deferred=True,utd_unit=c['utd_unit'],
                official_state_goal_representation=False,completed_iterations=start,physical_steps=physical_steps)
    def save(iteration):
        learner=dict(schema=5,iteration=iteration,config=c,source_sha256=sources,contract=env.contract,
                     actor=actor.state_dict(),model=model.state_dict(),critics=critics.state_dict(),
                     seed=seed,deployable=False,paper_reproduced=False)
        atomic_save(learner,out/'learner.pt')
        atomic_save(dict(schema=5,iteration=iteration,config=c,source_sha256=sources,contract=env.contract,
                         actor=actor.state_dict(),model=model.state_dict(),critics=critics.state_dict(),
                         optimizers=[o.state_dict() for o in opts],collector=collector.state_dict(),
                         replay=replay.state_dict(),rng=rng_state(),physical_steps=physical_steps,best=best,seed=seed,
                         deployable=False,paper_reproduced=False),out/'latest.pt')
    limit=c['iterations'] if max_iterations is None else min(c['iterations'],start+max_iterations)
    if start>=limit:raise ValueError('Checkpoint already at configured iteration limit')
    if resume is None:
        warm=collector.collect(c['warmup_steps'],replay,-1)
        physical_steps+=warm['physical_transitions']
    for iteration in range(start,limit):
        collection_steps=math.floor((iteration+1)/c['update_to_data_ratio'])-math.floor(iteration/c['update_to_data_ratio'])
        began=time.monotonic();torch.cuda.reset_peak_memory_stats(device)
        lr=c['learning_rate']*cosine_multiplier(iteration,c['lr_decay_iterations'],c['lr_minimum_fraction'])
        for opt in opts:
            for group in opt.param_groups:group['lr']=lr
        collection=collector.collect(collection_steps,replay,iteration);physical_steps+=collection['physical_transitions']
        for _ in range(c['fit_steps']):
            b=replay.sample_windows(c['wm_batch_size'],8,device)
            loss=masked_fit_loss(model,b);model_norm=step(loss,opts[0],model.parameters())
        actor_loss=torch.zeros((),device=device);actor_norm=0.;critic_loss=0.
        diag={'phase':'world_model_only'}
        if iteration>=c['wm_only_iterations']:
            start_batch=mixed_starts(replay,c['actor_batch_size'],c,device)
            with frozen(model,critics):
                actor_loss,data,diag=imagine_v3(model,actor,critics,tasks,collector,start_batch,c)
                actor_norm=step(actor_loss,opts[1],actor.parameters(),element_clip=True)
            critic_loss=fit_critics(critics,opts[2],data,c['critic_epochs'],c['critic_minibatches'])
            diag['phase']='joint_learning'
        row=dict(iteration=iteration,learning_rate=lr,collection=collection,physical_steps=physical_steps,
                 model_loss=float(loss.detach()),model_grad_norm=model_norm,actor_loss=float(actor_loss.detach()),
                 actor_grad_norm=actor_norm,critic_loss=critic_loss,diagnostics=diag,replay_size=len(replay),
                 wm_optimizer_steps_per_vector_control_step=c['fit_steps']/collection_steps,
                 wm_windows_per_new_transition=c['fit_steps']*c['wm_batch_size']/collection['physical_transitions'])
        if (iteration+1)%c['validation_interval']==0:
            from .validation import independent_model_validation
            row['model_validation']=independent_model_validation(model,actor,critics,tasks,env,c,device)
        if (iteration+1)%c['evaluation_interval']==0:
            from .evaluation_queue import publish
            row['evaluation_request']=publish(out,iteration,c,env.contract,sources,actor)
        row['cuda_peak_allocated_bytes']=torch.cuda.max_memory_allocated(device)
        free,total=torch.cuda.mem_get_info(device)
        row['cuda_device_used_bytes']=total-free
        row['seconds']=time.monotonic()-began
        report.update(completed_iterations=iteration+1,physical_steps=physical_steps,last_iteration=row)
        if stop:report['status']='stopped_signal'
        elif iteration+1==limit:report['status']='verification_budget_complete' if max_iterations is not None else 'completed'
        (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        with (out/'iterations.jsonl').open('a') as handle:handle.write(json.dumps(row,allow_nan=False)+'\n')
        if (iteration+1)%c['checkpoint_interval']==0 or report['status']!='running':save(iteration)
        print(json.dumps(dict(iteration=iteration+1,status=report['status'],model_loss=row['model_loss'],actor_loss=row['actor_loss'],seconds=row['seconds'])),flush=True)
        if stop:break
    if (iteration+1)%c['evaluation_interval']:
        from .evaluation_queue import publish
        publish(out,iteration,c,env.contract,sources,actor)
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/motion.json')
    p.add_argument('--out',type=Path);p.add_argument('--device',default='cuda:0');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--eval-device',default='cuda:0')
    p.add_argument('--resume',type=Path);p.add_argument('--preflight',action='store_true')
    p.add_argument('--verification-iterations',type=int,help='Explicit bounded integration verification, not paper training')
    args=p.parse_args(argv);c=json.loads(args.config.read_text());n=validate_config(c)
    if args.preflight:
        print(json.dumps(dict(config=c,vector_steps_per_iteration=n,physical_transitions_per_iteration=n*c['simulator_environments'],
                              replay_memory_estimate=ReplayV3.estimate_bytes(c['replay_transitions'],c['simulator_environments']),
                              filtering_deferred=True,paper_reproduced=False,training_started=False),indent=2));return
    if args.out is None:p.error('--out required for execution')
    if args.verification_iterations is not None and args.verification_iterations<1:p.error('Positive verification budget required')
    if args.out.exists() and args.resume is None:p.error('New recipe requires a fresh output directory')
    import os,subprocess,sys
    args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=True)
    worker_env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    if args.eval_device=='cpu':worker_env['CUDA_VISIBLE_DEVICES']=''
    with (args.out/'evaluation_worker.log').open('a') as log:
        worker=subprocess.Popen([sys.executable,'-m','robot_control.runtime.evaluation_worker',
            '--run',str(args.out),'--parent-pid',str(os.getpid()),'--device',args.eval_device],
            cwd=ROOT,env=worker_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    run(c,args.out,args.device,args.seed,args.resume,args.verification_iterations)


if __name__=='__main__':main()
