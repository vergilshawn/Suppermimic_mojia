"""Isolated Mojia joint-interface learner. Nothing runs on import.

Production training is user-started and forbidden while the protected G1 job
reports running. Tiny CPU smoke is explicit and has hard limits. Formal paper
reproduction and hardware deployment remain empirical outcomes, not flags set
by finishing this loop. Offline evaluation keeps evaluation time separate.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time
import uuid
import subprocess
import sys
from .regression import RegressionGuard
import numpy as np
import torch
from ..robot_layout import RobotLayout
from .kernel import frozen, proprio, step, lambda_returns
from .interfaces import MotionVelocityActor, IndependentCritics, InterfaceTasks, latent_alignment_loss
from .dynamics import ConsistentDynamics, masked_fit_loss
from .replay import Replay
from .collector import VectorCollector

ROOT=Path(__file__).resolve().parents[2]
MIGRATION=ROOT
PROTECTED_REPORT=ROOT/'runs/unused.json'
SCHEMA='mojia_ndof_v6'


def default_config(smoke=False):
    c=dict(profile='mojia_v3_8_motion',schema=SCHEMA,assets=None,dof=23,
           interfaces='motion',velocity_mode='rl',history=10,future=10,latent=32,
           dynamics_widths=[2048,2048,2048],encoder_widths=[2048,1024,512,256],
           decoder_widths=[2048,1024,512,256],velocity_encoder_widths=[256,256],
           critic_widths=[256,128],fit_window=8,horizon=32,fit_steps=10,
           actor_steps=1,critic_epochs=5,critic_minibatches=4,wm_batch_size=2048,
           actor_batch_size=4096,simulator_environments=4096,replay_transitions=4096000,
           replay_storage='cpu',simulator_backend='mujoco_warp',update_to_data_ratio=.4,
           utd_unit='outer_updates_per_vector_control_step',learning_rate=.0002,
           lr_decay_iterations=250000,lr_minimum_fraction=.03,iterations=50000,
           randomize=True,curriculum_ema=.1,threads=1,matmul_precision='high',
           evaluation_interval=1000,validation_interval=100,checkpoint_interval=1000,
           filtering_deferred=True,paper_equivalent=False,smoke_test=False,
           actuator_delay_steps=0,observation_noise=True,max_episode_steps=500,
           clean_collection_fraction=.5,reset_state_noise=True,warmup_steps=64,wm_only_iterations=500,
           regression_patience=3,regression_return_drop=.15,regression_completion_drop=.2)
    if smoke:
        c.update(profile='mojia_cpu_smoke',smoke_test=True,interfaces='motion_velocity',
                 dynamics_widths=[16,16],encoder_widths=[16,16],decoder_widths=[16,16],
                 velocity_encoder_widths=[16,16],critic_widths=[16,16],fit_steps=1,
                 critic_epochs=1,critic_minibatches=1,wm_batch_size=2,actor_batch_size=2,
                 simulator_environments=2,replay_transitions=128,simulator_backend='mujoco_cpu',
                 update_to_data_ratio=.125,iterations=2,randomize=False,
                 validation_interval=0,checkpoint_interval=2,evaluation_interval=0,warmup_steps=8,wm_only_iterations=0)
    return c


def source_hashes():
    paths=sorted(Path(__file__).parent.glob('*.py'))+[
        ROOT/'robot_control/robot_layout.py', ROOT/'robot_control/contact_metrics.py']
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_config(c):
    if c['schema']!=SCHEMA:raise ValueError('Wrong isolated schema')
    if c['interfaces'] not in ('motion','motion_velocity'):raise ValueError('Unknown interfaces')
    if c['velocity_mode'] not in ('rl','distillation'):raise ValueError('Unknown velocity objective')
    if c.get('actuator_delay_steps',0)!=0:
        raise ValueError('Delay requires explicit actuator queue state; nominal zero-delay runtime only')
    if c['replay_storage']!='cpu':raise ValueError('Only CPU replay is implemented')
    if (c['history'],c['future'],c['latent'],c['fit_window'],c['horizon'],c['actor_steps'])!=(10,10,32,8,32,1):
        raise ValueError('Current recipe requires history10/future10/latent32/k8/H32/one actor update')
    for k in ('dof','simulator_environments','replay_transitions','fit_steps','wm_batch_size',
              'actor_batch_size','iterations','critic_epochs','critic_minibatches','checkpoint_interval'):
        if isinstance(c[k],bool) or int(c[k])!=c[k] or c[k]<1:raise ValueError(k+' must be a positive integer')
    if c['interfaces']=='motion_velocity' and (c['simulator_environments']%2 or c['actor_batch_size']%2):
        raise ValueError('Joint recipe requires even environment and actor batch counts')
    if c['update_to_data_ratio']<=0 or c['utd_unit']!='outer_updates_per_vector_control_step':
        raise ValueError('Declare the local UTD unit explicitly')
    period=1/c['update_to_data_ratio']
    if period<1:raise ValueError('UTD must be <=1')
    if c['iterations']>50000:raise ValueError('Mojia trial capped at 50000 iterations')
    if c['max_episode_steps']!=500:raise ValueError('Trial requires 500-step training timeouts')
    if c['replay_transitions']<c['simulator_environments']:raise ValueError('Replay smaller than vector worlds')
    if c['evaluation_interval']<0:raise ValueError('Invalid evaluation interval')
    if c['validation_interval']<0:raise ValueError('Invalid validation interval')
    if not c['smoke_test'] and not c.get('assets'):raise ValueError('Production requires a prepared Mojia reference dataset')
    for key in ('dynamics_widths','encoder_widths','decoder_widths','velocity_encoder_widths','critic_widths'):
        if not c[key] or any(not isinstance(x,int) or x<1 for x in c[key]):raise ValueError('Invalid network widths')
    return round(period)


def require_no_competing_training(proc_root=Path('/proc'), ignore_pid=None):
    """Read-only process check, including both isolated learners and G1 v3.

    A protected running report is checked separately because sandboxed /proc
    may hide host processes. This function never signals or changes a process.
    """
    protected_modules = {'scripts.reproduction.train_v3', 'scripts.reproduction.train',
                         'experiments.reproduction_v4.train',
                         'experiments.reproduction_v4.runtime.train_v3',
                         'migration.mojia_v3_8.runtime.train','robot_control.runtime.train'}
    ignore_pid = os.getpid() if ignore_pid is None else ignore_pid
    for path in Path(proc_root).glob('[0-9]*/cmdline'):
        if path.parent.name == str(ignore_pid):
            continue
        try:
            argv = path.read_bytes().decode(errors='replace').split('\0')
        except OSError:
            continue
        if '-m' not in argv:
            continue
        index = argv.index('-m') + 1
        module = argv[index] if index < len(argv) else ''
        if module in protected_modules and '--preflight' not in argv and '--help' not in argv:
            raise RuntimeError(f'Competing training process {path.parent.name}: {module}; no job was started')


def guard_resources(c,device,iterations=None):
    smoke=c.get('smoke_test',False)
    if smoke:
        cap=c['iterations'] if iterations is None else iterations
        if (str(device)!='cpu' or cap>2 or c['iterations']>2 or c['simulator_environments']>2
            or c['wm_batch_size']>4 or c['actor_batch_size']>4 or c['threads']!=1
            or c['fit_steps']>2 or c['critic_epochs']>1 or c['critic_minibatches']>1
            or c['replay_transitions']>512 or validate_config(c)>16
            or any(max(c[k])>64 for k in ('dynamics_widths','encoder_widths','decoder_widths','velocity_encoder_widths','critic_widths'))):
            raise ValueError('Smoke exceeds bounded CPU-only verification budget')
        if c['simulator_backend']!='mujoco_cpu':raise ValueError('Smoke must use CPU physics')
        return
    require_no_competing_training()
    if not str(device).startswith('cuda') or c['simulator_backend']!='mujoco_warp':
        raise ValueError('Production requires explicit CUDA learner and MuJoCo Warp')


def make_environment(c):
    from .environment import MojiaTracking
    assets=c.get('assets')
    if assets is not None:
        assets=Path(assets)
        if not assets.is_absolute():assets=ROOT/assets
    env=MojiaTracking(assets,history=c['history'],future=c['future'],synthetic_only=c.get('smoke_test',False))
    if env.layout.dof!=c['dof']:raise ValueError('Robot contract and configured DOF disagree')
    return env


def make_networks(env,c,device):
    layout=RobotLayout(c['dof'],c['history'],c['future'],c['latent'])
    actor=MotionVelocityActor(c['history'],c['future'],c['encoder_widths'],c['decoder_widths'],c['velocity_encoder_widths'],layout=layout).to(device)
    critics=IndependentCritics(c['history'],c['future'],c['critic_widths'],layout=layout).to(device)
    model=ConsistentDynamics(c['dynamics_widths'],layout=layout).to(device)
    tasks=InterfaceTasks(env.task,mode=c['velocity_mode'],torso_index=env.torso-1).to(device)
    return actor,critics,model,tasks


def cosine_multiplier(iteration,decay_steps=250000,minimum=.03):
    return minimum+(1-minimum)*.5*(1+math.cos(math.pi*min(iteration,decay_steps)/decay_steps))


def rng_state():
    state=np.random.get_state()
    return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
                numpy=dict(kind=state[0],keys=state[1].tolist(),pos=state[2],gauss=state[3],cached=state[4]),python=random.getstate())


def restore_rng(state):
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:torch.cuda.set_rng_state_all(state['cuda'])
    n=state['numpy'];np.random.set_state((n['kind'],np.array(n['keys'],dtype=np.uint32),n['pos'],n['gauss'],n['cached']))
    random.setstate(state['python'])


def atomic_save(payload,path):
    temporary=path.with_suffix(path.suffix+'.tmp');torch.save(payload,temporary);temporary.replace(path)


def atomic_json(payload,path):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def mixed_starts(replay,batch,c,device):
    if c['interfaces']=='motion':return replay.sample_starts(batch,device,interface=0)
    parts=[replay.sample_starts(batch//2,device,interface=i) for i in (0,1)]
    return {k:torch.cat([p[k] for p in parts]) for k in parts[0]}


def imagine(model,actor,critics,tasks,collector,start,c):
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
        action=actor(history,future,state,ids,command).clamp(-20,20)
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
        state=nxt;obs=proprio(state,tasks.nominal,context,c.get('observation_noise',True))
        clean=torch.arange(len(state),device=state.device)<int(len(state)*c['clean_collection_fraction'])
        obs=torch.where(clean[:,None],proprio(state,tasks.nominal,context,False),obs)
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
                if j>=len(chunks[ident]):continue
                idx=chunks[ident][j];net=critics.motion if ident==0 else critics.velocity
                err=(net(features[idx]).squeeze(-1)-targets[idx]).square()
                loss=loss+(err*mask[idx]).sum()/mask[idx].sum().clamp_min(1)
            loss=loss/len(data);step(loss,optimizer,critics.parameters());losses.append(float(loss.detach()))
    return sum(losses)/len(losses)


@torch.no_grad()
def replay_model_validation(model,replay,device,batch=64):
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
                joint_rmse_rad=float((p[valid,13:13+model.layout.dof]-truth[valid,13:13+model.layout.dof]).square().mean().sqrt()),
                normalized_groups={k:float(v[valid].mean()) for k,v in groups.items()})
    return result



def run(c,out,device,seed=42,resume=None,max_iterations=None):
    collection_steps=validate_config(c);guard_resources(c,device,max_iterations)
    out=Path(out).resolve()
    if not out.is_relative_to(MIGRATION):
        raise ValueError('Isolated runtime outputs must stay under ')
    if max_iterations is not None and max_iterations<1:raise ValueError('Positive iteration budget required')
    if out.exists() and resume is None:raise FileExistsError('Choose a fresh isolated output directory')
    resume_report=None
    if resume:
        if Path(resume).resolve().parent!=out or not (out/'report.json').is_file():
            raise ValueError('Resume must target the checkpoint original run directory; branching is not implicit')
        resume_report=json.loads((out/'report.json').read_text())
    torch.set_num_threads(c['threads']);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_float32_matmul_precision(c.get('matmul_precision','high'))
    is_cuda=str(device).startswith('cuda')
    if is_cuda and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    began_run=time.monotonic();env=make_environment(c)
    actor,critics,model,tasks=make_networks(env,c,device)
    from .physics import CPUVectorPhysics, WarpPhysics
    backend=WarpPhysics if is_cuda else CPUVectorPhysics
    physics=backend(env,c['simulator_environments'],device=device)
    collector=VectorCollector(env,physics,actor,critics,tasks,c,device)
    collector.dynamics=model
    replay=Replay(c['replay_transitions'],c['simulator_environments'],c['history'],seed,env.layout)
    opts=[torch.optim.Adam(model.parameters(),lr=c['learning_rate']),
          torch.optim.Adam(actor.parameters(),lr=c['learning_rate'],betas=(.9,.95)),
          torch.optim.Adam(critics.parameters(),lr=c['learning_rate'])]
    sources=source_hashes();start=0;physical_steps=0;stop=False;run_id=uuid.uuid4().hex
    worker=None;eval_handle=None;guard=RegressionGuard(c['regression_patience'],c['regression_return_drop'],c['regression_completion_drop']);seen=set()
    previous_seconds=0.;phase_totals={key:0. for key in ('collection','model_fit','actor','critic','validation','checkpoint')}
    def halt(signum,frame):
        nonlocal stop
        stop=True
    old_handlers={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)}
    for sig in old_handlers:signal.signal(sig,halt)
    try:
        if resume:
            cp=torch.load(resume,map_location='cpu',weights_only=True)
            if cp.get('schema')!=SCHEMA or cp['config']!=c or cp['source_sha256']!=sources or cp['contract']!=env.contract:
                raise ValueError('Resume requires identical isolated schema/config/source/robot/data contract')
            if not cp.get('run_id') or cp['run_id']!=resume_report.get('run_id') or resume_report.get('config')!=c:
                raise ValueError('Resume run identity/config mismatch; refusing to mix existing logs')
            run_id=cp['run_id']
            for module,key in ((actor,'actor'),(model,'model'),(critics,'critics')):module.load_state_dict(cp[key])
            for opt,state in zip(opts,cp['optimizers']):opt.load_state_dict(state)
            replay.load_state_dict(cp['replay']);collector.load_state_dict(cp['collector']);restore_rng(cp['rng'])
            start=cp['iteration']+1;physical_steps=cp['physical_steps']
            previous_seconds=cp.get('elapsed_seconds_before_checkpoint',0.)
            phase_totals.update(cp.get('phase_totals',{}))
        out.mkdir(parents=True,exist_ok=resume is not None);atomic_json(c,out/'config.json')
        evaluation=out/'evaluation';evaluation.mkdir(exist_ok=True)
        def check_evaluations():
            nonlocal stop
            for result_path in sorted(evaluation.glob('job_*.result.json')):
                if str(result_path) in seen:continue
                result=json.loads(result_path.read_text());seen.add(str(result_path))
                if result['status']!='complete':
                    stop=True;report['stop_reason']='evaluation_error';continue
                improved,regressed=guard.observe(result)
                report['last_evaluation']=result;report['regression_bad_count']=guard.bad_count
                if regressed:stop=True;report['stop_reason']='persistent_deterministic_physics_regression'
            if worker is not None and worker.poll() is not None:
                stop=True;report['stop_reason']='evaluation_worker_exited'
        if c['evaluation_interval']:
            (out/'evaluation_stop').unlink(missing_ok=True)
            eval_handle=(out/'evaluation_worker.log').open('a')
            worker=subprocess.Popen([sys.executable,'-m','robot_control.runtime.evaluation_worker','--run',str(out),'--device',str(device)],stdout=eval_handle,stderr=subprocess.STDOUT,cwd=ROOT)
            atomic_json({'pid':worker.pid},out/'evaluation_worker_pid.json')
        report=dict(schema=SCHEMA,pid=os.getpid(),run_id=run_id,status='running',config=c,source_sha256=sources,contract=env.contract,
                    paper_reproduced=False,hardware_deployable=False,smoke_test=c['smoke_test'],
                    filtering_deferred=c['filtering_deferred'],utd_unit=c['utd_unit'],
                    completed_iterations=start,physical_steps=physical_steps,
                    initialization_seconds=time.monotonic()-began_run,
                    phase_timing_note='wall time includes initialization/checkpoint; CUDA phase timers synchronized',
                    current_protected_training_untouched=True)
        def sync():
            if is_cuda:torch.cuda.synchronize(device)
        def save(iteration):
            learner=dict(schema=SCHEMA,run_id=run_id,iteration=iteration,config=c,source_sha256=sources,contract=env.contract,
                         actor=actor.state_dict(),model=model.state_dict(),critics=critics.state_dict(),seed=seed,
                         trained_interfaces=['motion','velocity'] if c['interfaces']=='motion_velocity' else ['motion'],
                         smoke_test=c['smoke_test'],deployable=False,paper_reproduced=False)
            atomic_save(learner,out/'learner.pt')
            atomic_save(dict(**learner,optimizers=[o.state_dict() for o in opts],collector=collector.state_dict(),
                             replay=replay.state_dict(),rng=rng_state(),physical_steps=physical_steps,
                             phase_totals=phase_totals,elapsed_seconds_before_checkpoint=previous_seconds+time.monotonic()-began_run),out/'latest.pt')
        limit=c['iterations'] if max_iterations is None else min(c['iterations'],start+max_iterations)
        if start>=limit:raise ValueError('Checkpoint already meets requested iteration limit')
        if resume is None:
            warm=collector.collect(c['warmup_steps'],replay,-1);physical_steps+=warm['physical_transitions']
        for iteration in range(start,limit):
            collection_steps=math.floor((iteration+1)/c['update_to_data_ratio'])-math.floor(iteration/c['update_to_data_ratio'])
            check_evaluations()
            if c['evaluation_interval'] and iteration>4000 and iteration-report.get('last_evaluation',{}).get('iteration',0)>4000:
                stop=True;report['stop_reason']='evaluation_lag_exceeds_4000_iterations'
            began=time.monotonic();phases={};sync()
            if is_cuda:torch.cuda.reset_peak_memory_stats(device)
            lr=c['learning_rate']*cosine_multiplier(iteration,c['lr_decay_iterations'],c['lr_minimum_fraction'])
            for opt in opts:
                for group in opt.param_groups:group['lr']=lr
            mark=time.monotonic();collection=collector.collect(collection_steps,replay,iteration);sync()
            phases['collection']=time.monotonic()-mark;physical_steps+=collection['physical_transitions']
            mark=time.monotonic()
            for _ in range(c['fit_steps']):
                batch=replay.sample_windows(c['wm_batch_size'],c['fit_window'],device)
                loss=masked_fit_loss(model,batch);model_norm=step(loss,opts[0],model.parameters())
            sync();phases['model_fit']=time.monotonic()-mark
            actor_loss=loss.new_zeros(());actor_norm=0.;critic_loss=0.;diag={};phases['actor']=phases['critic']=0.
            if iteration>=c['wm_only_iterations']:
                mark=time.monotonic();start_batch=mixed_starts(replay,c['actor_batch_size'],c,device)
                with frozen(model,critics):
                    actor_loss,data,diag=imagine(model,actor,critics,tasks,collector,start_batch,c)
                    actor_norm=step(actor_loss,opts[1],actor.parameters(),element_clip=True)
                sync();phases['actor']=time.monotonic()-mark
                mark=time.monotonic();critic_loss=fit_critics(critics,opts[2],data,c['critic_epochs'],c['critic_minibatches'])
                sync();phases['critic']=time.monotonic()-mark
            row=dict(iteration=iteration,learning_rate=lr,collection=collection,physical_steps=physical_steps,
                     model_loss=float(loss.detach()),model_grad_norm=model_norm,actor_loss=float(actor_loss.detach()),
                     actor_grad_norm=actor_norm,critic_loss=critic_loss,diagnostics=diag,replay_size=len(replay),
                     wm_optimizer_steps_per_vector_control_step=c['fit_steps']/collection_steps,
                     wm_windows_per_new_transition=c['fit_steps']*c['wm_batch_size']/collection['physical_transitions'])
            mark=time.monotonic()
            if c['validation_interval'] and (iteration+1)%c['validation_interval']==0:
                row['replay_model_validation']=replay_model_validation(model,replay,device)
                row['replay_model_validation_note']='Diagnostic on training replay; independent physics validation is required offline'
            sync();phases['validation']=time.monotonic()-mark
            report.update(completed_iterations=iteration+1,physical_steps=physical_steps)
            if stop:report['status']='stopped_guard' if report.get('stop_reason') else 'stopped_signal'
            elif iteration+1==limit:
                report['status']='smoke_complete' if c['smoke_test'] else ('verification_budget_complete' if max_iterations else 'completed')
            for key,value in phases.items():phase_totals[key]+=value
            mark=time.monotonic()
            if (iteration+1)%c['checkpoint_interval']==0 or report['status']!='running':save(iteration)
            if c['evaluation_interval'] and (iteration+1)%c['evaluation_interval']==0:
                if (iteration+1)%c['checkpoint_interval']!=0:save(iteration)
                full=evaluation/f'full_{iteration+1:08d}.pt';learner=evaluation/f'learner_{iteration+1:08d}.pt'
                os.link(out/'latest.pt',full);os.link(out/'learner.pt',learner)
                atomic_json(dict(iteration=iteration+1,learner=str(learner),full_checkpoint=str(full)),evaluation/f'job_{iteration+1:08d}.json')
            sync();phases['checkpoint']=time.monotonic()-mark;phase_totals['checkpoint']+=phases['checkpoint']
            row['phase_seconds']=phases;row['seconds']=time.monotonic()-began
            row['wall_seconds_since_run_start']=time.monotonic()-began_run
            if is_cuda:
                row['cuda_peak_allocated_bytes']=torch.cuda.max_memory_allocated(device)
                free,total=torch.cuda.mem_get_info(device);row['cuda_device_used_bytes']=total-free
            report.update(last_iteration=row,phase_totals=phase_totals,
                          wall_seconds_this_launch=time.monotonic()-began_run,
                          previous_launch_seconds_before_checkpoint=previous_seconds)
            atomic_json(report,out/'report.json')
            with (out/'iterations.jsonl').open('a') as handle:handle.write(json.dumps(row,allow_nan=False)+'\n')
            print(json.dumps(dict(iteration=iteration+1,status=report['status'],model_loss=row['model_loss'],
                                  actor_loss=row['actor_loss'],seconds=row['seconds'])),flush=True)
            if stop:break
        return report
    finally:
        if worker is not None:
            (out/'evaluation_stop').touch()
        if eval_handle is not None:eval_handle.close()
        for sig,handler in old_handlers.items():signal.signal(sig,handler)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path);parser.add_argument('--out',type=Path)
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--resume',type=Path);parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--smoke',action='store_true',help='Explicit <=2 iteration tiny CPU physics and learning check')
    parser.add_argument('--verification-iterations',type=int)
    args=parser.parse_args(argv)
    if args.smoke and args.config is not None:parser.error('--smoke owns its fixed tiny config; do not combine with --config')
    if args.config is None and not args.smoke:parser.error('--config required unless --smoke')
    c=default_config(True) if args.smoke else json.loads(args.config.read_text())
    device='cpu' if args.smoke else args.device;steps=validate_config(c)
    if args.preflight:
        layout=RobotLayout(c['dof'],c['history'],c['future'],c['latent'])
        protected=json.loads(PROTECTED_REPORT.read_text()).get('status') if PROTECTED_REPORT.exists() else 'not_found'
        print(json.dumps(dict(config=c,layout=layout.to_dict(),vector_steps_per_iteration=steps,
                              physical_transitions_per_iteration=steps*c['simulator_environments'],
                              replay_memory=Replay.estimate_bytes(c['replay_transitions'],c['simulator_environments'],c['history'],layout=layout),
                              protected_training_status=protected,production_blocked=False,process_conflict_checked_at_launch=True,
                              training_started=False,paper_reproduced=False),indent=2));return
    if args.out is None:parser.error('--out required for explicit execution')
    run(c,args.out,device,args.seed,args.resume,args.verification_iterations)


if __name__=='__main__':main()
