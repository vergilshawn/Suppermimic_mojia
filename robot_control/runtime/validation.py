"""Fresh-policy physical validation and group-wise dynamics Jacobian diagnostics."""
import numpy as np
import torch
from .kernel import Q,DQ


def physical_jacobians(model,env,replay,device,count=2,epsilon=.01):
    """One-step action sensitivities; same reset state for every CPU perturbation.

    These are local dynamics derivatives, not full closed-loop policy gradients.
    Contact-scale sensitivity should be checked across more epsilons in audits.
    """
    import mujoco
    b=replay.sample_starts(count,device);rows=[]
    for i in range(count):
        s=b['states'][i];ctx=b['contexts'][i];a=b['actions'][i]
        # Independent CPU simulator: no collector state is modified.
        env.reset(int(b['clips'][i]),int(b['frames'][i]),ctx.cpu())
        def advance(action):
            data=mujoco.MjData(env.model)
            data.qpos[:7]=s[:7].cpu().numpy();data.qpos[7:]=s[13:42].cpu().numpy()
            data.qvel[:6]=s[7:13].cpu().numpy();data.qvel[6:]=s[42:71].cpu().numpy()
            mujoco.mj_forward(env.model,data)
            target=env.nominal+env.scale*action
            for _ in range(4):
                torque=env.kp*(target-data.qpos[7:]-ctx[1:30].cpu().numpy())-env.kd*data.qvel[6:]
                data.ctrl[:]=np.clip(torque,-env.effort,env.effort);mujoco.mj_step(env.model,data)
            return np.r_[data.qpos[:7],data.qvel[:6],data.qpos[7:],data.qvel[6:],action]
        physical=[];an=a.detach().cpu().numpy()
        for j in range(29):
            perturb=np.zeros(29);perturb[j]=epsilon
            physical.append((advance(an+perturb)-advance(an-perturb))/(2*epsilon))
        physical=torch.tensor(np.stack(physical,axis=-1),device=device,dtype=s.dtype)
        with torch.enable_grad():
            learned=torch.autograd.functional.jacobian(lambda action:model(s[None],action[None],ctx[None])[0],a)
        row={}
        for name,sl in [('root_position',slice(0,3)),('root_velocity',slice(7,10)),('joint_position',Q),('joint_velocity',DQ)]:
            x,y=learned[sl].flatten(),physical[sl].flatten()
            denom=x.norm()*y.norm()
            row[name]=dict(cosine=float((x@y)/denom.clamp_min(1e-12)),
                           relative_error=float((x-y).norm()/y.norm().clamp_min(1e-12)),
                           physical_norm=float(y.norm()))
        rows.append(row)
    return dict(scope='local one-control-step, qpos/qvel reset, fixed physical context',epsilon=epsilon,samples=rows)


def independent_model_validation(model,actor,critics,tasks,env,config,device):
    from .physics import WarpPhysics
    from .collector import VectorCollector
    from .replay import ReplayV3
    from .train import model_validation
    # Keep training random streams unchanged. Validation has its own worlds/replay.
    with torch.random.fork_rng(devices=[torch.device(device).index or 0]):
        torch.manual_seed(7911)
        c=dict(config,simulator_environments=32,replay_transitions=32*64,update_normalizers=False)
        p=WarpPhysics(env,32,device=device)
        collector=VectorCollector(env,p,actor,critics,tasks,c,device)
        replay=ReplayV3(32*64,32,seed=7911)
        collection=collector.collect(64,replay,policy_version=-1)
        with torch.no_grad():report=model_validation(model,replay,device,batch=128)
        report['collection']=collection;report['scope']='fresh current-policy physics; never added to training replay'
        report['jacobians']=physical_jacobians(model,env,replay,device)
    return report


@torch.no_grad()
def velocity_step_test(env,actor,tasks,device,commands=None,trials=4,duration=5.):
    """Appendix G grid/metric; standing initialization perturbation is local.

    Default: 3 planes x (15x15 minus zero)=672 commanded steps, four trials.
    Initial joints have seed-fixed +/-0.005rad perturbations; no sensor noise/DR.
    """
    import mujoco
    from .interfaces import heading_rotation
    from .kernel import apply,rotation,proprio
    if commands is None:
        x=np.linspace(-1.5,2.,15);y=np.linspace(-1.32,1.32,15);w=np.linspace(-3.35,3.35,15)
        commands=([ [a,b,0.] for a in x for b in y if abs(a)+abs(b)>1e-8 ]+
                  [ [a,0.,b] for a in x for b in w if abs(a)+abs(b)>1e-8 ]+
                  [ [0.,a,b] for a in y for b in w if abs(a)+abs(b)>1e-8 ])
    rows=[];steps=round(duration/.02)
    for ci,cmd in enumerate(commands):
        command=torch.tensor(cmd,device=device,dtype=torch.float32)[None]
        trial_rows=[]
        for trial in range(trials):
            ctx=torch.zeros(1,33,device=device);ctx[:,0]=1
            env.reset(0,0,ctx[0].cpu());env.data.qpos[:3]=[0,0,.78];env.data.qpos[3:7]=[1,0,0,0]
            env.data.qpos[7:]=env.nominal+np.random.default_rng(910001+trial).uniform(-.005,.005,29)
            env.data.qvel[:]=0;mujoco.mj_forward(env.model,env.data)
            s=torch.tensor(np.r_[env.data.qpos[:7],env.data.qvel[:6],env.data.qpos[7:],env.data.qvel[6:],np.zeros(29)],device=device,dtype=torch.float32)[None]
            history=proprio(s,tasks.nominal,ctx,False)[:,None].expand(-1,10,-1).clone();velocities=[];fell=False
            for t in range(steps):
                future=env.references[0,torch.arange(10)].to(device)[None]
                action=actor(history,future,s,torch.ones(1,device=device,dtype=torch.long),command)
                target=env.nominal+env.scale*action[0].cpu().numpy()
                for _ in range(4):
                    env.data.ctrl[:]=np.clip(env.kp*(target-env.data.qpos[7:])-env.kd*env.data.qvel[6:],-env.effort,env.effort)
                    mujoco.mj_step(env.model,env.data)
                s=env.state(action[0].cpu().numpy()).to(device)[None]
                local=apply(heading_rotation(s).transpose(-1,-2),s[:,7:10]);worldw=apply(rotation(s[:,3:7]),s[:,10:13])
                velocities.append(torch.cat((local[:,:2],worldw[:,2:3]),-1)[0])
                # Always the velocity task's fall-only criterion, even when evaluating a distillation encoder.
                gravity=-rotation(s[:,3:7])[:,2,2]
                fell=bool((s[:,2]<.35)|(1+gravity>.7))
                history=torch.cat((history[:,1:],proprio(s,tasks.nominal,ctx,False)[:,None]),1)
                if fell:break
            norm=command.norm().clamp_min(1e-8)
            error=float((torch.stack(velocities)[-min(100,len(velocities)):].mean(0)-command[0]).norm()/norm) if not fell else None
            trial_rows.append(dict(fell=fell,steps=t+1,relative_steady_state_error=error))
        survived=[r['relative_steady_state_error'] for r in trial_rows if not r['fell']]
        rows.append(dict(command=list(map(float,cmd)),trials=trial_rows,all_fell=not bool(survived),
                         median_relative_error=float(np.median(survived)) if survived else None))
    return dict(commands=rows,command_count=len(rows),trials=trials,duration_s=duration,
                initialization='local nominal G1 at .78m, seed-fixed joint noise +/- .005rad',
                metric='norm of mean final-2s heading velocity/yaw error divided by command norm; median surviving trials')
