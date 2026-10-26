"""GPU policy, references, FK/reward and MuJoCo Warp physical evaluation."""
import time
import torch
from robot_control.runtime.kernel import proprio
from robot_control.runtime.physics import WarpPhysics


@torch.inference_mode()
def evaluate(env,actor,clips=None,max_steps=None,device='cuda:0',**unused):
    device=torch.device(device)
    if device.type!='cuda':raise ValueError('GPU evaluator requires CUDA')
    free,total=torch.cuda.mem_get_info(device)
    if free<4*1024**3:raise RuntimeError('GPU evaluation needs at least 4 GiB free before construction')
    torch.cuda.reset_peak_memory_stats(device);began=time.monotonic()
    clips=list(range(len(env.lengths))) if clips is None else list(clips)
    if not clips or len(set(clips))!=len(clips):raise ValueError('Unique nonempty clips required')
    actor=actor.to(device);task=env.task.to(device);bank=env.references.to(device)
    ids=torch.tensor(clips,device=device);n=len(clips)
    lengths=torch.tensor([env.lengths[c]-1 for c in clips],device=device)
    ctx=torch.zeros(n,33,device=device);ctx[:,0]=1
    initial=bank[ids,0].clone();initial[:,71:]=0
    physics=WarpPhysics(env,n,str(device))
    worlds=torch.arange(n,device=device);physics.reset(worlds,initial,ctx)
    state=physics.state();history=proprio(state,task.nominal,ctx,False)[:,None].expand(-1,10,-1).clone()
    alive=torch.ones(n,dtype=torch.bool,device=device);failed=~alive
    steps=torch.zeros(n,dtype=torch.long,device=device)
    returns=torch.zeros(n,dtype=torch.float64,device=device);qe=torch.zeros_like(returns);pe=torch.zeros_like(returns)
    future_index=torch.arange(1,11,device=device)
    budget=int(lengths.max()) if max_steps is None else min(max_steps,int(lengths.max()))
    torch.cuda.synchronize(device);setup=time.monotonic()-began;start=time.monotonic()
    for f in range(budget):
        if not bool(alive.any()):break
        if bool((~alive).any()):physics.reset(worlds[~alive],initial[~alive],ctx[~alive])
        future=bank[ids[:,None],f+future_index[None,:]]
        action=actor(history,future,state)
        action=torch.where(alive[:,None],action,0.)
        nxt,diag=physics.step(action)
        ref=bank[ids,f+1]
        reward=task.reward(state,nxt,action,ref)
        if not bool(torch.isfinite(nxt[alive]).all() & torch.isfinite(reward[alive]).all()):raise FloatingPointError('Nonfinite evaluation')
        returns+=torch.where(alive,reward,0).double()
        qe+=torch.where(alive,(nxt[:,13:42]-ref[:,13:42]).square().mean(-1),0).double()
        pe+=torch.where(alive,(nxt[:,:3]-ref[:,:3]).square().sum(-1),0).double()
        fail=task.termination(nxt,ref,False)&alive
        steps+=alive;failed|=fail;alive=alive & ~fail & (steps<lengths)
        state=nxt;history=torch.cat((history[:,1:],proprio(state,task.nominal,ctx,False)[:,None]),1)
    torch.cuda.synchronize(device);elapsed=time.monotonic()-start
    rows=[]
    for i,c in enumerate(clips):
        rows.append(dict(clip=env.clip_names[c],steps=int(steps[i]),completed=bool(~failed[i] & (steps[i]==lengths[i])),
                         terminated=bool(failed[i]),budget_limited=bool(alive[i]),normalized_return=float(returns[i]/(.1*lengths[i])),
                         joint_rmse_rad=float((qe[i]/steps[i].clamp_min(1)).sqrt()),root_rms_distance_m=float((pe[i]/steps[i].clamp_min(1)).sqrt())))
    free_after,_=torch.cuda.mem_get_info(device)
    return dict(motions=rows,normalized_return=sum(r['normalized_return'] for r in rows)/n,
                completion_rate=sum(r['completed'] for r in rows)/n,
                full_library=clips==list(range(len(env.lengths))) and max_steps is None,
                device=str(device),physics='MuJoCo Warp, identical v3 clipped PD; backend differs from CPU evaluation',
                setup_seconds=setup,rollout_seconds=elapsed,free_before_bytes=free,free_after_bytes=free_after,
                torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
