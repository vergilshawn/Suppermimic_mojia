"""Independent GPU evaluation; no action/observation noise, full clips, immutable snapshots."""
import argparse,json,os,time
from pathlib import Path
import numpy as np
import torch
from .train import make_environment,make_networks,atomic_json,source_hashes
from .physics import WarpPhysics
from .kernel import proprio,spans
from .regression import RegressionGuard

@torch.no_grad()
def evaluate(env,actor,tasks,physics,device,max_steps=None):
    refs=env.references.to(device);n=len(env.lengths);ids=torch.arange(n,device=device)
    lengths=torch.tensor(env.lengths,device=device);state=refs[ids,0].clone();state[:,spans(state)[2]]=0
    ctx=torch.zeros(n,env.layout.context,device=device);ctx[:,0]=1
    physics.reset(ids,state,ctx);history=proprio(state,tasks.nominal,ctx)[:,None].repeat(1,env.layout.history,1)
    active=torch.ones(n,device=device,dtype=torch.bool);fallen=~active;counts=torch.zeros(n,device=device,dtype=torch.long)
    ret=torch.zeros(n,device=device);joint_error=ret.clone();root_error=ret.clone()
    budget=int(lengths.max())-1 if max_steps is None else min(max_steps,int(lengths.max())-1)
    for frame in range(budget):
        future=refs[ids[:,None],frame+torch.arange(env.layout.future,device=device)[None]]
        action=actor(history,future,state).clamp(-20,20);action[~active]=0
        nxt,_=physics.step(action);ref=refs[ids,frame+1]
        finite=torch.isfinite(nxt).all(-1);bad=tasks.termination(nxt,ref,False).bool()|~finite
        if not finite.all():raise FloatingPointError('Nonfinite evaluation physics')
        reward=tasks.reward(state,nxt,action,ref)
        ret+=torch.where(active,reward,0);counts+=active
        joint_error+=torch.where(active,(nxt[:,13:36]-ref[:,13:36]).square().mean(-1),0)
        root_error+=torch.where(active,(nxt[:,:3]-ref[:,:3]).square().sum(-1),0)
        fallen|=active&bad;active&=~bad & (frame+1<lengths-1)
        state=nxt;history=torch.cat((history[:,1:],proprio(state,tasks.nominal,ctx)[:,None]),1)
        if not active.any():break
        # Inactive worlds cannot poison GPU contact capacity after falling indefinitely.
        dead=ids[~active]
        if len(dead):physics.reset(dead,refs[dead,0],ctx[dead])
    motion=[]
    for i,name in enumerate(env.clip_names):
        steps=int(counts[i]);motion.append(dict(clip=name,steps=steps,fall=bool(fallen[i]),completed=bool(not fallen[i] and steps==env.lengths[i]-1),normalized_return=float(ret[i]/(.1*(env.lengths[i]-1))),joint_rmse_rad=float((joint_error[i]/max(steps,1)).sqrt()),root_rmse_m=float((root_error[i]/max(steps,1)).sqrt())))
    return dict(motions=motion,normalized_return=float(np.mean([r['normalized_return'] for r in motion])),completion_rate=float(np.mean([r['completed'] for r in motion])),evaluation_is_full=max_steps is None,action_noise=0,observation_noise=False)

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args(argv)
    torch.set_num_threads(1);torch.set_float32_matmul_precision('high');queue=a.run/'evaluation';queue.mkdir(exist_ok=True)
    env=actor=tasks=physics=None
    guard=RegressionGuard()
    for old in sorted(queue.glob("job_*.result.json")):
        previous=json.loads(old.read_text())
        if previous.get("status")=="complete":guard.observe(previous)
    while True:
        jobs=sorted(j for j in queue.glob('job_*.json') if '.result.' not in j.name)
        pending=[j for j in jobs if not j.with_suffix('.result.json').exists()]
        if not pending:
            if (a.run/'evaluation_stop').exists():break
            time.sleep(2);continue
        job=pending[0];spec=json.loads(job.read_text());began=time.monotonic()
        try:
            cp=torch.load(spec['learner'],map_location='cpu',weights_only=True)
            if cp['source_sha256']!=source_hashes():raise ValueError('Evaluator source mismatch')
            if env is None:
                env=make_environment(cp['config']);actor,_,_,tasks=make_networks(env,cp['config'],a.device)
                physics=WarpPhysics(env,len(env.lengths),device=a.device)
            if cp['contract']!=env.contract:raise ValueError('Evaluation contract mismatch')
            actor.load_state_dict(cp['actor']);actor.eval()
            result=evaluate(env,actor,tasks,physics,a.device)
            result.update(iteration=cp['iteration']+1,seconds=time.monotonic()-began,full_checkpoint=spec['full_checkpoint'],learner=spec['learner'],status='complete')
            improved,stop=guard.observe(result)
            if improved:
                for key,name in (('full_checkpoint','best.pt'),('learner','best_learner.pt')):
                    target=a.run/(name+'.tmp');target.unlink(missing_ok=True);os.link(spec[key],target);os.replace(target,a.run/name)
                atomic_json(dict(result,full_checkpoint=str((a.run/'best.pt').resolve()),learner=str((a.run/'best_learner.pt').resolve())),a.run/'best_evaluation.json')
            result['regression_bad_count']=guard.bad_count;result['regression_stop']=stop
            atomic_json(result,job.with_suffix('.result.json'))
            # Best survives via hard links; discard processed queue payloads, retain metadata.
            Path(spec['full_checkpoint']).unlink();Path(spec['learner']).unlink()
        except Exception as e:
            atomic_json(dict(status='error',error=repr(e),iteration=spec['iteration']),job.with_suffix('.result.json'));raise
if __name__=='__main__':main()
