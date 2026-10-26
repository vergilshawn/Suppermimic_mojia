"""Explicit fresh-physics model validation; no auto-start or background work."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from .kernel import proprio, spans
from .physics import CPUVectorPhysics

@torch.no_grad()
def independent_model_validation(env,model,actor,samples=8,seed=97031):
    """Fresh CPU physics transitions, separate RNG, open-loop prediction and Jacobian."""
    if samples<1:raise ValueError('Positive sample count required')
    model.cpu();actor.cpu();physics=CPUVectorPhysics(env,1,device='cpu')
    generator=torch.Generator().manual_seed(seed);errors={h:[] for h in (1,8,32)};jacobians=[]
    q,dq,last=spans(env.references[0,0]);epsilons=(.005,.01)
    for sample in range(samples):
        clip=sample%len(env.lengths);frame=int(torch.randint(max(env.lengths[clip]-33,1),(1,),generator=generator))
        context=torch.zeros(1,env.layout.context);context[:,0]=1
        state=env.references[clip,frame][None].clone();state[:,last]=0
        physics.reset(torch.tensor([0]),state,context)
        saved=physics.state_dict();initial=state.clone()
        history=proprio(state,env.task.nominal,context)[:,None].expand(-1,env.layout.history,-1).clone()
        predictions=state.clone();first_action=None
        for t in range(32):
            future=env.references[clip,torch.arange(frame+t,frame+t+env.layout.future)][None]
            action=actor(history,future,state)+.1*torch.randn(1,env.layout.dof,generator=generator)
            if first_action is None:first_action=action.clone()
            state,_=physics.step(action);predictions=model(predictions,action,context)
            history=torch.cat((history[:,1:],proprio(state,env.task.nominal,context)[:,None]),1)
            if t+1 in errors:
                errors[t+1].append(dict(root_position_mse=float((predictions[:,:3]-state[:,:3]).square().mean()),
                                        joint_mse=float((predictions[:,q]-state[:,q]).square().mean())))
        # One deterministic action-space direction per fresh state. No contact-gradient
        # accuracy claim can be inferred from a small directional sample.
        direction=torch.randn(first_action.shape,generator=generator);direction/=direction.norm().clamp_min(1e-9)
        with torch.enable_grad():
            _,predicted=torch.autograd.functional.jvp(lambda a:model(initial,a,context),first_action,direction)
        by_epsilon={}
        for epsilon in epsilons:
            physics.load_state_dict(saved);plus,_=physics.step(first_action+epsilon*direction)
            physics.load_state_dict(saved);minus,_=physics.step(first_action-epsilon*direction)
            physical=(plus-minus)/(2*epsilon);metrics={}
            for name,span in (('root_position',slice(0,3)),('joint_velocity',dq)):
                a=predicted[:,span].flatten();b=physical[:,span].flatten()
                metrics[name]=dict(cosine=float(torch.nn.functional.cosine_similarity(a[None],b[None],eps=1e-8)),
                                   relative_error=float((a-b).norm()/b.norm().clamp_min(1e-8)))
            by_epsilon[str(epsilon)]=metrics
        jacobians.append(by_epsilon)
    return dict(independent_physics_samples=samples,seed=seed,action_noise_std=.1,
                horizons={str(h):dict(root_position_rmse_m=float(np.sqrt(np.mean([r['root_position_mse'] for r in rows]))),
                                     joint_rmse_rad=float(np.sqrt(np.mean([r['joint_mse'] for r in rows])))) for h,rows in errors.items()},
                directional_jacobians=jacobians,finite_difference_epsilons=list(epsilons),
                markov_contract="82D physical state plus context assumes zero-delay torque-clipped PD; simulator integration/warmstart is restored for finite differences",
                note='Fresh reset/action trajectories, not replay data; small-sample directional checks do not certify contact derivatives')



def main():
    from .train import SCHEMA,MIGRATION,PROTECTED_REPORT,source_hashes,make_environment,make_networks,atomic_json
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--samples',type=int,default=8);p.add_argument('--smoke',action='store_true')
    args=p.parse_args();torch.set_num_threads(1);began=time.monotonic()
    if args.checkpoint.stat().st_size>512*1024*1024:raise ValueError('Use learner.pt rather than full replay checkpoint')
    cp=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if cp.get('schema')!=SCHEMA or cp['source_sha256']!=source_hashes():raise ValueError('Checkpoint/source schema mismatch')
    if args.smoke:
        if not cp['config'].get('smoke_test') or not 1<=args.samples<=2:
            raise ValueError('Only <=2 independent states from a tiny smoke learner are allowed during protected training')
    elif PROTECTED_REPORT.exists() and json.loads(PROTECTED_REPORT.read_text()).get('status')=='running':
        raise RuntimeError('Full model validation blocked while protected G1 training is running')
    env=make_environment(cp['config'])
    if env.contract!=cp['contract']:raise ValueError('Robot/data/physics contract mismatch')
    actor,critics,model,tasks=make_networks(env,cp['config'],'cpu')
    actor.load_state_dict(cp['actor']);model.load_state_dict(cp['model'])
    result=independent_model_validation(env,model,actor,args.samples)
    result.update(smoke_test=cp['config']['smoke_test'],checkpoint=str(args.checkpoint),
                  seconds=time.monotonic()-began,paper_reproduced=False,hardware_deployable=False)
    out=args.out.resolve()
    if out.exists() or not out.is_relative_to(MIGRATION):raise ValueError('Use a fresh isolated output')
    out.parent.mkdir(parents=True,exist_ok=True);atomic_json(result,out)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
