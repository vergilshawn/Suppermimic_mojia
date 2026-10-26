"""Evaluate a trusted policy checkpoint without exploration noise."""
import argparse,json
from pathlib import Path
import torch

def run(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True,type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--max-steps',type=int);a=p.parse_args(argv)
    torch.set_num_threads(1)
    from robot_control.runtime.train import make_environment,make_networks,source_hashes
    cp=torch.load(a.checkpoint,map_location='cpu',weights_only=True);env=make_environment(cp['config'])
    if cp['contract']!=env.contract:raise ValueError('Robot/data contract mismatch')
    actor,_,_,tasks=make_networks(env,cp['config'],a.device);actor.load_state_dict(cp['actor']);actor.eval()
    if a.device.startswith('cuda'):
     from robot_control.runtime.physics import WarpPhysics
     from robot_control.runtime.evaluation_worker import evaluate
     result=evaluate(env,actor,tasks,WarpPhysics(env,len(env.lengths),device=a.device),a.device,a.max_steps)
    else:
     from robot_control.runtime.evaluate import physical_evaluation
     result=physical_evaluation(env,actor,tasks,max_steps=a.max_steps)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='motions'}))
