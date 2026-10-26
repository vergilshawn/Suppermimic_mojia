"""Evaluate a trusted policy checkpoint without exploration noise."""
import argparse,json
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True,type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--max-steps',type=int);a=p.parse_args()
torch.set_num_threads(1)
from robot_control.runtime.evaluation_worker import load,evaluate
cp,env,actor=load(a.checkpoint)
if a.device.startswith('cuda'):
 from robot_control.runtime.gpu_evaluation import evaluate
result=evaluate(env,actor,max_steps=a.max_steps,device=a.device) if a.device.startswith('cuda') else evaluate(env,actor,max_steps=a.max_steps)
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='motions'}))
