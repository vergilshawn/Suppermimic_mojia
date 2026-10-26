"""Import a trusted local checkpoint after package-only relocation; tensors stay unchanged."""
import argparse,copy,json,sys
from pathlib import Path
import torch
from robot_control.runtime.train import make_environment

def run(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args(argv)
    if a.out.exists():raise FileExistsError(a.out)
    cp=torch.load(a.source,map_location='cpu',weights_only=True)
    origin=dict(source=str(a.source),config=copy.deepcopy(cp['config']),contract=copy.deepcopy(cp['contract']),source_sha256=cp.get('source_sha256'),note='Package and path relocation only; model, optimizer, RNG and replay tensors unchanged')
    robot='mojia' if cp.get('schema')=='mojia_ndof_v6' else 'g1'
    cp['config']['assets']='data/'+robot;cp['config']['profile']=robot+'_motion'
    env=make_environment(cp['config']);cp['contract']=env.contract
    if robot=='mojia':
     from robot_control.runtime.train import source_hashes
     cp['source_sha256']=source_hashes()
     if 'collector' in cp:cp['collector']['physics']['contract']=env.contract
    cp['packaging_origin']=origin
    a.out.parent.mkdir(parents=True,exist_ok=True);torch.save(cp,a.out)
    print(json.dumps(dict(robot=robot,output=str(a.out),iteration=cp['iteration'],full_state='optimizers' in cp)))
