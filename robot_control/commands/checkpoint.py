"""Inspect or explicitly import trusted local checkpoints after path relocation."""
import argparse
import sys
import json
from pathlib import Path

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['inspect', 'import'])
    if not argv or argv[0] in ('-h', '--help'): p.print_help(); return
    mode = p.parse_args(argv[:1]).mode
    if mode == 'import':
        run = import_checkpoint
        return run(argv[1:])
    p = argparse.ArgumentParser(description='Read metadata from a trusted local checkpoint (CPU).')
    p.add_argument('--path', required=True, type=Path)
    a = p.parse_args(argv[1:])
    import torch
    cp = torch.load(a.path, map_location='cpu', weights_only=True)
    print(json.dumps({k: cp.get(k) for k in ('schema', 'iteration', 'run_id', 'config', 'contract')}, indent=2))

def import_checkpoint(argv=None):
    import argparse,copy,json,sys
    from pathlib import Path
    import torch
    from robot_control.runtime.train import make_environment
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
