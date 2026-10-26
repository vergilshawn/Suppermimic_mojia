"""Atomic, immutable evaluation jobs; no learner RNG or optimizer changes."""
import hashlib
import json
from pathlib import Path
import torch


def runtime_hashes():
    base=Path(__file__).parent
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(base.glob('*.py'))}
    return files


def atomic_json(value,path):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def publish(out,iteration,config,contract,sources,actor):
    folder=Path(out)/'evaluation';folder.mkdir(exist_ok=True)
    path=folder/f'policy_{iteration+1:09d}.pt'
    job=path.with_suffix('.job.json')
    if job.exists():return dict(iteration=iteration+1,job=str(job),status='already_queued')
    payload=dict(schema=5,iteration=iteration,config=config,contract=contract,source_sha256=sources,
                 actor={k:v.detach().cpu().clone() for k,v in actor.state_dict().items()})
    tmp=path.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(path)
    atomic_json(dict(iteration=iteration+1,checkpoint=path.name,
                     sha256=hashlib.sha256(path.read_bytes()).hexdigest()),job)
    return dict(iteration=iteration+1,job=str(job),status='queued')
