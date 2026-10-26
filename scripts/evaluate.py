
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
"""Evaluate policies, a checkpoint series, or an existing asynchronous queue."""
import argparse
import sys
from pathlib import Path

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['single', 'batch', 'worker'])
    if not argv or argv[0] in ('-h', '--help'):
        parser.print_help(); return
    mode = parser.parse_args(argv[:1]).mode
    if mode == 'worker':
        from robot_control.runtime.evaluation_worker import main as worker
        return worker(argv[1:])
    run = run_single
    if mode == 'single':
        return run(argv[1:])
    p = argparse.ArgumentParser(description='Evaluate checkpoints sequentially with the same dataset and budget.')
    p.add_argument('--checkpoints', nargs='+', required=True, type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--max-steps', type=int)
    a = p.parse_args(argv[1:])
    for i, checkpoint in enumerate(a.checkpoints):
        args = ['--checkpoint', str(checkpoint), '--out', str(a.out / f'{i:03d}_{checkpoint.stem}.json'), '--device', a.device]
        if a.max_steps is not None: args += ['--max-steps', str(a.max_steps)]
        run(args)

def run_single(argv=None):
    import argparse,json
    from pathlib import Path
    import torch
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


if __name__ == "__main__":
    main()
