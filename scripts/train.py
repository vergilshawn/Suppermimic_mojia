import os, subprocess, sys
"""Training arguments and launch configuration."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robot_control.runtime.train import (Path, ROOT, ReplayV3, argparse, json, run, validate_config)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/motion.json')
    p.add_argument('--out',type=Path);p.add_argument('--device',default='cuda:0');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--eval-device',default='cuda:0')
    p.add_argument('--resume',type=Path);p.add_argument('--preflight',action='store_true')
    p.add_argument('--verification-iterations',type=int,help='Explicit bounded integration verification, not paper training')
    args=p.parse_args(argv);c=json.loads(args.config.read_text());n=validate_config(c)
    if args.preflight:
        print(json.dumps(dict(config=c,vector_steps_per_iteration=n,physical_transitions_per_iteration=n*c['simulator_environments'],
                              replay_memory_estimate=ReplayV3.estimate_bytes(c['replay_transitions'],c['simulator_environments']),
                              filtering_deferred=True,paper_reproduced=False,training_started=False),indent=2));return
    if args.out is None:p.error('--out required for execution')
    if args.verification_iterations is not None and args.verification_iterations<1:p.error('Positive verification budget required')
    if args.out.exists() and args.resume is None:p.error('New recipe requires a fresh output directory')
    import os,subprocess,sys
    args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=True)
    worker_env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    if args.eval_device=='cpu':worker_env['CUDA_VISIBLE_DEVICES']=''
    with (args.out/'evaluation_worker.log').open('a') as log:
        worker=subprocess.Popen([sys.executable,'-m','robot_control.runtime.evaluation_worker',
            '--run',str(args.out),'--parent-pid',str(os.getpid()),'--device',args.eval_device],
            cwd=ROOT,env=worker_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    run(c,args.out,args.device,args.seed,args.resume,args.verification_iterations)


if __name__=='__main__':main()
