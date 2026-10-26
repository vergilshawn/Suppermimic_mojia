"""Training arguments and launch configuration."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robot_control.runtime.train import (PROTECTED_REPORT, Path, Replay, RobotLayout, argparse, default_config, json, run, validate_config)

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path);parser.add_argument('--out',type=Path)
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--resume',type=Path);parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--smoke',action='store_true',help='Explicit <=2 iteration tiny CPU physics and learning check')
    parser.add_argument('--verification-iterations',type=int)
    args=parser.parse_args(argv)
    if args.smoke and args.config is not None:parser.error('--smoke owns its fixed tiny config; do not combine with --config')
    if args.config is None and not args.smoke:parser.error('--config required unless --smoke')
    c=default_config(True) if args.smoke else json.loads(args.config.read_text())
    device='cpu' if args.smoke else args.device;steps=validate_config(c)
    if args.preflight:
        layout=RobotLayout(c['dof'],c['history'],c['future'],c['latent'])
        protected=json.loads(PROTECTED_REPORT.read_text()).get('status') if PROTECTED_REPORT.exists() else 'not_found'
        print(json.dumps(dict(config=c,layout=layout.to_dict(),vector_steps_per_iteration=steps,
                              physical_transitions_per_iteration=steps*c['simulator_environments'],
                              replay_memory=Replay.estimate_bytes(c['replay_transitions'],c['simulator_environments'],c['history'],layout=layout),
                              protected_training_status=protected,production_blocked=False,process_conflict_checked_at_launch=True,
                              training_started=False,paper_reproduced=False),indent=2));return
    if args.out is None:parser.error('--out required for explicit execution')
    run(c,args.out,device,args.seed,args.resume,args.verification_iterations)


if __name__=='__main__':main()
