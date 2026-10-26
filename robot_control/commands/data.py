"""Prepare and validate robot motion datasets."""
import argparse
import sys
import json
from pathlib import Path

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['prepare', 'check', 'retarget', 'pipeline'])
    if not argv or argv[0] in ('-h', '--help'): p.print_help(); return
    mode = p.parse_args(argv[:1]).mode
    if mode == 'prepare':
        from robot_control.runtime.prepare import main as prepare
        return prepare(argv[1:])
    if mode == 'pipeline':
        from .prepare_bvh import run_batch
        return run_batch(argv[1:])
    if mode == 'retarget':
        p = argparse.ArgumentParser(description='Retarget one raw human BVH using robot_retargeter and GMR.')
        p.add_argument('--source', type=Path, required=True)
        p.add_argument('--out', type=Path, required=True)
        p.add_argument('--limit', type=int)
        a = p.parse_args(argv[1:])
        from .retarget import run
        print(json.dumps(run(a.source, a.out, a.limit))); return
    p = argparse.ArgumentParser(description='Validate the dataset and robot contract used by a training recipe.')
    p.add_argument('--config', type=Path, default=Path('configs/train.json'))
    a = p.parse_args(argv[1:])
    from robot_control.runtime.train import make_environment
    env = make_environment(json.loads(a.config.read_text()))
    print(json.dumps(dict(clips=len(env.lengths), frames=sum(map(int, env.lengths)), contract=env.contract), indent=2))
