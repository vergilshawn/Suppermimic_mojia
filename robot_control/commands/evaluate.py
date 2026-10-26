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
    from .evaluation import run
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
