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
        from .checkpoint_import import run
        return run(argv[1:])
    p = argparse.ArgumentParser(description='Read metadata from a trusted local checkpoint (CPU).')
    p.add_argument('--path', required=True, type=Path)
    a = p.parse_args(argv[1:])
    import torch
    cp = torch.load(a.path, map_location='cpu', weights_only=True)
    print(json.dumps({k: cp.get(k) for k in ('schema', 'iteration', 'run_id', 'config', 'contract')}, indent=2))
