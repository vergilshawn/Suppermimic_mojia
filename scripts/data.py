
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
"""Prepare and validate robot motion datasets."""
import argparse
import sys
import json
from pathlib import Path

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['prepare', 'check', 'audit', 'retarget', 'pipeline'])
    if not argv or argv[0] in ('-h', '--help'): p.print_help(); return
    mode = p.parse_args(argv[:1]).mode
    if mode == 'audit':
        p = argparse.ArgumentParser(description='Audit every training reference frame, including collision geometry.')
        p.add_argument('--config', type=Path, default=Path('configs/train.json'))
        p.add_argument('--out', type=Path, required=True)
        p.add_argument('--workers', type=int, default=4)
        a = p.parse_args(argv[1:])
        if a.workers < 1: p.error('--workers must be positive')
        from robot_control.dataset_audit import audit
        return audit(a.config, a.out, a.workers)
    if mode == 'prepare':
        from robot_control.runtime.prepare import main as prepare
        return prepare(argv[1:])
    if mode == 'pipeline':
        return run_batch(argv[1:])
    if mode == 'retarget':
        p = argparse.ArgumentParser(description='Retarget one raw human BVH using robot_retargeter and GMR.')
        p.add_argument('--source', type=Path, required=True)
        p.add_argument('--out', type=Path, required=True)
        p.add_argument('--limit', type=int)
        a = p.parse_args(argv[1:])
        from robot_control.retarget import run
        print(json.dumps(run(a.source, a.out, a.limit))); return
    p = argparse.ArgumentParser(description='Validate the dataset and robot contract used by a training recipe.')
    p.add_argument('--config', type=Path, default=Path('configs/train.json'))
    a = p.parse_args(argv[1:])
    from robot_control.runtime.train import make_environment
    env = make_environment(json.loads(a.config.read_text()))
    print(json.dumps(dict(clips=len(env.lengths), frames=sum(map(int, env.lengths)), contract=env.contract), indent=2))

ROOT = Path(__file__).resolve().parents[1]

def run_batch(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--raw-out',type=Path,default=ROOT/'data/mojia_raw');p.add_argument('--out',type=Path,default=ROOT/'data/mojia');p.add_argument('--audit',type=Path,default=ROOT/'configs/robot.json');a=p.parse_args(argv)
    from robot_control.retarget import run
    from robot_control.runtime.prepare import prepare
    if a.out.exists():raise FileExistsError(a.out)
    names=json.loads((ROOT/'configs/motions.json').read_text());fps=[]
    for name in names:
     metadata=a.raw_out/(name+'.json')
     if not metadata.exists():run(a.source/(name+'.bvh'),a.raw_out)
     record=json.loads(metadata.read_text())
     if 'shoulders abducted' not in record.get('elbow_zero',''):
      raise ValueError('Cached motion predates corrected elbow/T-pose calibration; use a fresh raw output directory')
     fps.append(record['fps'])
    if max(fps)-min(fps)>1e-6:raise ValueError('Mixed source frame rates require per-clip preparation')
    prepare(a.raw_out,a.out,json.loads((a.raw_out/'joint_order.json').read_text()),fps[0],floor_clearance_m=.002,audit=a.audit)


if __name__ == "__main__":
    main()
