"""Batch raw human LAFAN retargeting followed by target-robot FK/velocity preparation."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from robot_control.runtime.prepare import prepare

def run_batch(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--raw-out',type=Path,default=ROOT/'data/mojia_raw');p.add_argument('--out',type=Path,default=ROOT/'data/mojia');a=p.parse_args(argv)
    from .retarget import run
    if a.out.exists():raise FileExistsError(a.out)
    names=json.loads((ROOT/'configs/motions.json').read_text());fps=[]
    for name in names:
     metadata=a.raw_out/(name+'.json')
     if not metadata.exists():run(a.source/(name+'.bvh'),a.raw_out)
     fps.append(json.loads(metadata.read_text())['fps'])
    if max(fps)-min(fps)>1e-6:raise ValueError('Mixed source frame rates require per-clip preparation')
    prepare(a.raw_out,a.out,json.loads((a.raw_out/'joint_order.json').read_text()),fps[0],floor_clearance_m=.002)
