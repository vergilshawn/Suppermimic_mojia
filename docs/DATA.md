# Mojia data

Input is the original human LAFAN BVH dataset. Retargeting uses external `robot_retargeter` (including its Mojia v3.8 XML/config) and GMR's BVH reader/TPOSE. Set their paths explicitly; they are not vendored into this repository.

```bash
export ROBOT_RETARGETER=/path/to/robot_retargeter
export GMR_ROOT=/path/to/GMR
python tools/prepare_bvh.py --source /path/to/lafan_bvh --raw-out data/mojia_raw --out data/mojia
```

The retargeting environment needs Mink and the version of MuJoCo supported by robot_retargeter (the local data was generated using MuJoCo 3.9.0, while training uses 3.8.1). If using separate environments, generate CSVs with `tools/retarget_bvh.py`, then use `python -m robot_control.runtime.prepare --source data/mojia_raw --joint-order data/mojia_raw/joint_order.json --fps 30.000300003 --out data/mojia --correct-floor-clearance-m .002` in the training environment.

The IK solver enforces the target joint limits and per-frame velocity bounds. Floor corrections are recorded; the existing corpus includes corrections up to approximately 0.484 m, so kinematic validity is not a guarantee of physical trackability. No claim of hardware readiness is made.
