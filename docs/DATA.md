# Mojia reference data

The active data contains all 40 original human LAFAN clips, retargeted to the
short-arm Mojia v3.8 model and resampled to 50 Hz. No clips are filtered.
Elbow zero is bent approximately 90 degrees; T-pose orientation calibration uses
straight elbows (+pi/2) and abducted shoulders. Joint coordinates receive no
constant post-processing offset.

Retargeting uses external robot_retargeter and GMR. The local retargeting
Python environment uses MuJoCo 3.9; preparation/training use MuJoCo 3.8.1.
Set `ROBOT_RETARGETER` and `GMR_ROOT` before retargeting. External projects are
not vendored. Use `scripts/data.py retarget --help` for individual BVHs.

Prepare from the retained calibrated CSVs in the training environment:

```bash
python scripts/data.py prepare --source data/mojia_raw \
  --joint-order data/mojia_raw/joint_order.json --fps 30.000300003000028 \
  --audit configs/robot.json --correct-floor-clearance-m .002 \
  --out data/mojia_new
```

Outputs must use a fresh directory. First-order endpoint secants avoid velocity
extrapolation; interior differences remain centered. Penetrating reference poses
receive an explicitly logged upward translation, with velocities recomputed.
The current corpus still has leg collisions and up to 39.7cm floor correction.
Full inspection is retained at `runs/mojia_audit/index.html`; kinematic validity
alone does not establish dynamic trackability. No hardware validation is claimed.
