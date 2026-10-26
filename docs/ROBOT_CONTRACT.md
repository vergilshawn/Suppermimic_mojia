# G1 and Mojia robot contract audit

Checked 2026-10-26 against the active Mojia code and the G1 branch (5ae9e05).
Numerical evidence is in [VALIDATION.json](VALIDATION.json). G1 was loaded in an
isolated temporary directory; its branch and successful checkpoint were unchanged.

## Coordinates and dimensions

| Item | G1 | Mojia |
| --- | --- | --- |
| Floating root | pelvis | base_link |
| Torso / COM randomization body | torso_link | lumbar_yaw |
| Actuated joints | 29 | 23 |
| Physical state | 100 | 82 |
| Proprioception | 93 | 75 |
| Randomization context | 33 | 27 |
| Goal features per future frame | 74 | 62 |
| World-model input / output | 159 / 70 | 129 / 58 |
| Quaternion | world-from-root WXYZ | world-from-root WXYZ |
| Root linear velocity | world, body-frame origin, m/s | same |
| Root angular velocity | root-local rad/s | same |
| Future goals | t+1 through t+10 | corrected to match |

Mojia configuration order is NOT simulator DFS order. All qpos, qvel and actuator
accesses use named maps; direct index copying from G1 is invalid. Joint name suffixes
such as yaw do not establish equivalent axes or functions across the robots.
G1's additional waist/wrist DOFs have no one-to-one Mojia counterpart.

320 sampled states per robot, covering all 40 clips, were checked against MuJoCo:
Mojia FK position error <=1.47e-6m, velocity/angular-velocity error <=3.42e-6;
G1 position error <=1.26e-6m and velocity/angular error <=4.08e-6. World/body root
velocity conventions matched to double-precision numerical error. These are
sampled analytical-FK checks; full-frame reference geometry checks are separately
retained in `runs/mojia_audit/`.

## Corrected mismatches

- The prior Mojia goal window included the current frame and omitted the furthest
  next frame. Collection, BPTT, both evaluators, validation and policy rendering
  now share t+1..t+10 timing. Reference pose playback is unchanged.
- The layout advertised a 29-value dynamics output while the actual model emitted
  58. The class, exported dimension dictionary and network now agree on 58.
- Root, torso and tracking-point definitions are explicit in `configs/robot.json`.
  Compiled and prepared body-name order is checked rather than silently replaced.
- The prior Mojia distal arm target was the elbow-yaw body origin, near the elbow,
  whereas the G1 comparison tracks a distal wrist body. Mojia now uses a virtual
  hand tip 0.1384m along each elbow-yaw body's local x axis. This changes reward
  tracking locations, not robot geometry or reference joint poses. Point velocity
  includes angular velocity cross local offset. Its position matches the original
  retargeter hand target within 1.04e-6m across 40 tested states. This does not add
  nonexistent wrist joints or promise G1-equivalent dexterity.
- World-model validation previously used per-axis root RMS while policy evaluation
  used Euclidean-distance RMS, differing by sqrt(3). Published root-distance
  metrics now use sqrt(mean(sum_xyz(error squared))). Dimension-normalized fitting
  losses remain separate quantities.
- CPU inference action clipping matches GPU training/inference. Process guards
  now recognize the actual `python scripts/train.py` entry point.

Body pose rewards retain the same heading-aligned structure as G1. Body linear
and angular velocity rewards remain in world coordinates; they were not secretly
rotated or stripped of global direction. Global root position/orientation terms
remain active. Translating and yaw-rotating both robot and target preserves goal
features and rewards within floating-point tolerance.

## Physics and evaluation limitations

CPU and GPU are not interchangeable contact simulators. Across 40 reference
midpoints and 3 control steps, Mojia maximum joint difference reached 0.197rad;
G1 reached 0.0116rad in the corresponding short test. These are different robot
motions, not matched physical trajectories. The worst Mojia case started with
approximately 7.45cm of knee/hip penetration. With no contacts, discrepancies were
about 1e-6 even with randomized torso COM, supporting correct state and actuator
mapping. These bounded tests do not establish long-horizon simulator equivalence.

Mojia retains leg collisions in approximately 1.45% of reference frames at a 2cm
review threshold, and floor lifts up to 39.7cm. These references are unchanged by
this contract audit. Contact geometry, reference feasibility and backend behavior
remain unresolved; full training success is not established.

Use GPU policy rendering to inspect the training backend:

```bash
python scripts/visualize.py render --checkpoint runs/mojia/best_learner.pt \
  --device cuda:0 --motion dance1_subject2 --out runs/policy_preview
python scripts/evaluate.py single --checkpoint runs/mojia/best_learner.pt \
  --device cuda:0 --detailed --out runs/detailed_evaluation.json
```

Detailed evaluation adds full-rate global keypoint distance and torso orientation
metrics. Automatic training evaluation remains lightweight by default. Reference
playback is kinematic, independent of either controller backend. Completion only
means no configured height/tilt termination for the clip; it does not guarantee
accurate tracking, contact quality or hardware readiness.

## Parameters and hardware boundary

Network widths, history/future, latent size, PD timestep, fit horizon and major
batch settings match the G1 recipe structurally. Motor gains, effort limits,
armatures, nominal positions and action scales are robot-specific. Mojia uses a
50000-iteration run and cosine decay over 50000; G1 uses 100000 total and decay
period 250000. Reset perturbations, mixed clean collection, timeouts and regression
stopping are explicit Mojia settings, not coordinate-system fixes.

The real IMU mounting link has not been verified. An IMU on a moving torso cannot
be treated as a base_link IMU using a fixed mounting rotation alone; waist motion
and relative angular velocity must be accounted for. The offline bridge requires
root state in base_link coordinates. Simulation PD, hardware rate/torque limits,
parallel ankles and pose estimation remain unvalidated for real actuation.

Verification: 9 tests, two CPU smoke learning iterations including policy BPTT,
one full-size GPU world-model iteration, GPU detailed evaluation and a three-step
GPU policy render passed. No long training was launched. Temporary verification
weights/output folders were removed; this report retains the current evidence.
