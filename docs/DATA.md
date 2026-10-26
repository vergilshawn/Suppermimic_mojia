# G1 data

Place the 40 LAFAN G1 CSV sequences in a local directory. The expected layout is root XYZ (metres), XYZW quaternion, and 29 joint angles (radians), at 30 Hz. These are G1-retargeted motions, not raw human BVH.

```bash
python -m robot_control.runtime.prepare_lafan --source /path/to/g1_csv --assets robots/g1 --out data/g1
```

The builder generates 50 Hz references, FK, velocities, a control contract and checksums. Existing output directories are not overwritten. Data and checkpoint files are excluded from Git; see their original licenses before redistribution.
