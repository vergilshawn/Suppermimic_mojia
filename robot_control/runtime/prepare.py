"""Prepare named Mojia references at 50 Hz and rebuild FK on the target robot.

CSV layout: xyz metres, quaternion XYZW, 23 joint radians. NPZ accepts numeric
joint_pos/root_pos/root_rot_wxyz plus fps. Both require explicit source names;
object arrays are never unpickled. This is kinematic data, not transition data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from ..robot_layout import ExplicitJointOrder
from .environment import BASE, audited_control, build_model, named_addresses, sha


def read_motion(path, joint_names, target_names, fps=None):
    path = Path(path)
    mapping = ExplicitJointOrder(joint_names, target_names)
    order = mapping.target_source_indices
    if path.suffix.lower() == ".csv":
        values = np.loadtxt(path, delimiter=",")
        if values.ndim != 2 or values.shape[1] != 30:
            raise ValueError("CSV requires xyz, quaternion XYZW and exactly 23 joints")
        root, quat, q = values[:, :3], values[:, 3:7], values[:, 7:]
        if fps is None:
            raise ValueError("CSV source FPS must be explicit")
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            q = np.asarray(data["joint_pos"], dtype=np.float64)
            root = np.asarray(data["root_pos"], dtype=np.float64) if "root_pos" in data else np.asarray(data["body_pos_w"][:, 0], dtype=np.float64)
            quat_wxyz = np.asarray(data["root_rot_wxyz"], dtype=np.float64) if "root_rot_wxyz" in data else np.asarray(data["body_quat_w"][:, 0], dtype=np.float64)
            quat = quat_wxyz[:, [1, 2, 3, 0]]
            embedded_fps = float(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in data else None
            if fps is not None and embedded_fps is not None and not np.isclose(fps, embedded_fps):
                raise ValueError("Explicit source FPS disagrees with NPZ metadata")
            fps = embedded_fps if fps is None else fps
    else:
        raise ValueError("Only CSV and numeric NPZ inputs are supported")
    if fps is None or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Source FPS must be positive")
    if q.shape != (len(root), 23) or root.shape != (len(q), 3) or quat.shape != (len(q), 4) or len(q) < 3:
        raise ValueError("Motion needs at least 3 frames and consistent root/joint shapes")
    if not all(np.isfinite(x).all() for x in (root, quat, q)):
        raise ValueError("Nonfinite source values")
    norms = np.linalg.norm(quat, axis=1)
    if np.max(np.abs(norms - 1.)) > .01:
        raise ValueError("Source quaternion norm differs from one by more than .01")
    # Slerp normalizes near-unit source quaternions; no angle/height clipping.
    times = np.arange(len(q)) / fps
    target = np.arange(int(np.floor(times[-1] / .02 + 1e-8)) + 1) * .02
    if len(target) < 3:
        raise ValueError("Motion is too short for target 50 Hz velocity differences")
    ordered_q = q[:, order]
    pose = np.column_stack([np.interp(target, times, x) for x in np.column_stack((root, ordered_q)).T])
    rotations = Slerp(times, Rotation.from_quat(quat))(target)
    lo, hi = np.maximum(np.arange(len(target)) - 1, 0), np.minimum(np.arange(len(target)) + 1, len(target) - 1)
    omega_world = (rotations[hi] * rotations[lo].inv()).as_rotvec() / ((hi - lo) * .02)[:, None]
    states = np.zeros((len(target), 82), dtype=np.float32)
    states[:, :3] = pose[:, :3]
    states[:, 3:7] = rotations.as_quat()[:, [3, 0, 1, 2]]
    states[:, 7:10] = np.gradient(pose[:, :3], .02, axis=0, edge_order=2)
    states[:, 10:13] = rotations.inv().apply(omega_world)
    states[:, 13:36] = pose[:, 3:]
    states[:, 36:59] = np.gradient(pose[:, 3:], .02, axis=0, edge_order=2)
    metadata = dict(source=str(path.resolve()), source_sha256=sha(path), source_frames=len(q),
                    source_fps=fps, target_fps=50, joint_mapping=mapping.to_dict(),
                    source_quaternion_norm_max_error=float(np.max(np.abs(norms - 1))),
                    duration_s=float(target[-1]), source_duration_s=float(times[-1]),
                    source_root_pose_basis="NPZ root fields or body zero; CSV XYZ+XYZW")
    return states, metadata


def kinematics(model, control, states):
    qids, vids, _ = named_addresses(model, control["joint_names"])
    data = mujoco.MjData(model)
    shape = (len(states), model.nbody - 1)
    pos, quat, vel, omega = (np.empty((*shape, d), dtype=np.float32) for d in (3, 4, 3, 3))
    feet = [model.body(side + "_ankle_roll").id for side in ("left", "right")]
    floor = model.geom("floor").id
    foot_geoms = [[i for i in range(model.ngeom) if model.geom_bodyid[i] == body and
                   (model.geom_contype[i] or model.geom_conaffinity[i])] for body in feet]
    contact = np.zeros((len(states), 2), dtype=bool)
    speed = np.empty((len(states), 2), dtype=np.float32)
    foot_distance = np.empty((len(states), 2), dtype=np.float32)
    minimum = np.zeros(len(states), dtype=np.float32)
    floor_minimum = np.zeros(len(states), dtype=np.float32)
    spatial = np.zeros(6)
    for t, state in enumerate(states):
        data.qpos[:7], data.qpos[qids] = state[:7], state[13:36]
        data.qvel[:6], data.qvel[vids] = state[7:13], state[36:59]
        mujoco.mj_forward(model, data)
        pos[t], quat[t] = data.xpos[1:], data.xquat[1:]
        for body in range(1, model.nbody):
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body, spatial, 0)
            # mjOBJ_BODY's velocity origin is body COM; translate to body frame origin.
            omega[t, body - 1] = spatial[:3]
            vel[t, body - 1] = spatial[3:] + np.cross(spatial[:3], data.xpos[body] - data.xipos[body])
        for side, body in enumerate(feet):
            speed[t, side] = np.linalg.norm(vel[t, body - 1])
            # Geometry distance also detects a near-floor foot with a positive
            # clearance, where MuJoCo's collision list contains no contact yet.
            foot_distance[t, side] = min([.05] + [mujoco.mj_geomDistance(model, data, floor, geom, .05, None)
                                                 for geom in foot_geoms[side]])
            contact[t, side] = foot_distance[t, side] <= .025 and speed[t, side] < .2
        if data.ncon:
            minimum[t] = min(0., min(c.dist for c in data.contact))
            floor_minimum[t] = min([0.] + [c.dist for c in data.contact if floor in (c.geom1, c.geom2)])
    return dict(body_pos_w=pos, body_quat_w=quat, body_lin_vel_w=vel, body_ang_vel_w=omega,
                foot_contact_kinematic=contact, foot_origin_speed_m_s=speed,
                foot_floor_distance_m=foot_distance,
                minimum_contact_distance_m=minimum, minimum_floor_contact_distance_m=floor_minimum)


def validate_states(states, control):
    q, dq = states[:, 13:36], states[:, 36:59]
    violation = np.maximum(np.asarray(control["lower"]) - q, 0) + np.maximum(q - np.asarray(control["upper"]), 0)
    speed_excess = np.maximum(np.abs(dq) - np.asarray(control["velocity_limit"]), 0)
    return dict(max_joint_limit_violation_rad=float(violation.max()),
                joint_limit_violating_frames=int((violation > 1e-5).any(axis=1).sum()),
                max_joint_speed_excess_rad_s=float(speed_excess.max()),
                joint_speed_violating_frames=int((speed_excess > 1e-4).any(axis=1).sum()),
                kinematically_eligible=bool(violation.max() <= 1e-5 and speed_excess.max() <= 1e-4),
                clipping_applied=False, physical_tracking_validated=False)


def prepare(source, out, joint_names=None, fps=None, *, synthetic_standing=False, frames=128,
            floor_clearance_m=None):
    out = Path(out).resolve()
    if not out.is_relative_to(BASE):
        raise ValueError("Prepared output must remain inside isolated migration/mojia_v3_8")
    if out.exists():
        raise FileExistsError("Use a fresh prepared directory; existing data is never overwritten")
    if floor_clearance_m is not None and (not np.isfinite(floor_clearance_m) or not 0 <= floor_clearance_m <= .02):
        raise ValueError("Explicit floor-clearance correction must be between zero and .02m")
    control = audited_control()
    model = build_model(control)
    control["body_names"] = [model.body(i).name for i in range(1, model.nbody)]
    if synthetic_standing:
        if source is not None or frames < 3:
            raise ValueError("Synthetic smoke input must be explicit and separate from real sources")
        states = np.zeros((frames, 82), dtype=np.float32)
        states[:, 2], states[:, 3], states[:, 13:36] = .87, 1., control["nominal"]
        motions = [("SYNTHETIC_STANDING_SMOKE_ONLY", states, dict(source="synthetic nominal pose; not a motion dataset", duration_s=(frames - 1) * .02))]
    else:
        source = Path(source)
        files = sorted(p for p in source.iterdir() if p.suffix.lower() in (".csv", ".npz")) if source.is_dir() else [source]
        if not joint_names or not files:
            raise ValueError("Explicit source joint names and at least one CSV/NPZ required")
        motions = [(p.stem, *read_motion(p, joint_names, control["joint_names"], fps)) for p in files]
    out.mkdir(parents=True)
    (out / "motions").mkdir()
    (out / "control.json").write_text(json.dumps(control, indent=2) + "\n")
    clips = []
    for name, states, provenance in motions:
        dest = out / "motions" / (name + ".npz")
        if dest.exists():
            raise ValueError("Duplicate source clip stems would overwrite a prepared motion")
        values = kinematics(model, control, states)
        correction = {}
        if floor_clearance_m is not None:
            original_root = states[:, :3].copy()
            penetration = values["minimum_floor_contact_distance_m"].astype(np.float64)
            # Only lift penetrating frames. Preserve all joint poses and source
            # files, log every shift, and recompute linear/body velocities.
            shift = np.where(penetration < 0., -penetration + floor_clearance_m, 0.)
            states = states.copy()
            states[:, 2] += shift
            states[:, 7:10] = np.gradient(states[:, :3].astype(np.float64), .02, axis=0, edge_order=2)
            values = kinematics(model, control, states)
            values["original_root_pos"] = original_root
            values["root_height_correction_m"] = shift.astype(np.float32)
            correction = dict(explicit_root_height_correction=dict(
                method="per-frame minimal upward translation of floor-penetrating poses plus clearance",
                requested_clearance_m=floor_clearance_m, modified_frames=int((shift > 0).sum()),
                max_shift_m=float(shift.max()), rms_shift_m=float(np.sqrt(np.mean(shift ** 2))),
                root_linear_velocity_recomputed=True, joints_changed=False,
                dynamics_feasibility_validated=False))
        validation = validate_states(states, control)
        np.savez_compressed(dest, states=states, fps=np.array([50.]),
                            joint_names=np.array(control["joint_names"]), body_names=np.array(control["body_names"]),
                            root_pos=states[:, :3], root_rot_wxyz=states[:, 3:7],
                            joint_pos=states[:, 13:36], joint_vel=states[:, 36:59], **values)
        clips.append(dict(file=str(dest.relative_to(out)), frames=len(states), **provenance, **validation, **correction,
                          min_contact_distance_m=float(values["minimum_contact_distance_m"].min())))
    meshdir = (BASE / control["urdf"]).parent.parent / "meshes"
    manifest = dict(schema=1, robot="mojia_v3_8", synthetic_only=synthetic_standing,
                    target_fps=50, clips=clips, reference_data_only=True,
                    all_clips_kinematically_eligible=all(c["kinematically_eligible"] for c in clips),
                    physical_tracking_validated=False, hardware_validated=False,
                    foot_contact_labels="kinematic floor contact and foot-origin speed < .2m/s; not force labels",
                    asset_dependencies={str(p.relative_to(BASE)): sha(p) for p in sorted(meshdir.rglob("*")) if p.is_file()},
                    files={str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file()})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--joint-order", type=Path)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--synthetic-standing", action="store_true")
    parser.add_argument("--frames", type=int, default=128)
    parser.add_argument("--correct-floor-clearance-m", type=float,
                        help="Explicit derived-data correction only; logs every root-height shift and regenerates velocities")
    args = parser.parse_args(argv)
    names = json.loads(args.joint_order.read_text()) if args.joint_order else None
    if isinstance(names, dict):
        if args.source and args.source.is_file() and names.get("source_sha256") and sha(args.source) != names["source_sha256"]:
            raise ValueError("Joint-order sidecar source digest differs from the input motion")
        names = names["joint_names"]
    if not args.synthetic_standing and (args.source is None or names is None):
        parser.error("Real references require --source and --joint-order")
    manifest = prepare(args.source, args.out, names, args.fps,
                       synthetic_standing=args.synthetic_standing, frames=args.frames,
                       floor_clearance_m=args.correct_floor_clearance_m)
    print(json.dumps(dict(output=str(args.out), clips=len(manifest["clips"]),
                          synthetic_only=manifest["synthetic_only"],
                          all_clips_kinematically_eligible=manifest["all_clips_kinematically_eligible"])))


if __name__ == "__main__":
    main()
