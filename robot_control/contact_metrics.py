"""Foot/ground contact diagnostics sampled from the actual MuJoCo solver.

Sampling happens after a control-step endpoint ``mj_forward``. Forces are in N,
relative contact-point speeds in m/s, and penetration depths in m. The sampled
force maximum is neither a substep peak nor an impulse, and the statistics do
not integrate forces or slip distance between samples. No reference labels are
used to decide whether contact exists.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np


def rigid_body_point_velocity(model, data, body_id, point_world):
    """World velocity at a world-space point of a rigid body, in m/s.

    ``mjOBJ_BODY`` reports angular/linear velocity at the body's inertial COM.
    Translating to the contact point is essential: foot-origin velocity alone
    misses slip caused by foot rotation. A moving ground body is handled by the
    same operation before taking the relative velocity.
    """
    point = np.asarray(point_world, dtype=np.float64)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError("Expected a finite world-space contact point")
    if not 0 <= body_id < model.nbody:
        raise ValueError("Invalid body ID")
    if body_id == 0:
        return np.zeros(3)
    spatial = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, spatial, 0)
    return spatial[3:] + np.cross(spatial[:3], point - data.xipos[body_id])


class FootContactMetrics:
    """Aggregate actual, named foot/ground contacts at control endpoints.

    Geometric contacts include MuJoCo records with zero force, for example an
    inactive contact inside a geom margin. Slip is aggregated only for records
    whose compressive normal force exceeds ``minimum_normal_force_n``. No-load
    slip statistics are ``None``; all-sample force means and maximum penetration
    are zero when there are no contacts. Means over contact records are distinct
    from means over time samples, so multiple sole contacts are not mistaken for
    multiple elapsed control periods.
    """
    def __init__(self, model, *, foot_bodies=None, ground_geoms=("floor",),
                 minimum_normal_force_n=1e-6, control_dt_s=.02):
        if not math.isfinite(minimum_normal_force_n) or minimum_normal_force_n < 0:
            raise ValueError("Normal-force threshold must be finite and nonnegative")
        if not math.isfinite(control_dt_s) or control_dt_s <= 0:
            raise ValueError("Control period must be positive")
        self.model = model
        self.minimum_normal_force_n = float(minimum_normal_force_n)
        self.control_dt_s = float(control_dt_s)
        self.foot_names = dict(foot_bodies or {"left": "left_ankle_roll", "right": "right_ankle_roll"})
        if not self.foot_names or len(set(self.foot_names.values())) != len(self.foot_names):
            raise ValueError("Named foot bodies must be nonempty and distinct")
        self.foot_ids = {side: model.body(name).id for side, name in self.foot_names.items()}
        self.ground_names = tuple(ground_geoms)
        if not self.ground_names or len(set(self.ground_names)) != len(self.ground_names):
            raise ValueError("Named ground geoms must be nonempty and distinct")
        self.ground_ids = {model.geom(name).id for name in self.ground_names}
        self.geom_to_foot = {}
        self.foot_geoms = {}
        for side, body in self.foot_ids.items():
            ids = [i for i in range(model.ngeom) if model.geom_bodyid[i] == body and
                   (model.geom_contype[i] or model.geom_conaffinity[i])]
            if not ids:
                raise ValueError(f"Foot body has no collision geoms: {self.foot_names[side]}")
            if self.ground_ids.intersection(ids):
                raise ValueError("A ground geom cannot also be a foot geom")
            self.foot_geoms[side] = [model.geom(i).name for i in ids]
            self.geom_to_foot.update({i: side for i in ids})
        self.sample_count = 0
        self.stats = {side: dict(force_sum=0., force_square_sum=0., force_max=0.,
                                normal_world_sum=np.zeros(3), geometric_contact_samples=0,
                                loaded_control_samples=0, geometric_records=0,
                                loaded_records=0, slip_sum=0., slip_square_sum=0., slip_max=0.,
                                loaded_force_sum=0., force_times_slip_sum=0.,
                                penetration_sum=0., penetration_max=0.,
                                penetrating_records=0, penetrating_control_samples=0)
                      for side in self.foot_names}

    def sample(self, data):
        """Read one already-forwarded endpoint, update statistics, return records.

        This function never advances physics or reruns the solver. The caller
        must pass the model's corresponding ``MjData`` after ``mj_forward``.
        """
        records = {side: [] for side in self.foot_names}
        force = np.zeros(6)
        for index in range(data.ncon):
            contact = data.contact[index]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if g1 in self.ground_ids and g2 in self.geom_to_foot:
                side, foot_geom, ground_geom, direction = self.geom_to_foot[g2], g2, g1, 1.
            elif g2 in self.ground_ids and g1 in self.geom_to_foot:
                side, foot_geom, ground_geom, direction = self.geom_to_foot[g1], g1, g2, -1.
            else:
                continue
            mujoco.mj_contactForce(self.model, data, index, force)
            if not np.isfinite(force).all() or not math.isfinite(float(contact.dist)):
                raise FloatingPointError("Nonfinite physical contact force or distance")
            # MuJoCo contact-frame x is the compressive normal coordinate;
            # its normal points from geom1 to geom2. Keep a positive magnitude
            # for load, but reverse the world vector when the foot is geom1.
            if force[0] < -1e-7:
                raise ValueError("Unexpected tensile force in a unilateral MuJoCo contact")
            normal_force = max(0., float(force[0]))
            normal = np.asarray(contact.frame[:3], dtype=np.float64)
            if not np.isfinite(normal).all() or not np.isclose(np.linalg.norm(normal), 1., atol=1e-6):
                raise ValueError("Invalid MuJoCo contact normal")
            point = np.asarray(contact.pos, dtype=np.float64)
            foot_velocity = rigid_body_point_velocity(self.model, data, int(self.model.geom_bodyid[foot_geom]), point)
            ground_velocity = rigid_body_point_velocity(self.model, data, int(self.model.geom_bodyid[ground_geom]), point)
            relative = foot_velocity - ground_velocity
            tangent = relative - normal * np.dot(relative, normal)
            speed = float(np.linalg.norm(tangent))
            if not math.isfinite(speed):
                raise FloatingPointError("Nonfinite contact-point slip velocity")
            records[side].append(dict(geom_foot=self.model.geom(foot_geom).name,
                                      geom_ground=self.model.geom(ground_geom).name,
                                      position_world_m=point.tolist(), normal_force_n=normal_force,
                                      normal_force_world_n=(direction * normal_force * normal).tolist(),
                                      tangential_relative_speed_m_s=speed,
                                      penetration_m=max(0., -float(contact.dist)),
                                      loaded=normal_force > self.minimum_normal_force_n))
        self.sample_count += 1
        for side, contacts in records.items():
            stat = self.stats[side]
            loaded = [record for record in contacts if record["loaded"]]
            total_force = sum(record["normal_force_n"] for record in contacts)
            stat["force_sum"] += total_force
            stat["force_square_sum"] += total_force ** 2
            stat["force_max"] = max(stat["force_max"], total_force)
            stat["geometric_contact_samples"] += bool(contacts)
            stat["loaded_control_samples"] += bool(loaded)
            stat["geometric_records"] += len(contacts)
            stat["loaded_records"] += len(loaded)
            stat["penetrating_control_samples"] += any(record["penetration_m"] > 0 for record in contacts)
            for record in contacts:
                stat["normal_world_sum"] += record["normal_force_world_n"]
                stat["penetration_sum"] += record["penetration_m"]
                stat["penetration_max"] = max(stat["penetration_max"], record["penetration_m"])
                stat["penetrating_records"] += record["penetration_m"] > 0
            for record in loaded:
                speed, load = record["tangential_relative_speed_m_s"], record["normal_force_n"]
                stat["slip_sum"] += speed
                stat["slip_square_sum"] += speed ** 2
                stat["slip_max"] = max(stat["slip_max"], speed)
                stat["loaded_force_sum"] += load
                stat["force_times_slip_sum"] += load * speed
        return records

    def summary(self):
        result = {}
        for side, stat in self.stats.items():
            samples, contacts, loaded = self.sample_count, stat["geometric_records"], stat["loaded_records"]
            divide = lambda value, denominator: value / denominator if denominator else None
            force_mean = divide(stat["force_sum"], samples)
            force_square_mean = divide(stat["force_square_sum"], samples)
            slip_square_mean = divide(stat["slip_square_sum"], loaded)
            result[side] = dict(
                body=self.foot_names[side], collision_geoms=self.foot_geoms[side],
                geometric_contact_record_count=contacts, loaded_contact_record_count=loaded,
                geometric_contact_sample_fraction=divide(stat["geometric_contact_samples"], samples),
                loaded_contact_sample_fraction=divide(stat["loaded_control_samples"], samples),
                normal_force_n=dict(mean_over_all_samples=force_mean,
                                    rms_over_all_samples=math.sqrt(force_square_mean) if samples else None,
                                    max_sampled=stat["force_max"] if samples else None),
                normal_force_world_mean_n=(stat["normal_world_sum"] / samples).tolist() if samples else None,
                tangential_relative_speed_m_s=dict(mean_over_loaded_contact_records=divide(stat["slip_sum"], loaded),
                    rms_over_loaded_contact_records=math.sqrt(slip_square_mean) if loaded else None,
                    max_sampled=stat["slip_max"] if loaded else None,
                    normal_force_weighted_mean=divide(stat["force_times_slip_sum"], stat["loaded_force_sum"])),
                penetration_m=dict(max_sampled=stat["penetration_max"] if samples else None,
                    mean_over_geometric_contact_records=divide(stat["penetration_sum"], contacts),
                    penetrating_contact_record_count=stat["penetrating_records"],
                    penetrating_sample_fraction=divide(stat["penetrating_control_samples"], samples)))
        return dict(schema=1, source="MuJoCo physical contact records and mj_contactForce; no reference contact labels",
                    sampling="one post-mj_forward snapshot per control-step endpoint under the last applied substep torque",
                    control_period_s=self.control_dt_s, sample_count=self.sample_count,
                    ground_geoms=list(self.ground_names), minimum_loaded_normal_force_n=self.minimum_normal_force_n,
                    includes_substep_peaks=False, integrates_impulse_or_slip_distance=False,
                    no_contact_definition="Zero load and zero maximum penetration for sampled flight; conditional slip/contact-record means are null",
                    force_definition="Sum of compressive contact-normal magnitudes per foot; not total force including friction or necessarily world vertical",
                    feet=result)
