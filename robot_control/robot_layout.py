"""Pure-Python dimension and joint-order contracts for a future robot adapter.

This describes the *local v3 representation*, not the authors' unpublished
feature layout. It does not instantiate a network, simulator, or training job.
Changing DOF also changes learned layer shapes: a G1 policy cannot be converted
to Mojia by truncating action vectors or copying weights by array position.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, TypeVar

T = TypeVar("T")


def _positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class RobotLayout:
    """N-DOF schema of the currently implemented local v3 design.

    State: root xyz, wxyz quaternion, world linear velocity, body angular
    velocity, joint positions, joint velocities, previous executed actions.
    Context: friction, N calibration offsets, and 3 torso COM offsets.
    Goal: q, dq, height, relative rotation6D, local v/w, relative root xyz.
    Dynamics: local features -> independent root/joint pose and velocity
    increments; orientation is updated with a quaternion exponential. Quantities retain their existing physical units.
    """

    dof: int
    history: int = 10
    future: int = 10
    latent: int = 32

    def __post_init__(self) -> None:
        for name in ("dof", "history", "future", "latent"):
            _positive_integer(name, getattr(self, name))

    @property
    def state(self) -> int:
        return 13 + 3 * self.dof

    @property
    def observation(self) -> int:
        return 3 * self.dof + 6

    @property
    def context(self) -> int:
        return self.dof + 4

    @property
    def goal(self) -> int:
        return 2 * self.dof + 16

    @property
    def dynamics_input(self) -> int:
        return 14 + 5 * self.dof

    @property
    def dynamics_output(self) -> int:
        return 12 + 2 * self.dof

    @property
    def motion_encoder_input(self) -> int:
        return self.future * self.goal

    @property
    def velocity_encoder_input(self) -> int:
        return self.history * self.observation + 3

    @property
    def decoder_input(self) -> int:
        return self.history * self.observation + self.latent

    @property
    def decoder_output(self) -> int:
        return self.dof

    @property
    def critic_common_input(self) -> int:
        return self.state + self.context + self.history * self.observation + 5

    @property
    def motion_critic_input(self) -> int:
        return self.critic_common_input + self.motion_encoder_input

    @property
    def velocity_critic_input(self) -> int:
        return self.critic_common_input + 3

    @property
    def state_slices(self) -> dict[str, slice]:
        n = self.dof
        return {
            "root_position": slice(0, 3),
            "root_quaternion_wxyz": slice(3, 7),
            "root_linear_velocity_world": slice(7, 10),
            "root_angular_velocity_body": slice(10, 13),
            "joint_position": slice(13, 13 + n),
            "joint_velocity": slice(13 + n, 13 + 2 * n),
            "previous_executed_action": slice(13 + 2 * n, 13 + 3 * n),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible dimensions and half-open state spans."""
        dimensions = (
            "state", "observation", "context", "goal", "dynamics_input",
            "dynamics_output", "motion_encoder_input", "velocity_encoder_input",
            "decoder_input", "decoder_output", "critic_common_input",
            "motion_critic_input", "velocity_critic_input",
        )
        return {
            "schema": "mojia_local_increment_v1",
            "dof": self.dof,
            "history": self.history,
            "future": self.future,
            "latent": self.latent,
            "dimensions": {key: getattr(self, key) for key in dimensions},
            "state_spans": {
                key: [span.start, span.stop]
                for key, span in self.state_slices.items()
            },
        }


def _joint_names(label: str, values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{label} must be a sequence of joint names")
    names = tuple(values)
    if not names or any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError(f"{label} must contain nonempty joint names")
    if len(set(names)) != len(names):
        raise ValueError(f"{label} contains duplicate joint names")
    return names


@dataclass(frozen=True, init=False)
class ExplicitJointOrder:
    """A checked bijection between two complete named joint orders.

    ``target_to_source`` explicitly maps every target name to a source name
    when aliases differ. Without it the names must match exactly. Inputs of
    differing cardinality, partial aliases, unknown names, duplicate names and
    repeated source assignments are rejected. This is an ordering operation,
    not retargeting, joint-limit conversion, or G1-to-Mojia pose transfer.
    """

    source_names: tuple[str, ...]
    target_names: tuple[str, ...]
    target_source_indices: tuple[int, ...]
    source_target_indices: tuple[int, ...]

    def __init__(
        self,
        source_names: Sequence[str],
        target_names: Sequence[str],
        *,
        target_to_source: Mapping[str, str] | None = None,
    ) -> None:
        source = _joint_names("source_names", source_names)
        target = _joint_names("target_names", target_names)
        if len(source) != len(target):
            raise ValueError("Joint counts differ; truncation is forbidden; retargeting is required")
        aliases = dict(target_to_source) if target_to_source is not None else {
            name: name for name in target
        }
        if set(aliases) != set(target):
            raise ValueError("target_to_source must explicitly cover exactly all target joints")
        alias_values = tuple(aliases[name] for name in target)
        if any(not isinstance(name, str) for name in alias_values):
            raise ValueError("Every mapped source joint must be a joint-name string")
        if set(alias_values) != set(source) or len(set(alias_values)) != len(source):
            raise ValueError("Mapping must be a bijection over all source joints; check missing/unknown names")
        indices = {name: index for index, name in enumerate(source)}
        forward = tuple(indices[name] for name in alias_values)
        inverse = [0] * len(source)
        for target_index, source_index in enumerate(forward):
            inverse[source_index] = target_index
        object.__setattr__(self, "source_names", source)
        object.__setattr__(self, "target_names", target)
        object.__setattr__(self, "target_source_indices", forward)
        object.__setattr__(self, "source_target_indices", tuple(inverse))

    def to_target(self, source_values: Sequence[T]) -> tuple[T, ...]:
        if len(source_values) != len(self.source_names):
            raise ValueError("Values must contain the entire source joint vector")
        return tuple(source_values[index] for index in self.target_source_indices)

    def to_source(self, target_values: Sequence[T]) -> tuple[T, ...]:
        if len(target_values) != len(self.target_names):
            raise ValueError("Values must contain the entire target joint vector")
        return tuple(target_values[index] for index in self.source_target_indices)

    def rows_to_target(self, rows: Sequence[Sequence[T]]) -> tuple[tuple[T, ...], ...]:
        return tuple(self.to_target(row) for row in rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_names": list(self.source_names),
            "target_names": list(self.target_names),
            "target_source_indices": list(self.target_source_indices),
            "source_target_indices": list(self.source_target_indices),
            "target_to_source": {
                name: self.source_names[index]
                for name, index in zip(self.target_names, self.target_source_indices)
            },
        }

