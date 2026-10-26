"""CPU-backed, bounded vector replay for continuous physics collection.

Only sampled tensors are copied to the learner device.  Each world has a ring
of transitions and a slightly longer ring of noisy observations.  The extra
``history - 1`` observations preserve exact history even for the oldest retained
transition.  Initial histories default to repetition of the reset observation;
arbitrary reset prefixes can be supplied explicitly and are retained separately.

Windows include short failures and the newest transition, with an explicit mask.
They never join different episodes, contexts, clips, or controller interfaces.
"""
from copy import deepcopy

import numpy as np
import torch


class ReplayV3:
    VERSION = 1
    # State contains qpos/qvel and last executed action; histories contain the
    # actual noisy observations, not observations regenerated from those states.
    FIELDS = {
        "states": ((100,), np.float32),
        "next_states": ((100,), np.float32),
        "actions": ((29,), np.float32),
        "contexts": ((33,), np.float32),
        "clips": ((), np.int32),
        "frames": ((), np.int32),
        "interfaces": ((), np.int16),
        "commands": ((3,), np.float32),
        "next_commands": ((3,), np.float32),
        "command_elapsed": ((), np.int32),
        "reference_commands": ((), np.bool_),
        "terminated": ((), np.bool_),
        "truncated": ((), np.bool_),
        "clip_end": ((), np.bool_),
        "episode_ids": ((), np.int64),
        "policy_versions": ((), np.int64),
        "episode_starts": ((), np.int64),
        "rewards": ((), np.float32),
    }

    def __init__(self, capacity=4_096_000, num_envs=4096, history=10, seed=0):
        if int(capacity) < int(num_envs) or int(num_envs) < 1 or int(history) < 1:
            raise ValueError("capacity >= num_envs >= 1 and history >= 1 required")
        self.requested_capacity = int(capacity)
        self.num_envs = int(num_envs)
        self.history_length = int(history)
        self.slots = self.requested_capacity // self.num_envs
        self.capacity = self.slots * self.num_envs
        self.observation_slots = self.slots + self.history_length - 1
        self.steps = 0
        self.rng = np.random.default_rng(seed)
        self._data = {
            name: np.empty((self.num_envs, self.slots, *shape), dtype=dtype)
            for name, (shape, dtype) in self.FIELDS.items()
        }
        self._observations = np.empty(
            (self.num_envs, self.observation_slots, 93), dtype=np.float32)
        self._prefixes = {}
        self._interface_cache = {}

    def __len__(self):
        return self.num_envs * min(self.steps, self.slots)

    @classmethod
    def estimate_bytes(cls, capacity=4_096_000, num_envs=4096, history=10,
                       custom_prefixes=0):
        """Allocated array bytes; no GPU cache or full-history replication.

        An explicitly supplied, nontrivial reset prefix costs (history-1)*93*4
        extra bytes per retained episode.  The usual repeated reset observation
        needs no prefix allocation.  Interface sampling caches cost at most one
        int64 per retained transition per requested interface (disjoint indices).
        """
        if num_envs < 1 or capacity < num_envs or history < 1:
            raise ValueError("Invalid replay dimensions")
        actual = (int(capacity) // int(num_envs)) * int(num_envs)
        per_transition = sum(int(np.prod(shape)) * np.dtype(dtype).itemsize
                             for shape, dtype in cls.FIELDS.values())
        return {
            "capacity": actual,
            "transition_bytes": actual * per_transition,
            "observation_bytes": (actual + num_envs*(history-1))*93*4,
            "custom_prefix_bytes": int(custom_prefixes)*(history-1)*93*4,
            "maximum_interface_cache_bytes": actual*8,
            "total_array_bytes": actual*per_transition
            + (actual + num_envs*(history-1))*93*4
            + int(custom_prefixes)*(history-1)*93*4,
        }

    def _array(self, name, value, shape, dtype):
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        value = np.asarray(value)
        if value.shape == () and shape == ():
            value = np.full((self.num_envs,), value.item())
        expected = (self.num_envs, *shape)
        if value.shape != expected:
            raise ValueError(f"{name} has shape {value.shape}; expected {expected}")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains nonfinite values")
        target = np.dtype(dtype)
        if np.issubdtype(target, np.integer) and not np.issubdtype(target, np.bool_):
            bounds = np.iinfo(target)
            if (value < bounds.min).any() or (value > bounds.max).any() or (value != np.floor(value)).any():
                raise ValueError(f"{name} cannot be represented as {target}")
        if target == np.bool_ and not np.isin(value, (0, 1)).all():
            raise ValueError(f"{name} must be a boolean mask")
        return value.astype(target, copy=False)

    def add_batch(self, states, actions, next_states, observations,
                  next_observations, contexts, clips, frames, interfaces,
                  commands, next_commands, reference_commands, terminated,
                  truncated, clip_end, episode_ids, policy_version, *,
                  initial_histories=None, rewards=None, command_elapsed=None):
        """Append one simultaneous control step from *every* environment.

        Episode IDs must increase on reset.  Context, clip, and interface cannot
        change inside an episode.  ``next_states`` always contain the real final
        physics state, including on failure; reset states belong to the next call.
        ``next_observations`` is checked but need not be stored: model fitting
        uses physical states, and every sampled start history uses the recorded
        current observations.  No independently sampled noise is regenerated.

        If a collector initializes history differently from repeated first obs,
        pass ``initial_histories[E,history,93]`` at reset (nonreset rows ignored).
        """
        supplied = dict(states=states, actions=actions, next_states=next_states,
                        contexts=contexts, clips=clips, frames=frames,
                        interfaces=interfaces, commands=commands,
                        next_commands=next_commands,
                        reference_commands=reference_commands,
                        terminated=terminated, truncated=truncated,
                        clip_end=clip_end, episode_ids=episode_ids,
                        policy_versions=policy_version,
                        command_elapsed=0 if command_elapsed is None else command_elapsed,
                        rewards=0. if rewards is None else rewards)
        batch = {name: self._array(name, value, *self.FIELDS[name])
                 for name, value in supplied.items()}
        if (batch["command_elapsed"] < 0).any():
            raise ValueError("command_elapsed must be nonnegative")
        obs = self._array("observations", observations, (93,), np.float32)
        self._array("next_observations", next_observations, (93,), np.float32)
        if np.logical_and(batch["terminated"], batch["truncated"]).any():
            raise ValueError("A transition cannot be both terminated and truncated")
        previous = (self.steps - 1) % self.slots
        if self.steps:
            old_ids = self._data["episode_ids"][:, previous]
            if (batch["episode_ids"] < old_ids).any():
                raise ValueError("Episode IDs must increase on reset")
            reset = batch["episode_ids"] != old_ids
            old_ends = (self._data["terminated"][:, previous]
                        | self._data["truncated"][:, previous]
                        | self._data["clip_end"][:, previous])
            if (old_ends & ~reset).any():
                raise ValueError("Ended environments must start a new episode")
            for name in ("contexts", "clips", "interfaces"):
                changed = batch[name] != self._data[name][:, previous]
                if changed.ndim > 1:
                    changed = changed.any(-1)
                if (changed & ~reset).any():
                    raise ValueError(f"{name} changed without an episode reset")
            batch["episode_starts"] = np.where(
                reset, self.steps, self._data["episode_starts"][:, previous])
        else:
            reset = np.ones(self.num_envs, dtype=bool)
            batch["episode_starts"] = np.zeros(self.num_envs, dtype=np.int64)
        prefixes = {}
        if initial_histories is not None:
            hist = self._array("initial_histories", initial_histories,
                               (self.history_length, 93), np.float32)
            if not np.array_equal(hist[reset, -1], obs[reset]):
                raise ValueError("Initial history must end at the current observation")
            for env in np.flatnonzero(reset):
                if not np.array_equal(hist[env], np.broadcast_to(obs[env], hist[env].shape)):
                    prefixes[(int(env), int(batch["episode_ids"][env]))] = (
                        self.steps, hist[env, :-1].copy())
        slot = self.steps % self.slots
        for name, value in batch.items():
            self._data[name][:, slot] = value
        self._observations[:, self.steps % self.observation_slots] = obs
        self._prefixes.update(prefixes)
        self.steps += 1
        oldest = max(0, self.steps - self.slots)
        # Only early starts of a retained episode need its explicit reset prefix.
        self._prefixes = {key: value for key, value in self._prefixes.items()
                          if value[0] + self.history_length - 2 >= oldest}
        self._interface_cache.clear()

    def _draw(self, batch, interface):
        batch = int(batch)
        if batch < 1 or not len(self):
            raise ValueError("A positive batch and nonempty replay are required")
        filled = min(self.steps, self.slots)
        if interface is None:
            flat = self.rng.integers(0, self.num_envs*filled, size=batch)
        else:
            key = int(interface)
            if key not in self._interface_cache:
                self._interface_cache[key] = np.flatnonzero(
                    self._data["interfaces"][:, :filled].reshape(-1) == key)
            candidates = self._interface_cache[key]
            if not len(candidates):
                raise ValueError(f"No transitions for interface {key}")
            flat = candidates[self.rng.integers(0, len(candidates), size=batch)]
        envs, slots = flat // filled, flat % filled
        steps = self.steps - 1 - ((self.steps - 1 - slots) % self.slots)
        return envs, slots, steps

    def _history(self, envs, slots, steps):
        starts = self._data["episode_starts"][envs, slots]
        times = steps[:, None] + np.arange(1-self.history_length, 1)
        safe_times = np.maximum(times, starts[:, None])
        result = self._observations[
            envs[:, None], safe_times % self.observation_slots].copy()
        episodes = self._data["episode_ids"][envs, slots]
        prefix_rows = (np.flatnonzero((times < starts[:, None]).any(-1))
                       if self._prefixes else ())
        for row in prefix_rows:
            prefix = self._prefixes.get((int(envs[row]), int(episodes[row])))
            if prefix is not None:
                missing = times[row] < starts[row]
                indices = times[row, missing] - starts[row] + self.history_length - 1
                result[row, missing] = prefix[1][indices]
        return result

    @staticmethod
    def _tensors(result, device):
        # No persistent GPU storage.  NumPy indexing has already copied batches.
        return {name: torch.as_tensor(value, device=device)
                for name, value in result.items()}

    def sample_starts(self, batch, device="cpu", interface=None):
        envs, slots, steps = self._draw(batch, interface)
        result = {name: values[envs, slots] for name, values in self._data.items()}
        result.update(history=self._history(envs, slots, steps),
                      env_ids=envs, steps=steps)
        return self._tensors(result, device)

    def sample_windows(self, batch, horizon=8, device="cpu", interface=None):
        horizon = int(horizon)
        if horizon < 1:
            raise ValueError("horizon must be positive")
        envs, slots, steps = self._draw(batch, interface)
        times = steps[:, None] + np.arange(horizon)
        indices = times % self.slots
        gathered = {name: values[envs[:, None], indices]
                    for name, values in self._data.items()}
        same_episode = gathered["episode_ids"] == self._data["episode_ids"][envs, slots, None]
        same_interface = gathered["interfaces"] == self._data["interfaces"][envs, slots, None]
        valid = (times < self.steps) & same_episode & same_interface
        ends = gathered["terminated"] | gathered["truncated"] | gathered["clip_end"]
        # Include the transition that ends the episode; exclude everything after.
        prior_end = np.concatenate((np.zeros((int(batch), 1), dtype=bool),
                                    ends[:, :-1]), axis=1)
        valid &= ~np.maximum.accumulate(prior_end, axis=1)
        valid = np.minimum.accumulate(valid, axis=1)
        lengths = valid.sum(-1)
        after = np.minimum(np.arange(horizon)[None, :], lengths[:, None]-1)
        last_next = gathered["next_states"][np.arange(int(batch))[:, None], after]
        states = np.concatenate((self._data["states"][envs, slots, None], last_next), axis=1)
        # Do not expose uninitialized future slots, even when the caller masks
        # loss correctly.  0*NaN is still NaN; repeat the final recorded metadata.
        result = {name: value[np.arange(int(batch))[:, None], after]
                  for name, value in gathered.items()}
        result["states"] = states
        result["actions"] = np.where(valid[..., None], gathered["actions"], 0.)
        result["mask"] = valid
        result["contexts"] = self._data["contexts"][envs, slots]
        result["history"] = self._history(envs, slots, steps)
        result["env_ids"] = envs
        result["steps"] = steps
        result["lengths"] = lengths
        return self._tensors(result, device)

    def state_dict(self):
        """Snapshot initialized slots using weights_only-safe CPU tensors.

        NumPy arrays in a checkpoint require the unrestricted pickle loader.
        Copying them into CPU tensors keeps this trusted-local checkpoint usable
        with torch.load(..., weights_only=True), including custom reset prefixes.
        """
        filled = min(self.steps, self.slots)
        observed = min(self.steps, self.observation_slots)
        return dict(version=self.VERSION, capacity=self.requested_capacity,
                    num_envs=self.num_envs, history=self.history_length,
                    steps=self.steps, rng=deepcopy(self.rng.bit_generator.state),
                    arrays={name: torch.from_numpy(value[:, :filled].copy())
                            for name, value in self._data.items()},
                    observations=torch.from_numpy(self._observations[:, :observed].copy()),
                    prefixes={key: (start, torch.from_numpy(value.copy()))
                              for key, (start, value) in self._prefixes.items()})

    def load_state_dict(self, state):
        expected = (self.VERSION, self.requested_capacity, self.num_envs, self.history_length)
        got = (state["version"], state["capacity"], state["num_envs"], state["history"])
        if expected != got:
            raise ValueError(f"Incompatible replay dimensions/version: {got} != {expected}")
        steps = int(state["steps"])
        if steps < 0:
            raise ValueError("Invalid checkpoint step count")
        filled, observed = min(steps, self.slots), min(steps, self.observation_slots)
        for name, value in self._data.items():
            restored = np.asarray(state["arrays"][name])
            if restored.shape != value[:, :filled].shape or restored.dtype != value.dtype:
                raise ValueError(f"Invalid checkpoint array {name}")
            value[:, :filled] = restored
        observations = np.asarray(state["observations"])
        if observations.shape != (self.num_envs, observed, 93) or observations.dtype != np.float32:
            raise ValueError("Invalid checkpoint observations")
        self._observations[:, :observed] = observations
        self._prefixes = {key: (start, np.asarray(value).copy())
                          for key, (start, value) in state["prefixes"].items()}
        self.rng.bit_generator.state = deepcopy(state["rng"])
        self.steps = steps
        self._interface_cache.clear()
