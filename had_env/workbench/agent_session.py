"""Direct two-team flight control, including heterogeneous native HAD rosters.

This is a separate protocol from the grouping benchmark. The native reward and
full relative observation encoding are preserved. A caller-imposed horizon is
a sampling truncation, not an invented win. Functional actions remain ruled.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import uuid

import numpy as np

from .identity import source_identity
from .recording import EpisodeRecorder, serializable
from .session import PolicyAdapter


class _FlightObservation:
    def __init__(self, value):
        self.value = value

    def state(self):
        return self.value


@dataclass(frozen=True)
class FlightScenarioSpec:
    red_attackers: int = 4
    blue_attackers: int = 4
    targets: int = 2
    red_scouts: int = 0
    red_disturbers: int = 0
    blue_scouts: int = 0
    blue_disturbers: int = 0
    seed: int = 20260907
    max_steps: int = 50
    action_mode: str = "discrete27"

    def __post_init__(self):
        for name in ("red_attackers", "blue_attackers", "targets", "max_steps"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("red_scouts", "red_disturbers", "blue_scouts", "blue_disturbers"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.action_mode not in ("discrete27", "continuous_native"):
            raise ValueError("Unknown flight action mode")


@dataclass
class FlightTransition:
    observations: dict
    rewards: dict
    terminated: bool
    truncated: bool
    outcome_red: int
    info: dict
    delta: int = 1


class FlightSession:
    def __init__(self, scenario=None, *, red_policy=None, blue_policy=None, record=True):
        from had_env.core.make_env import HADEnv
        from had_env.core.config import Interval
        from had_env.core.version import CORE_VERSION, PHYSICS_PROTOCOL
        self.scenario = scenario or FlightScenarioSpec()
        if isinstance(self.scenario, dict):
            self.scenario = FlightScenarioSpec(**self.scenario)
        s = self.scenario
        self.env = HADEnv(s.red_attackers, s.blue_attackers, s.targets,
                          red_scout_n=s.red_scouts, red_disturb_n=s.red_disturbers,
                          blue_scout_n=s.blue_scouts, blue_disturb_n=s.blue_disturbers, seed=s.seed)
        self.env.reset(seed=s.seed)
        self.env.record_events = bool(record)
        self.step_count, self.done, self._closed = 0, False, False
        self._numpy_state = np.random.RandomState(s.seed).get_state()
        self.policies = {}
        for index, (side, policy) in enumerate((("red", red_policy), ("blue", blue_policy))):
            if hasattr(policy, "act_env"):
                raise ValueError("Native flight policies use act(observation), not the grouping act_env interface")
            self.policies[side] = PolicyAdapter(policy, seed=s.seed + index) if policy is not None else None
        self.recorder = EpisodeRecorder(dict(
            episode_id=uuid.uuid4().hex, protocol_id=f"had-native-flight-{s.action_mode}-v1",
            physics_version=CORE_VERSION, physics_protocol=PHYSICS_PROTOCOL, dt=Interval,
            scenario=asdict(s), policies={side: p.name if p is not None else "zero_acceleration"
                                         for side, p in self.policies.items()},
            observation_protocol="native_full_relative_legacy_normalization",
            reward_protocol="native_RealReward_and_LatentReward",
            horizon_semantics="sampling_truncation_without_awarded_win",
            source_identity=source_identity()))
        self.recorder.append_frame(self._frame())

    def observations(self):
        values = self.env.get_observation(is_relative_observation=True)
        result, index = {}, 0
        for side in ("red", "blue"):
            agents = getattr(self.env, f"{side}_agents")
            result[side] = dict(observation=copy.deepcopy(values[index:index + len(agents)]),
                                agent_ids=[a.Id for a in agents], alive_mask=[a.Health > 0 for a in agents])
            index += len(agents)
        return result

    def _decode(self, side, values):
        from had_env.actions import ACCELERATION_PRIMITIVES
        agents = getattr(self.env, f"{side}_agents")
        if isinstance(values, dict):
            values = {int(k): v for k, v in values.items()}
            if set(values) != {a.Id for a in agents}:
                raise ValueError(f"{side} action IDs must match the fixed roster")
            values = [values[a.Id] for a in agents]
        if values is None:
            values = [0] * len(agents) if self.scenario.action_mode == "discrete27" else np.zeros((len(agents), 3))
        array = np.asarray(values)
        if self.scenario.action_mode == "discrete27":
            if array.shape != (len(agents),) or not np.issubdtype(array.dtype, np.integer):
                raise ValueError(f"{side} requires one integer action per agent")
            if np.any(array < 0) or np.any(array >= len(ACCELERATION_PRIMITIVES)):
                raise ValueError("Flight action index outside 0..26")
            return ACCELERATION_PRIMITIVES[array].tolist()
        if array.shape != (len(agents), 3) or not np.isfinite(array.astype(float)).all():
            raise ValueError(f"{side} requires finite Nx3 native accelerations")
        return array.astype(float).tolist()

    def step(self, actions=None):
        if self.done or self._closed:
            raise RuntimeError("Cannot step a completed or closed flight session")
        ambient = np.random.get_state()
        np.random.set_state(self._numpy_state)
        try:
            before = self.observations()
            chosen = dict(actions or {})
            if set(chosen) - {"red", "blue"}:
                raise ValueError("Actions must be keyed by red/blue")
            for side, policy in self.policies.items():
                if side not in chosen:
                    chosen[side] = policy.act(_FlightObservation(before[side]), planning_seed=self.scenario.seed) if policy is not None else None
            # Validate BOTH teams before modifying any physical state.
            decoded = {side: self._decode(side, chosen[side]) for side in ("red", "blue")}
            _, _, _, rewards, _, info = self.env.step(decoded["red"] + decoded["blue"])
            self.step_count += 1
            outcome = int(self.env.is_terminal())
            terminated = bool(outcome)
            truncated = self.step_count >= self.scenario.max_steps and not terminated
            self.done = terminated or truncated
            actions_record = {side: {str(a.Id): value for a, value in
                                     zip(getattr(self.env, f"{side}_agents"), decoded[side])}
                              for side in ("red", "blue")}
            self.recorder.append_decision(dict(step=self.step_count - 1, selected_plan=None,
                                               actions=serializable(chosen), trace={},
                                               action_mode=self.scenario.action_mode))
            frame = self._frame(actions_record)
            if self.done:
                frame["events"].append(dict(kind="terminal" if terminated else "truncation", step=self.step_count))
            self.recorder.append_frame(frame)
            reason = "target_destroyed" if outcome < 0 else "blue_attackers_destroyed" if outcome > 0 else "sampling_horizon" if truncated else "running"
            if self.done:
                self.recorder.finish(dict(outcome_red=outcome, success_native=outcome > 0 if terminated else None,
                                          termination_reason=reason, terminated=terminated, truncated=truncated,
                                          physical_steps=self.step_count))
            return FlightTransition(self.observations(), rewards, terminated, truncated, outcome,
                                    {**info, "events": frame["events"], "event_reason": reason})
        finally:
            self._numpy_state = copy.deepcopy(np.random.get_state())
            np.random.set_state(ambient)

    def _frame(self, actions=None):
        from had_env.core.config import Interval, initial_health, AttackDistance
        rows = []
        for side in ("red", "blue", "targets"):
            entities = self.env.targets if side == "targets" else getattr(self.env, f"{side}_agents")
            for index, e in enumerate(entities):
                rows.append(dict(id=index if side == "targets" else e.Id, entity_id=e.Id, side=side, role=e.Type,
                                 position=list(e.position), velocity=list(e.velocity), health=float(e.Health),
                                 alive=bool(e.Health > 0), max_health=initial_health if side == "targets" else 1.,
                                 attack_range=max(AttackDistance) if e.Type == "Attack" else None))
        return dict(step=self.step_count, sim_time=self.step_count * Interval, entities=rows,
                    groups={"red": [], "blue": []}, assignments={"red": {}, "blue": {}},
                    events=copy.deepcopy(self.env.last_physics_events), actions=actions or {"red": {}, "blue": {}})

    def run(self, max_decisions=None):
        count = 0
        while not self.done and (max_decisions is None or count < max_decisions):
            self.step()
            count += 1
        return self.recorder.episode

    def snapshot(self):
        return copy.deepcopy(dict(scenario=asdict(self.scenario), entities=[e.__dict__ for e in self.env.world],
                                  rng=self.env.np_random.bit_generator.state, numpy=self._numpy_state,
                                  step=self.step_count, done=self.done, policies=self.policies,
                                  episode=self.recorder.episode))

    def branch(self, continuation_seed=None):
        if self.done or self._closed:
            raise RuntimeError("Cannot branch a terminal or closed flight session")
        saved = self.snapshot()
        following = FlightSession(self.scenario)
        for e, state in zip(following.env.world, saved["entities"]):
            e.__dict__.clear()
            e.__dict__.update(state)
        following.env.np_random.bit_generator.state = saved["rng"]
        following._numpy_state = saved["numpy"]
        following.step_count = following.env.physics_step_count = saved["step"]
        following.done, following.policies = saved["done"], saved["policies"]
        following.env.update_alive_agents()
        following.recorder.episode = saved["episode"]
        metadata = following.recorder.episode.metadata
        metadata["parent_episode_id"] = metadata["episode_id"]
        metadata["episode_id"] = uuid.uuid4().hex
        metadata["branch_step"] = self.step_count
        if continuation_seed is not None:
            following.env.np_random = np.random.default_rng(continuation_seed)
            following._numpy_state = np.random.RandomState(continuation_seed).get_state()
            metadata["continuation_seed"] = continuation_seed
        return following

    def close(self):
        self.env.close()
        self._closed = True
