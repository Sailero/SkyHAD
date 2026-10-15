"""Shared-world adapter that executes grouping assignments in the native simulator.

Assignments control navigation only: collision, firing, damage and native
outcomes stay in one shared physical world. Agent IDs are global HAD entity
IDs; targets use stable zero-based indices. Blue's lower-level rule is
``rush``: each assigned attacker flies the discrete acceleration closest to
the direction of its target, and unassigned attackers coast.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from had_env.actions import ACCELERATION_PRIMITIVES

AssignmentInput = Union[Mapping[int, Optional[int]], Sequence[Optional[int]]]
ActionInput = Union[Mapping[int, int], Sequence[int]]


@dataclass(frozen=True)
class GroupingEvent:
    """One event observed at a command boundary."""

    kind: str
    step: int
    side: Optional[str] = None
    agent_ids: Tuple[int, ...] = ()
    target_ids: Tuple[int, ...] = ()
    terminal: bool = False

    def as_dict(self) -> Dict[str, object]:
        return {"kind": self.kind, "step": int(self.step), "side": self.side,
                "agent_ids": list(self.agent_ids), "target_ids": list(self.target_ids),
                "terminal": bool(self.terminal)}


@dataclass(frozen=True)
class GroupingSnapshot:
    """Exact in-memory branch point, including assignments and random streams."""

    step_count: int
    entity_states: Tuple[Mapping[str, object], ...]
    red_assignment: Tuple[Tuple[int, Optional[int]], ...]
    blue_assignment: Tuple[Tuple[int, Optional[int]], ...]
    environment_rng_state: Mapping[str, object]
    adapter_rng_state: Mapping[str, object]
    numpy_random_state: Tuple[object, ...]
    last_events: Tuple[GroupingEvent, ...]
    task_mode: str
    target_health: float
    physics_protocol: str
    max_steps: int
    horizon_policy: str
    last_physics_events: Tuple[Mapping[str, object], ...]
    spatial_dim: int
    plane_altitude: float
    target_initialization: str
    env_agent_type: str
    env_agent_action_type: str
    effective_config: Mapping[str, object]


class GroupingAdapter:
    """Shared multi-target world commanded by Red and Blue target assignments.

    Red acceleration actions come from the caller (normally ``RuleExecutor``);
    Blue actions come from the known ``rush`` rule. Native semantics remain:
    firing attackers self-destruct, destroying any target is a Blue win, and
    destroying every Blue attacker is a Red win (survival task).
    """

    def __init__(self, red_attackers: int, blue_attackers: int, targets: int, *,
                 max_steps: int = 100, target_region=None, target_positions=None,
                 target_initialization: str = "random", horizon_policy: str = "red_win",
                 task_mode: str = "survival", target_health: Optional[float] = None,
                 spatial_dim: int = 2, plane_altitude: Optional[float] = None, effective_config=None) -> None:
        if red_attackers < 1 or blue_attackers < 1:
            raise ValueError("grouping requires at least one agent on each side")
        if targets < 1:
            raise ValueError("grouping requires at least one target")
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        if horizon_policy not in ("red_win", "draw", "blue_win"):
            raise ValueError("horizon_policy must be red_win, draw, or blue_win")
        from had_env.config import EnvConfig
        from had_env.simulation import Simulation
        effective_config = effective_config or EnvConfig(spatial_dim=spatial_dim, task_mode=task_mode,
                                                         target_region=target_region, plane_altitude=plane_altitude)
        self.env = Simulation(red_attackers, blue_attackers, targets, target_region=target_region,
                              target_initialization=target_initialization, target_positions=target_positions,
                              task_mode=task_mode, target_health=target_health, spatial_dim=spatial_dim,
                              plane_altitude=plane_altitude, effective_config=effective_config)
        self.env_agent_type = self.env.env_agent_type
        self.env_agent_action_type = self.env.env_agent_action_type
        self.effective_config = self.env.effective_config
        self.scene_scale = self.env.scene_scale
        self.task_mode = self.env.task_mode
        self.target_health = self.env.target_health
        self.spatial_dim = self.env.spatial_dim
        self.plane_altitude = self.env.plane_altitude
        self.target_initialization = self.env.target_initialization
        self.fixed_target_positions = self.env.fixed_target_positions
        self.valid_action_ids = tuple(int(i) for i in range(len(ACCELERATION_PRIMITIVES))
                                      if self.spatial_dim == 3 or ACCELERATION_PRIMITIVES[i, 2] == 0)
        self.max_steps = int(max_steps)
        self.horizon_policy = str(horizon_policy)
        self.step_count = 0
        self.rng = np.random.default_rng()
        self._red_assignment: Dict[int, Optional[int]] = {}
        self._blue_assignment: Dict[int, Optional[int]] = {}
        self.last_events: Tuple[GroupingEvent, ...] = ()

    @property
    def red_ids(self) -> Tuple[int, ...]:
        return tuple(int(agent.Id) for agent in self.env.red_agents)

    @property
    def blue_ids(self) -> Tuple[int, ...]:
        return tuple(int(agent.Id) for agent in self.env.blue_agents)

    @property
    def target_ids(self) -> Tuple[int, ...]:
        return tuple(range(len(self.env.targets)))

    @property
    def red_assignment(self) -> Dict[int, Optional[int]]:
        return dict(self._red_assignment)

    @property
    def blue_assignment(self) -> Dict[int, Optional[int]]:
        return dict(self._blue_assignment)

    @property
    def assignments(self) -> Dict[str, Dict[int, Optional[int]]]:
        return {"Red": self.red_assignment, "Blue": self.blue_assignment}

    def _agents(self, side: str):
        if side not in ("Red", "Blue"):
            raise ValueError("side must be Red or Blue")
        return self.env.red_agents if side == "Red" else self.env.blue_agents

    def _assignment(self, side: str) -> Dict[int, Optional[int]]:
        if side not in ("Red", "Blue"):
            raise ValueError("side must be Red or Blue")
        return self._red_assignment if side == "Red" else self._blue_assignment

    def reset(self, seed: Optional[int] = None, *, target_positions=None,
              red_assignment: Optional[AssignmentInput] = None,
              blue_assignment: Optional[AssignmentInput] = None) -> Dict[str, object]:
        """Reset the world; without explicit assignments, IDs are spread round-robin over targets."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        positions = self.fixed_target_positions if target_positions is None else target_positions
        if positions is not None:
            positions = np.asarray(positions, dtype=np.float64).copy()
            bounds = self.env.world_bounds
            if positions.shape != (len(self.env.targets), 3) or not np.all(np.isfinite(positions)):
                raise ValueError("target_positions must have shape [targets, 3]")
            if np.any(positions < bounds[:, 0]) or np.any(positions > bounds[:, 1]):
                raise ValueError("target_positions must stay inside the HAD world")
            if self.spatial_dim == 2:
                positions[:, 2] = self.plane_altitude
            self.env.fixed_target_positions, self.env.target_initialization = positions, 'fixed'
        else:
            self.env.fixed_target_positions, self.env.target_initialization = None, self.target_initialization
        self.env.reset(seed=seed)
        self.step_count = 0
        targets = self.target_ids
        self._red_assignment = {agent_id: targets[i % len(targets)] for i, agent_id in enumerate(self.red_ids)}
        self._blue_assignment = {agent_id: targets[i % len(targets)] for i, agent_id in enumerate(self.blue_ids)}
        if red_assignment is not None:
            self.set_assignments("Red", red_assignment, record_event=False)
        if blue_assignment is not None:
            self.set_assignments("Blue", blue_assignment, record_event=False)
        self.last_events = ()
        return self.global_state()

    def set_assignments(self, side: str, assignment: AssignmentInput, *,
                        record_event: bool = True) -> Tuple[GroupingEvent, ...]:
        """Replace one side's allocation; ``None`` (or an omitted ID) is an uncommitted reserve."""
        ids = tuple(int(agent.Id) for agent in self._agents(side))
        if isinstance(assignment, Mapping):
            unknown = set(int(key) for key in assignment) - set(ids)
            if unknown:
                raise ValueError(f"assignment contains non-{side} IDs: {sorted(unknown)}")
            candidate = {agent_id: assignment.get(agent_id) for agent_id in ids}
        else:
            values = list(assignment)
            if len(values) != len(ids):
                raise ValueError(f"{side} assignment must contain {len(ids)} entries")
            candidate = dict(zip(ids, values))
        normalised: Dict[int, Optional[int]] = {}
        for agent_id, target_id in candidate.items():
            if target_id is None:
                normalised[agent_id] = None
                continue
            target_index = int(target_id)
            if target_index not in self.target_ids:
                raise ValueError(f"target_id must be in {self.target_ids}")
            if self.env.targets[target_index].Health <= 0:
                raise ValueError("cannot assign an agent to a destroyed target")
            normalised[agent_id] = target_index
        old = dict(self._assignment(side))
        changed_ids = tuple(agent_id for agent_id in ids if old.get(agent_id) != normalised[agent_id])
        if side == "Red":
            self._red_assignment = normalised
        else:
            self._blue_assignment = normalised
        if not changed_ids:
            return ()
        targets = tuple(sorted({value for agent_id in changed_ids
                                for value in (old.get(agent_id), normalised[agent_id]) if value is not None}))
        event = GroupingEvent(kind="assignment_changed", step=self.step_count, side=side,
                              agent_ids=changed_ids, target_ids=targets)
        if record_event:
            self.last_events = (event,)
        return (event,)

    def set_joint_assignments(self, red_assignment: AssignmentInput,
                              blue_assignment: AssignmentInput) -> Tuple[GroupingEvent, ...]:
        """Install simultaneous Red and Blue allocations."""
        red_events = self.set_assignments("Red", red_assignment, record_event=False)
        blue_events = self.set_assignments("Blue", blue_assignment, record_event=False)
        self.last_events = red_events + blue_events
        return self.last_events

    def _nearest_acceleration(self, direction: np.ndarray) -> int:
        """Index of the acceleration primitive best aligned with ``direction`` (0 for zero)."""
        direction = np.asarray(direction, dtype=np.float64).copy()
        if self.spatial_dim == 2:
            direction[2] = 0.0
        norm = float(np.linalg.norm(direction))
        if norm < 1e-8:
            return 0
        candidates = np.asarray(self.valid_action_ids, dtype=np.int64)
        return int(candidates[np.argmax(ACCELERATION_PRIMITIVES[candidates] @ (direction / norm))])

    def blue_rush_actions(self) -> np.ndarray:
        """Known Blue lower rule: fly toward the assigned target; reserves and the dead coast."""
        result = []
        for agent in self.env.blue_agents:
            target_id = self._blue_assignment[int(agent.Id)]
            if agent.Health <= 0 or target_id is None:
                result.append(0)
                continue
            destination = np.asarray(self.env.targets[target_id].position, dtype=np.float64)
            result.append(self._nearest_acceleration(destination - np.asarray(agent.position, dtype=np.float64)))
        return np.asarray(result, dtype=np.int64)

    def _coerce_actions(self, side: str, actions: ActionInput) -> np.ndarray:
        agents = self._agents(side)
        ids = tuple(int(agent.Id) for agent in agents)
        if isinstance(actions, Mapping):
            unknown = set(int(key) for key in actions) - set(ids)
            if unknown:
                raise ValueError(f"actions contain non-{side} IDs: {sorted(unknown)}")
            missing_alive = [int(agent.Id) for agent in agents if agent.Health > 0 and int(agent.Id) not in actions]
            if missing_alive:
                raise ValueError(f"actions missing live {side} IDs: {missing_alive}")
            values = [int(actions.get(agent_id, 0)) for agent_id in ids]
        else:
            values = [int(value) for value in actions]
            if len(values) != len(agents):
                raise ValueError(f"{side} actions must contain {len(agents)} entries")
        for index, agent in enumerate(agents):
            if agent.Health <= 0:
                values[index] = 0
            if not 0 <= values[index] < len(ACCELERATION_PRIMITIVES):
                raise ValueError("action id is outside the 27 acceleration primitives")
            if values[index] not in self.valid_action_ids:
                raise ValueError("2D commanded actions must use a planar acceleration ID")
        return np.asarray(values, dtype=np.int64)

    def step(self, red_action_ids: ActionInput, blue_action_ids: Optional[ActionInput] = None
             ) -> Tuple[Dict[str, object], Dict[str, float], bool, Dict[str, object]]:
        """Advance the shared world one physical step; omitted Blue actions use the rush rule."""
        if self.env.is_episode_done() or self.step_count >= self.max_steps:
            raise RuntimeError("cannot step a completed grouping episode")
        red = self._coerce_actions("Red", red_action_ids)
        blue = self.blue_rush_actions() if blue_action_ids is None else self._coerce_actions("Blue", blue_action_ids)
        before = self._alive_signature()
        action_by_id = {int(agent.Id): int(action) for agent, action in zip(self.env.red_agents, red)}
        action_by_id.update({int(agent.Id): int(action) for agent, action in zip(self.env.blue_agents, blue)})
        self.env.step_physics([ACCELERATION_PRIMITIVES[action_by_id[int(agent.Id)]] for agent in self.env.agents])
        self.step_count += 1
        terminal_sign = self._terminal_sign()
        terminated = self.env.is_episode_done()
        truncated = self.step_count >= self.max_steps and not terminated
        done = terminated or truncated
        self.last_events = self._detect_events(before, terminated=terminated, truncated=truncated,
                                               terminal_sign=terminal_sign)
        if self.task_mode == "damage":
            outcome = 0
        elif terminated:
            outcome = terminal_sign
        elif truncated:
            outcome = {"red_win": 1, "blue_win": -1, "draw": 0}[self.horizon_policy]
        else:
            outcome = 0
        red_reward = (-self.env.step_target_damage if self.task_mode == "damage"
                      else float(outcome if done else 0.0))
        state = self.global_state()
        info: Dict[str, object] = {
            **self.env.task_info(), "terminated": terminated, "truncated": truncated,
            "outcome": int(outcome), "outcome_red": float(outcome), "horizon_policy": self.horizon_policy,
            "step": int(self.step_count), "events": [event.as_dict() for event in self.last_events],
            "red_assignment": self.red_assignment, "blue_assignment": self.blue_assignment,
            "red_actions": red.tolist(), "blue_actions": blue.tolist(), "targets": state["targets"],
            "spatial_dim": self.spatial_dim, "plane_altitude": self.plane_altitude,
            "bootstrap_mask": float(not terminated), "remaining_steps": max(0, self.max_steps - self.step_count),
        }
        return state, {"Red": red_reward, "Blue": -red_reward}, done, info

    def _terminal_sign(self) -> int:
        if self.task_mode == "damage":
            return 0
        if any(target.Health < 1e-3 for target in self.env.targets):
            return -1
        if all(agent.Health < 1e-3 for agent in self.env.blue_agents if agent.Type == "Attack"):
            return 1
        return 0

    def _alive_signature(self) -> Dict[str, Dict[int, bool]]:
        return {"Red": {int(agent.Id): bool(agent.Health > 0) for agent in self.env.red_agents},
                "Blue": {int(agent.Id): bool(agent.Health > 0) for agent in self.env.blue_agents},
                "targets": {target_id: bool(self.env.targets[target_id].Health > 0) for target_id in self.target_ids}}

    def _detect_events(self, before, *, terminated: bool, truncated: bool,
                       terminal_sign: int) -> Tuple[GroupingEvent, ...]:
        events = []
        for side in ("Red", "Blue"):
            dead = tuple(int(agent.Id) for agent in self._agents(side)
                         if before[side][int(agent.Id)] and agent.Health <= 0)
            if dead:
                targets = tuple(sorted({self._assignment(side).get(agent_id) for agent_id in dead} - {None}))
                events.append(GroupingEvent(kind="agents_destroyed", step=self.step_count, side=side,
                                            agent_ids=dead, target_ids=targets))
        breached = tuple(target_id for target_id in self.target_ids
                         if before["targets"].get(target_id, False) and self.env.targets[target_id].Health <= 0)
        if breached:
            events.append(GroupingEvent(kind="targets_breached", step=self.step_count, side="Blue",
                                        target_ids=breached, terminal=True))
        for target_id in self.target_ids:
            previously_active = any(before["Blue"][agent_id] and assigned == target_id
                                    for agent_id, assigned in self._blue_assignment.items())
            currently_active = any(agent.Health > 0 and self._blue_assignment[int(agent.Id)] == target_id
                                   for agent in self.env.blue_agents)
            if previously_active and not currently_active:
                events.append(GroupingEvent(kind="battlefield_resolved", step=self.step_count, side="Red",
                                            target_ids=(target_id,)))
        if terminated:
            damage = self.task_mode == "damage"
            events.append(GroupingEvent(
                kind="episode_finished" if damage else "red_win" if terminal_sign > 0 else "blue_win",
                step=self.step_count, side=None if damage else "Red" if terminal_sign > 0 else "Blue",
                terminal=True))
        elif truncated:
            damage = self.task_mode == "damage"
            events.append(GroupingEvent(kind="time_limit" if damage else "horizon_survived", step=self.step_count,
                                        side=None if damage else "Red", terminal=True))
        return tuple(events)

    @staticmethod
    def _entity_record(entity) -> Dict[str, object]:
        record = {"id": int(entity.Id), "position": np.asarray(entity.position, dtype=np.float64).copy(),
                  "velocity": np.asarray(entity.velocity, dtype=np.float64).copy(),
                  "health": float(entity.Health), "alive": bool(entity.Health > 0),
                  "initial_health": float(entity.initial_health), "step_damage": float(entity.step_damage),
                  "cumulative_damage": float(entity.cumulative_damage)}
        if getattr(entity, 'dynamics', None) is not None:
            record.update(rigid_state=entity.rigid_state.copy(), attitude=list(entity.attitude),
                          angular_velocity=list(entity.angular_velocity))
        return record

    def agent_states(self, side: str) -> Dict[int, Dict[str, object]]:
        """Global ID, kinematics, health and current assignment for one side."""
        assignment = self._assignment(side)
        return {int(agent.Id): {**self._entity_record(agent), "side": side, "type": str(agent.Type),
                                "assigned_target": assignment[int(agent.Id)]}
                for agent in self._agents(side)}

    def target_states(self) -> Dict[int, Dict[str, object]]:
        """Stable target indices with their underlying HAD entity IDs."""
        result = {}
        for target_id in self.target_ids:
            record = self._entity_record(self.env.targets[target_id])
            record["entity_id"] = record.pop("id")
            record["target_id"] = target_id
            result[target_id] = record
        return result

    def global_state(self) -> Dict[str, object]:
        """Copy-only command state suitable for logging or planning."""
        return {**self.env.task_info(), "spatial_dim": self.spatial_dim, "plane_altitude": self.plane_altitude,
                "valid_action_ids": self.valid_action_ids, "step": int(self.step_count),
                "red": self.agent_states("Red"), "blue": self.agent_states("Blue"),
                "targets": self.target_states(), "assignments": self.assignments,
                "terminal_sign": self._terminal_sign(), "horizon_reached": bool(self.step_count >= self.max_steps)}

    def snapshot(self) -> GroupingSnapshot:
        """Capture an exact physical and command-level branch point."""
        from had_env.config import PHYSICS_PROTOCOL
        transient = ('dynamics', '_next_rigid_state', '_next_clamped_axes')
        return GroupingSnapshot(
            step_count=int(self.step_count),
            entity_states=tuple(copy.deepcopy({k: v for k, v in entity.__dict__.items() if k not in transient})
                                for entity in self.env.entities),
            red_assignment=tuple(sorted(self._red_assignment.items())),
            blue_assignment=tuple(sorted(self._blue_assignment.items())),
            environment_rng_state=copy.deepcopy(self.env.np_random.bit_generator.state),
            adapter_rng_state=copy.deepcopy(self.rng.bit_generator.state),
            numpy_random_state=copy.deepcopy(np.random.get_state()),
            last_events=copy.deepcopy(self.last_events),
            task_mode=self.task_mode, target_health=self.target_health, physics_protocol=PHYSICS_PROTOCOL,
            max_steps=self.max_steps, horizon_policy=self.horizon_policy,
            last_physics_events=tuple(copy.deepcopy(self.env.last_physics_events)),
            spatial_dim=self.spatial_dim, plane_altitude=self.plane_altitude,
            target_initialization=self.env.target_initialization, env_agent_type=self.env_agent_type,
            env_agent_action_type=self.env_agent_action_type, effective_config=self.effective_config.to_dict())

    def restore(self, snapshot: GroupingSnapshot, *, continuation_seed: Optional[int] = None) -> Dict[str, object]:
        """Restore a branch, optionally replacing the continuation random streams."""
        from had_env.config import PHYSICS_PROTOCOL
        if (snapshot.physics_protocol != PHYSICS_PROTOCOL
                or snapshot.env_agent_type != self.env_agent_type
                or snapshot.env_agent_action_type != self.env_agent_action_type
                or snapshot.effective_config != self.effective_config.to_dict()
                or snapshot.task_mode != self.task_mode
                or snapshot.spatial_dim != self.spatial_dim
                or snapshot.plane_altitude != self.plane_altitude
                or snapshot.target_health != self.target_health
                or snapshot.max_steps != self.max_steps
                or snapshot.horizon_policy != self.horizon_policy):
            raise ValueError("Snapshot physics or task configuration differs; reset the environment")
        if len(snapshot.entity_states) != len(self.env.entities):
            raise ValueError("snapshot roster differs from this environment")
        if (set(dict(snapshot.red_assignment)) != set(self.red_ids)
                or set(dict(snapshot.blue_assignment)) != set(self.blue_ids)):
            raise ValueError("snapshot assignment roster differs from this environment")
        if not 0 <= snapshot.step_count <= self.max_steps:
            raise ValueError("snapshot step lies outside the episode horizon")
        for entity, state in zip(self.env.entities, snapshot.entity_states):
            if state.get("Id") != entity.Id or state.get("Type") != entity.Type:
                raise ValueError("snapshot entity identity differs from this environment")
        for entity, state in zip(self.env.entities, snapshot.entity_states):
            dynamics = entity.dynamics
            entity.__dict__.clear()
            entity.__dict__.update(copy.deepcopy(dict(state)))
            entity.dynamics = dynamics
        self.step_count = int(snapshot.step_count)
        self.env.target_initialization = snapshot.target_initialization
        self.env.physics_step_count = self.step_count
        self.env.last_physics_events = list(copy.deepcopy(snapshot.last_physics_events))
        self.env.boundary_clips = sum(len(getattr(entity, "_clamped_axes", ())) for entity in self.env.agents)
        self._red_assignment = dict(snapshot.red_assignment)
        self._blue_assignment = dict(snapshot.blue_assignment)
        self.last_events = copy.deepcopy(snapshot.last_events)
        self.env.update_alive_agents()
        if continuation_seed is None:
            self.env.np_random.bit_generator.state = copy.deepcopy(snapshot.environment_rng_state)
            self.rng.bit_generator.state = copy.deepcopy(snapshot.adapter_rng_state)
            np.random.set_state(copy.deepcopy(snapshot.numpy_random_state))
        else:
            seed = int(continuation_seed)
            self.env.np_random = np.random.default_rng(seed)
            self.rng = np.random.default_rng(seed ^ 0x5A17D0A1)
            np.random.seed(seed % (2**32))
        return self.global_state()
