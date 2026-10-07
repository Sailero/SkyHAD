"""Per-step n-versus-one integer assignment with linear Blue prediction."""
from __future__ import annotations

# Adapted from Open-SCORE open_score/rules/coverage_rule.py; native HAD physics remains in had_env.

import copy
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .wrapper import (
    ACCELERATION_PRIMITIVES, HADStage3Adapter, DecisionState, Entity, Group, Grouping,
    sample, build_decision_state, EpisodeDiagnostics, trajectory_frame,
    had_config as config, HADEnv,
)


def env_constants(spatial_dim=2):
    inner, outer = config.attack_distance_for(spatial_dim)
    fire_range = inner
    lethal_ratio = 1.0 / max(float(config.AttackIntensity), 1e-9)
    if lethal_ratio >= 1.0:
        kill_radius = float(inner)
    else:
        kill_radius = float(inner + (1.0 - lethal_ratio) * (outer - inner))
    return {
        "dt": float(config.Interval),
        "dim": int(config.EnvDim),
        "plane_altitude": float(getattr(config, "PlanarAltitude", 100.0)),
        "vmin": float(config.vDomain[0]),
        "vmax_red": float(config.vDomain[1]),
        "vmax_blue": float(config.vDomain[1]) * float(config.BlueVmaxCoef),
        "amax_red": float(config.aMax),
        "amax_blue": float(config.aMax) * float(config.BlueAmaxCoef),
        "fire_range": fire_range,
        "kill_radius": kill_radius,
        "attack_inner": float(inner),
        "attack_outer": float(outer),
        "attack_intensity": float(config.AttackIntensity),
        "target_hp": float(config.initial_health),
        "avoid": float(config.AvoidanceDistance),
        "bounds": np.asarray(config.AeroPoint, dtype=np.float64),
        "horizon": int(getattr(config, "DefaultMaxSteps", 100)),
    }


CONST = env_constants()
PRIMITIVES = np.asarray(ACCELERATION_PRIMITIVES, dtype=np.float64)
RED_RULE_VERSION = "rule_nv1_v1"


def refresh_constants():
    global CONST
    CONST = env_constants()
    return CONST


def apply_live_params(updates):
    for name, value in updates.items():
        setattr(config, name, value)
    return refresh_constants()


def clipspeed(velocity, vmin, vmax):
    value = np.asarray(velocity, dtype=np.float64)
    speed = float(np.linalg.norm(value))
    if speed < 1e-3:
        fallback = np.zeros(3, dtype=np.float64)
        fallback[0] = vmin
        return fallback
    return value * (float(np.clip(speed, vmin, vmax)) / speed)


def clamp_position(position, bounds=None):
    point = np.asarray(position, dtype=np.float64).copy()
    box = CONST["bounds"] if bounds is None else bounds
    axes = []
    for dim in range(3):
        if point[dim] > box[dim][1]:
            point[dim] = box[dim][1]
            axes.append((dim, 1))
        elif point[dim] < box[dim][0]:
            point[dim] = box[dim][0]
            axes.append((dim, -1))
    return point, axes


def step_kinematics(position, velocity, accel_dir, vmin, vmax, amax, dt=None):
    dt = CONST["dt"] if dt is None else dt
    position = np.asarray(position, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)
    accel = np.asarray(accel_dir, dtype=np.float64) * amax
    next_position, axes = clamp_position(position + velocity * dt)
    next_velocity = clipspeed(velocity + accel * dt, vmin, vmax)
    for axis, sign in axes:
        if sign > 0 and next_velocity[axis] > 0:
            next_velocity[axis] = 0.0
        elif sign < 0 and next_velocity[axis] < 0:
            next_velocity[axis] = 0.0
    return next_position, next_velocity


def nearest_primitive(direction):
    norm = float(np.linalg.norm(direction))
    if norm < 1e-8:
        return 0
    return int(np.argmax(PRIMITIVES @ (np.asarray(direction, dtype=np.float64) / norm)))


def unit(vector):
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm < 1e-9:
        fallback = np.zeros(3, dtype=np.float64)
        fallback[0] = 1.0
        return fallback
    return value / norm


def classify_targets(blue_states, target_positions):
    targets = np.asarray(target_positions, dtype=np.float64)
    mapping = {}
    for blue_id, position, velocity in blue_states:
        best_k, best_key = 0, None
        for index, goal in enumerate(targets):
            offset = np.asarray(position, dtype=np.float64) - goal
            vel = np.asarray(velocity, dtype=np.float64)
            speed2 = float(np.dot(vel, vel))
            tau = 0.0 if speed2 < 1e-9 else max(0.0, -float(np.dot(offset, vel)) / speed2)
            distance = float(np.linalg.norm(offset + tau * vel))
            key = (distance, tau, index)
            if best_key is None or key < best_key:
                best_key, best_k = key, index
        mapping[int(blue_id)] = int(best_k)
    return mapping


def predict_blue(blue_states, horizon=1):
    """Extrapolate p + v * dt * step without acceleration or boundary rollout."""
    offsets = np.arange(int(horizon) + 1, dtype=np.float64)[:, None] * CONST["dt"]
    return {
        int(blue_id): np.asarray(position, dtype=np.float64)
        + offsets * np.asarray(velocity, dtype=np.float64)
        for blue_id, position, velocity in blue_states
    }


def steer(position, velocity, aim, remaining_steps, vmin=None, vmax=None, amax=None,
          loiter=False, forbidden=None, action_ids=None, fire_range=None):
    vmin = CONST["vmin"] if vmin is None else vmin
    vmax = CONST["vmax_red"] if vmax is None else vmax
    amax = CONST["amax_red"] if amax is None else amax
    dt = CONST["dt"]
    fire = CONST["fire_range"] if fire_range is None else fire_range
    position = np.asarray(position, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)
    aim = np.asarray(aim, dtype=np.float64)
    offset = aim - position
    distance = float(np.linalg.norm(offset))
    heading = unit(velocity)
    desired = unit(offset)
    cosine = float(np.clip(np.dot(heading, desired), -1.0, 1.0))
    theta = float(np.arccos(cosine))
    speed_turn = amax / max(theta, 1e-3)
    speed_time = distance / max(float(remaining_steps) * dt, dt)
    command = float(np.clip(min(vmax, speed_time, speed_turn), vmin, vmax))
    if loiter and distance < 200.0:
        tangent = np.cross(offset if distance > 1e-6 else np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))
        reference = vmin * unit(tangent)
    else:
        reference = command * desired

    def next_state(primitive):
        return step_kinematics(position, velocity, primitive, vmin, vmax, amax, dt)

    def blocked(next_p, next_v):
        if not forbidden:
            return False
        future = next_p + next_v * dt
        return any(float(np.linalg.norm(future - point)) < fire for point in forbidden)

    best_id, best_err = 0, float("inf")
    found = False
    next_velocities = {}
    candidates = range(len(PRIMITIVES)) if action_ids is None else action_ids
    for index in candidates:
        primitive = PRIMITIVES[index]
        next_p, next_v = next_state(primitive)
        next_velocities[index] = next_v
        if blocked(next_p, next_v):
            continue
        error = float(np.sum((next_v - reference) ** 2))
        if error < best_err - 1e-12 or (abs(error - best_err) <= 1e-12 and index < best_id):
            best_id, best_err, found = index, error, True
    if found:
        return best_id
    for index, next_v in next_velocities.items():
        error = float(np.sum((next_v - reference) ** 2))
        if error < best_err - 1e-12 or (abs(error - best_err) <= 1e-12 and index < best_id):
            best_id, best_err = index, error
    return best_id


class CoveragePolicy:
    """Every step: binary n-versus-one assignment covering all selected Blue."""

    def __init__(self, name=RED_RULE_VERSION, seed=0):
        self.name = name
        self.seed = int(seed)
        self.reset()

    def reset(self):
        self.last_plan = {}
        self.last_grouping = None
        self.last_state_key = None

    @staticmethod
    def state_key(state):
        return (state.step, state.red, state.blue, state.targets)

    def act(self, state: DecisionState) -> Grouping:
        reds = sorted(state.alive("red"), key=lambda entity: entity.id)
        blues = sorted(state.alive("blue"), key=lambda entity: entity.id)
        targets = sorted(state.alive("targets"), key=lambda entity: entity.id)
        assignments = {}
        predicted = {}
        selected = []
        objective = 0.0
        grouping = Grouping((), tuple(entity.id for entity in reds))
        if reds and blues and targets:
            red_positions = np.asarray([entity.position for entity in reds], dtype=np.float64)
            blue_positions = np.asarray([entity.position for entity in blues], dtype=np.float64)
            # When outnumbered, rank Blue by current distance to its nearest Red.
            distances = np.linalg.norm(red_positions[:, None, :] - blue_positions[None, :, :], axis=-1)
            nearest = np.min(distances, axis=0)
            chosen = sorted(range(len(blues)), key=lambda j: (nearest[j], blues[j].id))[:len(reds)]
            selected = sorted((blues[j] for j in chosen), key=lambda entity: entity.id)
            paths = predict_blue([(entity.id, entity.position, entity.velocity) for entity in selected], horizon=1)
            predicted = {entity.id: paths[entity.id][1].tolist() for entity in selected}
            costs = np.linalg.norm(red_positions[:, None, :] - np.asarray(list(predicted.values()))[None, :, :], axis=-1)
            n_red, n_blue = costs.shape
            size = n_red * n_blue
            variables = np.arange(size)
            # x[i,b] is binary: one Blue per Red; at least one Red per Blue.
            rows = np.concatenate((np.repeat(np.arange(n_red), n_blue),
                                   n_red + np.tile(np.arange(n_blue), n_red)))
            columns = np.concatenate((variables, variables))
            matrix = coo_matrix((np.ones(2 * size), (rows, columns)),
                                shape=(n_red + n_blue, size)).tocsc()
            lower = np.ones(n_red + n_blue)
            upper = np.concatenate((np.ones(n_red), np.full(n_blue, n_red)))
            solution = milp(
                c=costs.ravel(), integrality=np.ones(size, dtype=np.uint8),
                bounds=Bounds(0.0, 1.0), constraints=LinearConstraint(matrix, lower, upper),
                options={"mip_rel_gap": 0.0},
            )
            if not solution.success or solution.x is None:
                raise RuntimeError(f"nv1 整数规划未求得最优解（第 {state.step} 步）：{solution.message}")
            binary = np.rint(solution.x).reshape(n_red, n_blue)
            if (np.max(np.abs(solution.x.reshape(n_red, n_blue) - binary)) > 1e-6
                    or np.any((binary < 0) | (binary > 1))
                    or np.any(binary.sum(axis=1) != 1) or np.any(binary.sum(axis=0) < 1)):
                raise RuntimeError(f"nv1 整数规划返回了不满足覆盖约束的解（第 {state.step} 步）。")
            assignments = {entity.id: selected[int(np.argmax(binary[i]))].id for i, entity in enumerate(reds)}
            # Native Group.target still denotes a protected target, never a Blue ID.
            inferred = classify_targets([(entity.id, entity.position, entity.velocity) for entity in selected],
                                        [entity.position for entity in targets])
            grouping = Grouping(tuple(
                Group(targets[inferred[blue.id]].id, tuple(red.id for red in reds if assignments[red.id] == blue.id))
                for blue in selected
            ))
            grouping.validate((entity.id for entity in reds), (entity.id for entity in targets), max_members=None)
            objective = float(np.sum(costs * binary))
        selected_ids = [entity.id for entity in selected]
        self.last_plan = {
            "rule_version": RED_RULE_VERSION, "step": int(state.step),
            "assignment": assignments, "selected_blue": selected_ids,
            "ignored_blue": [entity.id for entity in blues if entity.id not in selected_ids],
            "predicted_blue": predicted, "prediction_steps": 1, "total_distance": objective,
        }
        self.last_grouping = grouping
        self.last_state_key = self.state_key(state)
        return grouping


class CoverageExecutor:
    """Follow the current nv1 assignment; reuse only this step's exact solution."""

    def __init__(self, guard_distance=1100.0, *, policy=None):
        self.config = {"version": RED_RULE_VERSION, "prediction_steps": 1,
                       "guard_distance": float(guard_distance)}
        self.policy = policy if policy is not None else CoveragePolicy()
        self.reset(())

    def reset(self, ids):
        self.last_actions = {int(i): -1 for i in ids}
        self.last_stations = {}
        self.last_roles = {}
        self.last_interception = {}
        self.selected_blue = []
        self.ignored_blue = []
        self.forced_contact = 0
        self.switches = 0
        self.policy.reset()

    def prune(self, ids):
        live = set(map(int, ids))
        self.last_actions = {i: a for i, a in self.last_actions.items() if i in live}

    def memory(self):
        return {}

    def snapshot(self):
        return {"config": dict(self.config), "last_actions": dict(self.last_actions),
                "switches": self.switches}

    def restore(self, snapshot):
        if snapshot["config"] != self.config:
            raise ValueError("nv1 executor parameters differ from snapshot")
        self.reset(snapshot["last_actions"])
        self.last_actions = {int(i): int(a) for i, a in snapshot["last_actions"].items()}
        self.switches = int(snapshot.get("switches", 0))

    def plan_stations(self, state: DecisionState, grouping: Grouping):
        if self.policy.last_state_key != self.policy.state_key(state):
            self.policy.act(state)
        if self.policy.last_grouping != grouping:
            raise ValueError("nv1 执行器必须使用同一步整数规划产生的分组。")
        plan = self.policy.last_plan
        stations = {red_id: np.asarray(plan["predicted_blue"][blue_id], dtype=np.float64)
                    for red_id, blue_id in plan["assignment"].items()}
        roles = {red_id: {"kind": "intercept", "tau": 1, "cover": [blue_id]}
                 for red_id, blue_id in plan["assignment"].items()}
        targets = state.alive("targets")
        for red in state.alive("red"):
            if red.id not in stations:
                stations[red.id] = (np.mean([entity.position for entity in targets], axis=0)
                                    + np.array([self.config["guard_distance"], 0.0, 0.0])
                                    if targets else np.asarray(red.position, dtype=np.float64))
                roles[red.id] = {"kind": "guard", "tau": 1, "cover": []}
        self.last_stations = {i: point.tolist() for i, point in stations.items()}
        self.last_roles = roles
        self.last_interception = dict(plan["assignment"])
        self.selected_blue = list(plan["selected_blue"])
        self.ignored_blue = list(plan["ignored_blue"])
        return stations, roles, list(stations.values())

    def act(self, adapter, grouping):
        state = _adapter_state(adapter, grouping)
        grouping = grouping.prune(state.ids("red"))
        reds = {entity.id: entity for entity in state.alive("red")}
        self.prune(reds)
        stations, roles, _ = self.plan_stations(state, grouping)
        result = {int(i): 0 for i in adapter.red_ids}
        for red_id, red in reds.items():
            action = steer(red.position, red.velocity, stations[red_id], 1,
                           loiter=roles[red_id]["kind"] == "guard",
                           action_ids=adapter.valid_action_ids, fire_range=adapter.env.fire_range)
            if self.last_actions.get(red_id, action) != action:
                self.switches += 1
            result[red_id] = int(action)
        self.last_actions.update({i: result[i] for i in reds})
        return result


class DirectActionPolicy:
    """Observation-to-action rule baselines, with no grouping stage or MILP."""

    def __init__(self, kind, seed=0):
        self.kind, self.seed = kind, int(seed)
        self.reset()

    def reset(self):
        self.rng = np.random.default_rng(self.seed)

    def act(self, state, side, action_ids):
        fire_range = config.attack_distance_for(state.spatial_dim)[0]
        friends = state.red if side == "red" else state.blue
        goals = state.alive("blue" if side == "red" else "targets")
        actions = {entity.id: 0 for entity in friends}
        for entity in friends:
            if not entity.alive:
                continue
            if self.kind == "random_accel":
                actions[entity.id] = int(self.rng.choice(action_ids))
            elif goals:
                target = min(goals, key=lambda goal: (
                    float(np.linalg.norm(np.asarray(goal.position) - entity.position)), goal.id))
                actions[entity.id] = steer(
                    entity.position, entity.velocity, target.position, 1,
                    vmax=CONST["vmax_red" if side == "red" else "vmax_blue"],
                    amax=CONST["amax_red" if side == "red" else "amax_blue"],
                    action_ids=action_ids, fire_range=fire_range,
                )
        return actions


def _adapter_state(adapter, grouping, opponent="reactive"):
    return build_decision_state(adapter, grouping, opponent)

