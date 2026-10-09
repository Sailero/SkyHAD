"""Evaluate a Red callable against one frozen Blue rule; no training required.

Run after installation: python examples/defender_evaluation.py --seeds 11 12 13
See docs/BENCHMARK.md for the control contract and fair comparison protocol.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from had_env import __version__, make_env
from had_env.config import Interval


DEFAULTS = dict(api="parallel", env_agent_type="particle", env_agent_action_type="acceleration",
                spatial_dim=3, task_mode="survival", continuous=True, max_cycles=100,
                red_count=4, blue_count=4, target_count=2, red_scouts=0, blue_scouts=0,
                red_disturbers=0, blue_disturbers=0, initialization="random", evaluate=False,
                target_initialization="random", target_positions=None, target_health=2.,
                reward_weights=None, render_mode=None, record_events=False)
OPPONENT = dict(id="nearest-target-velocity-v1", target_selection="nearest_live_asset",
                tie_break="fixed_observation_row_order", clock_seconds=Interval,
                speed="model_preset_reference_speed", acceleration_norm_limit=1.,
                stochastic=False)


def _navigation_state(observation, *, context):
    rows = observation["entities"] if isinstance(observation, dict) else observation
    targets = rows[(rows[:, 7] > 0) & (rows[:, 10] > 0)]
    velocity_scale = 2. * context.preset.max_speed
    # Assets are stationary. Particle acceleration has no separate self-state.
    velocity = (observation["self_state"][3:6] if isinstance(observation, dict)
                else -targets[0, 3:6] * velocity_scale)
    bounds = np.asarray(context.world_bounds)
    return rows, np.asarray(velocity, dtype=float), np.linalg.norm(bounds[:, 1] - bounds[:, 0])


def acceleration_toward(displacement, velocity, *, context):
    """Velocity matching expressed as a bounded native acceleration intention."""
    displacement, velocity = np.asarray(displacement)[:context.spatial_dim], velocity[:context.spatial_dim]
    distance = np.linalg.norm(displacement)
    desired = displacement * (context.preset.reference_speed / max(distance, 1e-12))
    request = (desired - velocity) / Interval
    return (request / max(context.preset.acceleration_limit, np.linalg.norm(request))).astype(np.float32)


def blue_navigation(observation, info, *, context):
    """Frozen deterministic nearest-asset pursuit, evaluated before Red actions."""
    if not info["agent_mask"]:
        return np.zeros(context.spatial_dim, np.float32)
    rows, velocity, diagonal = _navigation_state(observation, context=context)
    targets = rows[(rows[:, 7] > 0) & (rows[:, 10] > 0)]
    nearest = targets[np.argmin(np.linalg.norm(targets[:, :context.spatial_dim], axis=1))]
    return acceleration_toward(nearest[:3] * diagonal, velocity, context=context)


def red_intercept(observation, info, *, context):
    """Example Red actor: pursue the nearest Blue contact's two-second prediction."""
    if not info["agent_mask"]:
        return np.zeros(context.spatial_dim, np.float32)
    rows, velocity, diagonal = _navigation_state(observation, context=context)
    enemies = rows[(rows[:, 7] > 0) & (rows[:, 9] > 0)]
    if not len(enemies):
        return np.zeros(context.spatial_dim, np.float32)
    nearest = enemies[np.argmin(np.linalg.norm(enemies[:, :context.spatial_dim], axis=1))]
    enemy_velocity = nearest[3:6] * (2. * context.preset.max_speed) + velocity
    destination = nearest[:3] * diagonal + 2. * enemy_velocity
    return acceleration_toward(destination, velocity, context=context)


def red_coast(observation, info, *, context):
    """Zero acceleration intention; this does not stop an aircraft or imply hover."""
    return np.zeros(context.spatial_dim, np.float32)


def run_episode(config, seed, red_policy=red_intercept):
    """Replace only red_policy(observation, info, *, context); reset its memory externally."""
    values = {**DEFAULTS, **config}
    if (values["api"] != "parallel" or values["env_agent_action_type"] != "acceleration"
            or not values["continuous"] or values["reward_weights"] is not None
            or any(values[k] for k in ("red_scouts", "blue_scouts", "red_disturbers", "blue_disturbers"))):
        raise ValueError("This example uses Parallel continuous acceleration, unshaped rewards and Attack-only rosters")
    env = make_env(**values)
    try:
        observations, infos = env.reset(seed=seed)
        context = env.effective_config
        initial = next(iter(infos.values()))
        resolved = {**values, **context.to_dict(), "target_initialization": initial["target_initialization"]}
        if resolved["target_positions"] is not None:
            points = np.asarray(resolved["target_positions"], dtype=float).copy()
            if context.spatial_dim == 2:
                points[:, 2] = context.plane_altitude
            resolved["target_positions"] = points.tolist()
        setup = dict(version=__version__, core_version=initial["core_version"],
                     physics_protocol=initial["physics_protocol"], run_config=resolved,
                     effective_config=context.to_dict(), navigation_preset=asdict(context.preset),
                     interval_seconds=Interval, opponent=OPPONENT.copy())
        red_return = 0.
        while env.agents:
            blue_actions = {a: blue_navigation(observations[a], infos[a], context=context)
                            for a in env.agents if a.startswith("blue_")}
            red_actions = {a: red_policy(observations[a], infos[a], context=context)
                           for a in env.agents if a.startswith("red_")}
            observations, rewards, terminations, truncations, infos = env.step({**red_actions, **blue_actions})
            task = next(iter(infos.values()))
            # One full team reward, even when Survival has retired every Red slot.
            red_return += task["team_reward"] * (1 if task["side"] == "red" else -1)
        outcome = ("truncated" if task["global_truncated"] else
                   "win" if task["outcome_red"] == 1 else "loss") if context.task_mode == "survival" else (
                   "terminated" if task["global_terminated"] else "truncated")
        return dict(setup=setup, seed=seed, task_mode=context.task_mode, steps=task["cycle"],
                    sim_time=task["sim_time"], red_return=red_return, outcome=outcome,
                    outcome_red=task["outcome_red"], global_terminated=task["global_terminated"],
                    global_truncated=task["global_truncated"], termination_reason=task["termination_reason"],
                    target_damage=task["target_damage"], target_damage_by_target=task["target_damage_by_target"],
                    blue_return=-red_return)
    finally:
        env.close()


def summarize(episodes):
    result = dict(episodes=len(episodes))
    if episodes[0]["task_mode"] == "survival":
        result.update(wins=sum(r["outcome"] == "win" for r in episodes),
                      losses=sum(r["outcome"] == "loss" for r in episodes),
                      truncations=sum(r["global_truncated"] for r in episodes))
        result["win_fraction_all_episodes"] = result["wins"] / len(episodes)
        result["truncation_fraction_all_episodes"] = result["truncations"] / len(episodes)
    else:
        result.update(mean_target_damage=float(np.mean([r["target_damage"] for r in episodes])),
                      natural_endings=sum(r["global_terminated"] for r in episodes),
                      truncations=sum(r["global_truncated"] for r in episodes))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("particle", "UAV_fixedwing", "UAV_quadrotor"), default="particle")
    parser.add_argument("--task", choices=("survival", "damage"), default="survival")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 12, 13])
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--spatial-dim", type=int, choices=(2, 3), default=3)
    parser.add_argument("--red-policy", choices=("intercept", "coast"), default="intercept")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if any(s < 0 for s in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Provide distinct nonnegative episode seeds")
    policy = red_intercept if args.red_policy == "intercept" else red_coast
    config = dict(env_agent_type=args.model, task_mode=args.task,
                  spatial_dim=args.spatial_dim, max_cycles=args.horizon)
    episodes = [run_episode(config, seed, policy) for seed in args.seeds]
    setup = episodes[0]["setup"]
    for row in episodes:
        row.pop("setup")
    result = dict(schema="skyhad-native-defender-example-v1", **setup,
                  red_policy=dict(id=args.red_policy, stochastic=False,
                                  lookahead_seconds=2. if args.red_policy == "intercept" else None),
                  seeds=args.seeds, summary=summarize(episodes), episodes=episodes)
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
