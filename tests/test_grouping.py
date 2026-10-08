"""Standalone regressions recorded before removing the research packages."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from had_env.actions import ACCELERATION_PRIMITIVES
from had_env.grouping.domain import Group, Grouping
from had_env.grouping.environment import KnownOpponentEnv
from had_env.grouping.policies import RulePolicy
from had_env.grouping.rules import make_env
from make_env import make_env as make_public_env


_REFERENCE = json.loads((Path(__file__).parent / "data" / "grouping_extraction_validation.json").read_text())


def _quantize(value):
    # Micrometre-scale rounding avoids asserting platform-level float noise.
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, (tuple, list)):
        return [_quantize(item) for item in value]
    if isinstance(value, dict):
        return {key: _quantize(item) for key, item in value.items()
                if key not in {"env_agent_type", "env_agent_action_type", "effective_config", "world_bounds", "scene_scale"}}
    return value


@pytest.mark.parametrize("policy_name,roster,seed,expected", [
    (case["policy_name"], case["roster"], case["seed"], case["sha256"])
    for case in _REFERENCE["golden_cases"]
])
def test_complete_episode_preserves_current_particle_baseline(policy_name, roster, seed, expected):
    """Reference includes each command, state, reward, native event and outcome.

    References were captured from unmodified cf26fd4 before this upgrade.
    Added provenance/config metadata are excluded; every physical result remains compared.
    """
    env = make_env(*roster, seed=seed)
    policy = RulePolicy(policy_name, seed + 1)
    try:
        state = env.reset(seed)
        rows = [state.to_dict()]
        while not env.done:
            action = policy.act(state)
            state, reward, done, info = env.step(action)
            rows.append(dict(action=action.to_dict(), state=state.to_dict(),
                             reward=reward, done=done, info=info))
        encoded = json.dumps(_quantize(rows), sort_keys=True, separators=(",", ":")).encode()
        assert hashlib.sha256(encoded).hexdigest() == expected
    finally:
        env.close()


def test_default_grouping_executor_requires_no_checkpoint_and_has_no_size_cap():
    env = KnownOpponentEnv(red=8, blue=8, max_steps=1)
    try:
        state = env.reset(seed=31)
        grouping = Grouping((Group(state.ids("targets")[0], state.ids("red")),))
        _, _, done, info = env.step(grouping)
        assert type(env.executor).__name__ == "RuleExecutor"
        assert env.group_max_size is None
        assert done and info["delta"] == 1
    finally:
        env.close()


def test_grouping_snapshot_restores_exact_independent_future():
    parent, child = make_env(8, seed=8), make_env(8, seed=8)
    policy = RulePolicy("rule")
    try:
        state = parent.reset(seed=8)
        state, _, _, _ = parent.step(policy.act(state))
        child.reset(seed=8)
        child.restore(parent.snapshot())
        before = parent.state().to_dict()
        child_result = child.step(policy.act(child.state()))
        assert parent.state().to_dict() == before
        parent_result = parent.step(policy.act(parent.state()))
        assert parent_result[0].to_dict() == child_result[0].to_dict()
        assert parent_result[1:] == child_result[1:]
    finally:
        parent.close()
        child.close()


def test_primitive_zero_and_26_directions_have_stable_native_order():
    assert ACCELERATION_PRIMITIVES.shape == (27, 3)
    assert ACCELERATION_PRIMITIVES.dtype == np.float32
    np.testing.assert_array_equal(ACCELERATION_PRIMITIVES[0], np.zeros(3))
    np.testing.assert_allclose(np.linalg.norm(ACCELERATION_PRIMITIVES[1:], axis=1), 1.0)
    np.testing.assert_allclose(ACCELERATION_PRIMITIVES[1], -np.ones(3) / np.sqrt(3))
    np.testing.assert_allclose(ACCELERATION_PRIMITIVES[-1], np.ones(3) / np.sqrt(3))


def test_legacy_grouping_factory_resolves_config_before_omitted_defaults():
    env = make_env(1, config={"max_steps": 7, "opponent": "balanced", "seed": 88})
    try:
        assert (env.max_steps, env.opponent, env.seed) == (7, "balanced", 88)
    finally:
        env.close()


def test_known_opponent_constructor_honors_config_before_explicit_overrides():
    from had_env.grouping.environment import KnownOpponentEnv
    env = KnownOpponentEnv(red=1, blue=1, config={"env_agent_type": "UAV_quadrotor", "task_mode": "damage"})
    try:
        assert env.env_agent_type == "UAV_quadrotor"
        assert env.task_mode == "damage"
        assert env.spatial_dim == 3
    finally:
        env.adapter.env.close()
    overridden = KnownOpponentEnv(red=1, blue=1,
        config={"env_agent_type": "UAV_quadrotor", "task_mode": "damage"},
        env_agent_type="UAV_fixedwing", task_mode="survival")
    try:
        assert overridden.env_agent_type == "UAV_fixedwing"
        assert overridden.task_mode == "survival"
        assert overridden.spatial_dim == 3
    finally:
        overridden.adapter.env.close()


def test_grouping_factory_routes_custom_geometry_and_scales_rule_executor():
    config = {"env_agent_type": "UAV_quadrotor", "scene_scale": .2,
              "world_bounds": [[-500., 500.], [-500., 500.], [0., 500.]]}
    env = make_public_env(api="grouping", config=config, red_count=1, blue_count=1)
    try:
        env.reset(seed=17)
        np.testing.assert_array_equal(env.adapter.env.world_bounds, config["world_bounds"])
        assert env.effective_config.scene_scale == .2
        assert env.executor.config["guard_distance"] == 140.
    finally:
        env.adapter.env.close()


@pytest.mark.parametrize("model", ["UAV_fixedwing", "UAV_quadrotor"])
def test_scaled_uav_rush_reaches_target_fire_before_default_horizon(model):
    from had_env.grouping.environment import KnownOpponentEnv
    env = KnownOpponentEnv(red=1, blue=1, targets=1, max_steps=100,
        env_agent_type=model, task_mode="damage", seed=17)
    try:
        env.reset(seed=17)
        adapter, world = env.adapter, env.adapter.env
        target, blue = world.targets[0], world.blue_agents[0]
        velocity = [-30., 0., 0.] if model == "UAV_fixedwing" else [0., 0., 0.]
        blue.reset([0., target.position[1], target.position[2]], velocity)
        world.red_agents[0].Health = 0.
        world.update_alive_agents()
        adapter.set_joint_assignments({adapter.red_ids[0]: None}, {adapter.blue_ids[0]: 0})
        done = False
        while not done and adapter.step_count < 100:
            _, _, done, _ = adapter.step({adapter.red_ids[0]: 0}, blue_style="rush")
            assert np.isfinite(blue.rigid_state).all()
            assert abs(np.linalg.norm(blue.rigid_state[6:10])-1.) < 1e-12
        assert adapter.step_count < 100
        assert world.target_damage > 0.
        assert target.Health == target.initial_health  # damage task remains live
    finally:
        env.adapter.env.close()
