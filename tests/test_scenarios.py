"""Model configuration contracts, independent concurrent worlds and controls."""
import copy

import numpy as np
import pytest

from make_env import make_env


def test_public_config_override_and_concurrent_model_geometry():
    env = make_env(config={"red_count": 2, "spatial_dim": 2}, red_count=1)
    other = make_env(env_agent_type="UAV_quadrotor")
    env.reset(seed=17)
    other.reset(seed=17)
    assert len(env.world.red_agents) == 1
    assert env.world.attack_distance == (200., 400.)
    assert other.world.attack_distance == (48., 96.)
    assert other.world.world_bounds.tolist() == [[-400., 400.], [-400., 400.], [0., 400.]]
    env.step({a: 0 for a in env.agents})
    assert env.world.attack_distance == (200., 400.)
    env.close()
    other.close()


@pytest.mark.parametrize("model", ["particle", "UAV_fixedwing", "UAV_quadrotor"])
def test_position_command_moves_without_teleport_and_invalid_batch_is_atomic(model):
    env = make_env(env_agent_type=model, env_agent_action_type="position", red_count=1,
                   blue_count=1, max_cycles=2)
    obs, _ = env.reset(seed=42)
    assert env.observation_space("red_0").contains(obs["red_0"])
    before = copy.deepcopy(env.world.agents[0].position)
    target = np.asarray(before) + [20., 0., 0.]
    target = np.clip(target, env.world.world_bounds[:, 0], env.world.world_bounds[:, 1])
    bad = {a: target for a in env.agents}
    bad["blue_0"] = [float("nan"), 0, 0]
    with pytest.raises(ValueError):
        env.step(bad)
    assert env.world.agents[0].position == before
    assert env.world.physics_step_count == 0
    env.step({a: target for a in env.agents})
    assert np.linalg.norm(np.asarray(env.world.agents[0].position) - before) > 0
    assert not np.array_equal(env.world.agents[0].position, target)
    env.close()


@pytest.mark.parametrize("kwargs", [dict(env_agent_type="UAV_fixedwing", spatial_dim=2),
    dict(env_agent_type="particle", env_agent_action_type="actuator"),
    dict(env_agent_type="unknown"), dict(env_agent_action_type="unknown")])
def test_unsupported_model_control_combinations_fail_at_construction(kwargs):
    with pytest.raises(ValueError):
        make_env(**kwargs)


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
    env = make_env(api="grouping", config=config, red_count=1, blue_count=1)
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


def test_native_effective_config_sets_uav_dimensions_and_task():
    from had_env.config import EnvConfig
    from had_env.core.make_env import HADEnv
    config = EnvConfig(env_agent_type="UAV_quadrotor", spatial_dim=3, task_mode="damage")
    env = HADEnv(1, 1, 1, effective_config=config)
    try:
        assert env.spatial_dim == 3
        assert env.task_mode == "damage"
        assert env.control_space.shape == (3,)
        env.reset(seed=17)
        assert all(entity.spatial_dim == 3 for entity in env.world)
    finally:
        env.close()
