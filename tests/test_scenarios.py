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
