"""Public APIs, immutable config, fixed observations and independent RNG streams."""
import copy
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from make_env import make_env
from had_env.simulation import Simulation
from pettingzoo.test import parallel_api_test, parallel_seed_test


def _world_rows(env):
    return [[list(map(float, x.position)), list(map(float, x.velocity)), float(x.Health),
             bool(getattr(x, "IsFire", False)), bool(getattr(x, "IsDisturb", False))]
            for x in env.simulation.entities]


def _place(env, positions):
    for e, (p, v) in zip(env.simulation.entities, positions):
        e.reset(list(p), list(v))
    env.simulation.reset_episode_state()


@pytest.mark.parametrize("continuous", [False, True])
def test_official_parallel_api_and_seed_checks(continuous):
    factory = lambda: make_env(red_count=3, blue_count=3, continuous=continuous)
    env = factory()
    parallel_api_test(env, num_cycles=30)
    parallel_seed_test(factory, num_cycles=20)
    env.close()


def test_stable_spaces_order_shapes_and_reward_components():
    env = make_env(red_count=3, blue_count=3, red_scouts=1, blue_disturbers=1, reward_weights=[1, .3, .7, 1])
    obs, infos = env.reset(seed=77)
    names = env.possible_agents.copy()
    action_spaces = [env.action_space(a) for a in names]
    observation_spaces = [env.observation_space(a) for a in names]
    assert infos["red_0"]["role"] == "scout"
    assert infos["blue_0"]["role"] == "disturb"
    assert all(x.shape == (7, 11) for x in obs.values())
    for a, row in obs.items():
        assert env.observation_space(a).contains(row)
        assert len(infos[a]["observation_entities"]) == 7
        assert a not in infos[a]["observation_entities"]
    _, rewards, _, _, infos = env.step({a: 0 for a in env.agents})
    for a, value in rewards.items():
        assert value == pytest.approx(np.dot(infos[a]["LatentReward"], [1, .3, .7, 1]))
    env.reset(seed=88)
    assert env.possible_agents == names
    assert all(env.action_space(a) is s for a, s in zip(names, action_spaces))
    assert all(env.observation_space(a) is s for a, s in zip(names, observation_spaces))
    assert env.state_space.contains(env.state())
    env.close()


@pytest.mark.parametrize("continuous,bad", [(False, 27), (False, -1), (False, True), (False, 2.0),
                                          (True, [0, float("nan"), 0]), (True, [0, 2, 0]),
                                          (True, [1, 0]), (True, ["0", "1", "0"])])
def test_invalid_joint_action_has_no_side_effects(continuous, bad):
    env = make_env(red_count=1, blue_count=1, continuous=continuous)
    env.reset(seed=12)
    before = _world_rows(env)
    rng = copy.deepcopy(env.np_random.bit_generator.state)
    good = [0, 0, 0] if continuous else 0
    for actions in ({"red_0": good, "blue_0": bad}, {"red_0": good}, {"unknown": good}):
        with pytest.raises(ValueError):
            env.step(actions)
        assert _world_rows(env) == before
        assert env.num_cycles == env.simulation.physics_step_count == 0
        assert env.np_random.bit_generator.state == rng
    env.close()


def test_death_final_transition_then_removal_without_slot_shifts():
    env = make_env(red_count=3, blue_count=1, target_count=1, record_events=True)
    env.reset(seed=3)
    _place(env, [([-500, -500, 100], [20, 0, 0]), ([-500, -500, 100], [20, 0, 0]),
                 ([-1000, 1000, 100], [20, 0, 0]), ([2000, 0, 100], [-20, 0, 0]),
                 ([-2200, 0, 100], [0, 0, 0])])
    obs, rewards, terms, truncs, infos = env.step({a: 0 for a in env.agents})
    assert terms["red_0"] and terms["red_1"]
    assert not terms["red_2"] and not terms["blue_0"]
    assert env.agents == ["red_2", "blue_0"]
    assert set(obs) == set(rewards) == set(terms) == set(truncs) == set(infos) == set(env.possible_agents)
    assert any(e["kind"] == "collision" for e in infos["red_0"]["events"])
    ids = infos["red_2"]["observation_entities"]
    assert obs["red_2"][ids.index("red_0"), -1] == 0
    obs2, rewards2, *_ = env.step({a: 0 for a in env.agents})
    assert set(obs2) == set(rewards2) == {"red_2", "blue_0"}
    assert obs2["red_2"].shape == obs["red_2"].shape
    assert obs2["red_2"][ids.index("red_0"), -1] == 0
    env.close()


def test_private_rng_does_not_change_ambient_numpy_and_is_repeatable(monkeypatch):
    import importlib
    # The shipped pi limit normally leaves this fallback dormant. Lower it
    # only inside this test to exercise the preserved native random path.
    monkeypatch.setattr(importlib.import_module("had_env.geometry"), "wMax", np.pi / 2)
    first, second = make_env(red_count=1, blue_count=1), make_env(red_count=1, blue_count=1)
    np.random.seed(143)
    before = copy.deepcopy(np.random.get_state())
    obs1, _ = first.reset(seed=91)
    obs2, _ = second.reset(seed=91)
    for a in obs1:
        np.testing.assert_array_equal(obs1[a], obs2[a])
    for e in (first, second):
        _place(e, [([-1000, 0, 100], [20, 0, 0]), ([2000, 0, 100], [-20, 0, 0]),
                   ([-2200, -700, 100], [0, 0, 0]), ([-2200, 700, 100], [0, 0, 0])])
        e.simulation.agents[0].velocity = [20., 0., 0.]
    action = {a: 0 for a in first.agents}
    action["red_0"] = 5  # a discrete acceleration containing negative x
    first.step(action)
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])
    np.random.random(100)
    second.step(action)
    np.testing.assert_array_equal(first.state(), second.state())
    first.close()
    second.close()


def test_mpe_fixed_lists_flattened_obs_one_hot_and_done_semantics():
    env = make_env(api="mpe", red_count=1, blue_count=1, max_cycles=1, initialization="uniform")
    obs = env.reset(seed=13)
    assert env.n == 2 and len(obs) == 2
    assert all(space.contains(row) for space, row in zip(env.observation_space, obs))
    actions = [np.eye(27, dtype=np.float32)[0], 0]
    obs, rewards, dones, info = env.step(actions)
    assert dones == [True, True] and rewards == [0., 0.]
    assert all(x["truncated"] and not x["terminated"] for x in info["n"])
    obs, rewards, dones, info = env.step(actions)
    assert all(not np.any(row) for row in obs) and rewards == [0., 0.]
    assert all(x["inactive"] for x in info["n"])
    env.close()


def test_public_import_is_headless_without_research_dependencies():
    code = "from make_env import make_env; import sys; e=make_env(); e.reset(seed=1); e.step({a:0 for a in e.agents}); e.close(); assert not any(k in sys.modules for k in ['torch','pygame','PySide6','open_score'])"
    subprocess.run([sys.executable, "-B", "-c", code], cwd=Path(__file__).parents[1], check=True)


@pytest.mark.parametrize("kwargs", [dict(red_count=0), dict(blue_count=True), dict(target_count=0),
    dict(red_count=1, red_scouts=2), dict(blue_count=1, blue_scouts=1),
    dict(max_cycles=0), dict(reward_weights=[1, 2]), dict(reward_weights=[1, 2, 3, float("nan")]),
    dict(target_region=[[0, float("nan")]] * 3), dict(initialization="missing"),
    dict(api="missing"), dict(scenario_name="missing"), dict(continuous="yes")])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        make_env(**kwargs)


@pytest.mark.parametrize("task", ["survival", "damage"])
@pytest.mark.parametrize("control", ["actuator", "position"])
def test_native_box_controls_keep_valid_fixed_slots_after_agent_death(task, control):
    env = make_env(env_agent_type="UAV_quadrotor", env_agent_action_type=control,
                   task_mode=task, red_count=1, blue_count=1, max_cycles=3)
    env.reset(seed=17)
    env.simulation.red_agents[0].Health = 0
    for _ in range(2):
        actions = {a: ([.4]*4 if control == "actuator" else env._entity(a).position)
                   for a in env.agents}
        obs, _, _, _, _ = env.step(actions)
        assert env.state_space.contains(env.state())
        assert all(env.observation_space(a).contains(row) for a, row in obs.items())
    env.close()


def test_headless_import_steps_and_close_never_import_pygame():
    # A fresh interpreter is necessary: rendering tests in this process may
    # legitimately have already imported pygame during test collection.
    project = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(project)
    code = """
import importlib.abc
import sys
class NoRenderer(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'pygame' or fullname.startswith('pygame.') or fullname.startswith('had_env.render'):
            raise AssertionError('headless physics imported renderer: ' + fullname)
sys.meta_path.insert(0, NoRenderer())
from had_env.config import AeroPoint
from had_env.simulation import Simulation
import numpy as np
env = Simulation(2, 1, 1)
env.reset(seed=4, evaluate=True)
env.step_physics(np.zeros((3, 3)))
assert len(env.step(np.zeros((3, 3)))) == 6
env.close()
env.close()
assert env.render() is False
assert not any(name == 'pygame' or name.startswith('pygame.') for name in sys.modules)
print('headless-ok')
"""
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=project,
                            env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "headless-ok"


def test_public_config_override_and_concurrent_model_geometry():
    env = make_env(config={"red_count": 2, "spatial_dim": 2}, red_count=1)
    other = make_env(env_agent_type="UAV_quadrotor")
    env.reset(seed=17)
    other.reset(seed=17)
    assert len(env.simulation.red_agents) == 1
    assert env.simulation.attack_distance == (200., 400.)
    assert other.simulation.attack_distance == (48., 96.)
    assert other.simulation.world_bounds.tolist() == [[-400., 400.], [-400., 400.], [0., 400.]]
    env.step({a: 0 for a in env.agents})
    assert env.simulation.attack_distance == (200., 400.)
    env.close()
    other.close()


@pytest.mark.parametrize("kwargs", [dict(env_agent_type="UAV_fixedwing", spatial_dim=2),
    dict(env_agent_type="particle", env_agent_action_type="actuator"),
    dict(env_agent_type="unknown"), dict(env_agent_action_type="unknown")])
def test_unsupported_model_control_combinations_fail_at_construction(kwargs):
    with pytest.raises(ValueError):
        make_env(**kwargs)


def test_native_effective_config_sets_uav_dimensions_and_task():
    from had_env.config import EnvConfig
    from had_env.simulation import Simulation
    config = EnvConfig(env_agent_type="UAV_quadrotor", spatial_dim=3, task_mode="damage")
    env = Simulation(1, 1, 1, effective_config=config)
    try:
        assert env.spatial_dim == 3
        assert env.task_mode == "damage"
        assert env.control_space.shape == (3,)
        env.reset(seed=17)
        assert all(entity.spatial_dim == 3 for entity in env.entities)
    finally:
        env.close()
