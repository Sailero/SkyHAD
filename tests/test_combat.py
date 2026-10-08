"""Independent golden trajectories, combat boundaries, events and task termination."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from make_env import make_env
from had_env.simulation import Simulation
from had_env.geometry import attack_intensity_ratio
from had_env.config import CORE_VERSION


REFERENCE = json.loads((Path(__file__).parent / "data" / "physics_reference.json").read_text(encoding="utf-8"))


def _world_rows(env):
    return [[list(map(float, x.position)), list(map(float, x.velocity)), float(x.Health),
             bool(getattr(x, "IsFire", False)), bool(getattr(x, "IsDisturb", False))]
            for x in env.simulation.entities]


def _place(env, positions):
    for e, (p, v) in zip(env.simulation.entities, positions):
        e.reset(list(p), list(v))
    env.simulation.reset_episode_state()


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_physics_matches_original_recorded_trajectory(case):
    env = make_env(**case["kwargs"], max_cycles=50)
    obs, _ = env.reset(seed=case["seed"])
    if case["preset"]:
        _place(env, case["preset"])
    assert _world_rows(env) == case["initial_world"]
    np.testing.assert_allclose(env.simulation.get_observation(), case["initial_observation"], rtol=0, atol=0)
    for reference in case["frames"]:
        acting = env.agents.copy()
        obs, rewards, terms, truncs, infos = env.step({a: reference["actions"][env.agent_name_mapping[a]] for a in acting})
        for actual, expected in zip(_world_rows(env), reference["world"]):
            np.testing.assert_allclose(actual[0], expected[0], rtol=0, atol=1e-11)
            np.testing.assert_allclose(actual[1], expected[1], rtol=0, atol=1e-11)
            assert actual[2:] == expected[2:]
        np.testing.assert_allclose(env.state(), reference["state"], rtol=1e-7, atol=1e-7)
        for a in acting:
            i = env.agent_name_mapping[a]
            np.testing.assert_allclose(obs[a], reference["observations"][i], rtol=1e-7, atol=1e-7)
            side = "Red" if a.startswith("red") else "Blue"
            side_index = i - (case["kwargs"]["red_count"] if side == "Blue" else 0)
            assert rewards[a] == reference["rewards"]["RealReward"][side]
            np.testing.assert_allclose(infos[a]["LatentReward"], reference["rewards"]["LatentReward"][side][side_index], rtol=0, atol=1e-12)
            assert terms[a] == reference["done"][i]
            assert not truncs[a]
        assert env.outcome_red == reference["outcome"]
    env.close()


def test_horizon_does_not_invent_win_and_terminal_reward_paid_once():
    env = make_env(red_count=1, blue_count=1, max_cycles=1, initialization="uniform")
    env.reset(seed=22)
    _, rewards, terms, truncs, infos = env.step({a: 0 for a in env.agents})
    assert not any(terms.values()) and all(truncs.values())
    assert all(r == 0 for r in rewards.values()) and env.outcome_red == 0
    assert env.agents == [] and env.step({}) == ({}, {}, {}, {}, {})
    env.reset(seed=22)
    env.simulation.blue_agents[0].Health = 0
    _, rewards, terms, truncs, _ = env.step({a: 0 for a in env.agents})
    assert rewards == {"red_0": 10., "blue_0": -10.}
    assert all(terms.values()) and not any(truncs.values())
    assert env.step({})[1] == {}
    env.close()


def physical_state(env):
    return np.asarray([entity.position + list(entity.velocity) + [float(entity.Health)]
                       for entity in env.entities])


def set_pose(agent, position, velocity=(20., 0., 0.)):
    agent.position = list(position)
    agent.velocity = list(velocity)


def test_attack_boundary_scalar_vector_agree_and_remain_finite():
    values = [199.999, 200., 400., 400.001]
    expected = [attack_intensity_ratio(value) for value in values]
    assert expected == [1, 1., 0, 0]
    np.testing.assert_array_equal(attack_intensity_ratio(values), expected)
    np.testing.assert_array_equal(attack_intensity_ratio(np.asarray(values)), expected)


def test_exact_boundary_in_a_physical_step_is_finite():
    env = Simulation(1, 1, 1)
    env.reset(seed=4)
    set_pose(env.red_agents[0], [-1500., 0., 100.])
    set_pose(env.blue_agents[0], [500., 0., 100.])
    set_pose(env.targets[0], [0., 0., 100.], [0., 0., 0.])
    env.step_physics(np.zeros((2, 3)))
    assert np.isfinite(physical_state(env)).all()
    assert env.targets[0].Health == 2.


def test_heterogeneous_step_preserves_each_roles_existing_reward():
    env = Simulation(1, 1, 1, red_scout_n=1, red_disturb_n=1,
                 blue_scout_n=1, blue_disturb_n=1)
    env.reset(seed=7, evaluate=True)
    result = env.step(np.zeros((6, 3)))
    rewards = result[3]
    assert len(result) == 6
    for side, agents in (("Red", env.red_agents), ("Blue", env.blue_agents)):
        assert np.asarray(rewards["LatentReward"][side]).shape == (3, 4)
        for agent, reward in zip(agents, rewards["LatentReward"][side]):
            if agent.Health > 0 and agent.Type != "Attack":
                assert reward[:3] == [agent.get_reward(env.entities), 0., 0.]


@pytest.mark.parametrize("task_type", ["Training", "Normal Showcase"])
def test_reset_clears_actions_events_and_repairs_first_step_collision(task_type):
    env = Simulation(2, 1, 1, task_type=task_type)
    env.record_events = True
    env.reset(seed=4)
    for agent in env.agents:
        agent.Health = 0
        agent.IsFire = True
        agent.acceleration = [1., 2., 3.]
    env.targets[0].Health = 0
    env.update_alive_agents()
    env.physics_step_count = 17
    env.last_physics_events = [{"kind": "old"}]
    env.reset(seed=4)
    assert env.alive_agents == env.agents
    assert env.alive_targets == env.targets
    assert env.physics_step_count == 0 and env.last_physics_events == []
    assert all(agent.acceleration == [0., 0., 0.] and not agent.IsFire for agent in env.agents)
    for agent in env.red_agents:
        set_pose(agent, [-500., 0., 100.])
    set_pose(env.blue_agents[0], [2000., 0., 100.])
    env.step_physics(np.zeros((3, 3)))
    assert [agent.Health for agent in env.agents] == [0, 0, 1]
    collisions = [event for event in env.last_physics_events if event["kind"] == "collision"]
    assert len(collisions) == 2
    assert {event["target_id"] for event in collisions} == {0, 1}
    assert all(event["step"] == 1 and event["damage"] == 1 for event in collisions)


@pytest.mark.parametrize("invalid", [
    [], [[0, 0, 0]], np.zeros((3, 3)), np.zeros((2, 2)),
    [[0, 0], [0, 0, 0]], [[float("nan"), 0, 0], [0, 0, 0]],
    [[float("inf"), 0, 0], [0, 0, 0]], [["1", "0", "0"], ["0", "0", "0"]],
    [[1j, 0, 0], [0, 0, 0]],
])
def test_invalid_action_rejected_before_any_physical_mutation(invalid):
    env = Simulation(1, 1, 1)
    env.reset(seed=4)
    env.record_events = True
    before = physical_state(env).copy()
    with pytest.raises(ValueError):
        env.step_physics(invalid)
    np.testing.assert_array_equal(before, physical_state(env))
    assert env.physics_step_count == 0
    assert env.last_physics_events == []
    assert all(agent.acceleration == [0., 0., 0.] for agent in env.agents)


@pytest.mark.parametrize("seed", [4, 31, 8123])
def test_optional_recording_preserves_current_physics_trajectory(seed):
    env = Simulation(4, 4, 2)
    env.reset(seed=seed, evaluate=True)
    silent = copy.deepcopy(env)
    env.record_events = True
    rng = np.random.default_rng(seed)
    for _ in range(30):
        actions = rng.uniform(-1, 1, (8, 3)).astype(np.float32)
        env.step_physics(actions)
        silent.step_physics(actions)
        np.testing.assert_array_equal(physical_state(env), physical_state(silent))
        assert env.is_terminal() == silent.is_terminal()
        assert [a.IsFire for a in env.agents] == [a.IsFire for a in silent.agents]
        assert silent.last_physics_events == []
        json.dumps(env.last_physics_events, allow_nan=False)


def test_damage_events_explain_snapshot_distance_and_firing_self_destruction():
    env = Simulation(1, 1, 1)
    env.reset(seed=4)
    env.record_events = True
    set_pose(env.red_agents[0], [0., 0., 100.])
    set_pose(env.blue_agents[0], [100., 0., 100.])
    set_pose(env.targets[0], [-2000., 0., 100.], [0., 0., 0.])
    env.step_physics(np.zeros((2, 3)))
    events = env.last_physics_events
    hit = next(event for event in events if event["kind"] == "attack_damage")
    # Hit distance is 100 at task start; the blue rendered endpoint is 120.
    assert (hit["source_id"], hit["target_id"], hit["distance"]) == (0, 1, 100.)
    assert hit["source_position"] == [20., 0., 100.]
    assert hit["target_position"] == [120., 0., 100.]
    assert (hit["health_before"], hit["health_after"], hit["damage"]) == (1., 0., 1.)
    assert hit["damage_scope"] == "source_unclipped" and hit["simultaneous"]
    assert any(e["kind"] == "fire" and e["source_id"] == 0 for e in events)
    assert any(e["kind"] == "self_destruct" and e["target_id"] == 0 for e in events)
    assert env.core_version == CORE_VERSION
    assert env.get_info()["core_version"] == CORE_VERSION
