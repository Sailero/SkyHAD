"""The public example preserves native team metrics and freezes only Blue code."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from had_env import make_env
from had_env import __version__


def _example():
    path = Path(__file__).resolve().parents[1] / "examples/defender_evaluation.py"
    assert path.is_file(), "The self-contained defender evaluation example is missing"
    spec = importlib.util.spec_from_file_location("defender_evaluation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _place(env, poses):
    for entity, (position, velocity) in zip(env.simulation.entities, poses):
        entity.reset(position, velocity)
    env.simulation.reset_episode_state()


def _fixture_factory(monkeypatch, example, poses):
    def factory(**config):
        env = make_env(**config)
        reset = env.reset

        def placed_reset(**kwargs):
            reset(**kwargs)
            _place(env, poses)
            observations = env._observations(env.simulation.get_observation(), env.agents)
            return observations, {a: env._info(a) for a in env.agents}

        env.reset = placed_reset
        return env

    monkeypatch.setattr(example, "make_env", factory)


@pytest.mark.parametrize("model", ["particle", "UAV_fixedwing", "UAV_quadrotor"])
@pytest.mark.parametrize("task", ["survival", "damage"])
def test_six_acceleration_cells_are_seed_reproducible(model, task):
    example = _example()
    config = dict(env_agent_type=model, task_mode=task, max_cycles=3)
    first = example.run_episode(config, 11)
    example.run_episode(config, 29, example.red_coast)
    second = example.run_episode(config, 11)
    assert first == second
    assert first["steps"] == 3
    assert first["global_truncated"] and not first["global_terminated"]
    assert first["setup"]["run_config"]["continuous"] is True
    assert first["setup"]["run_config"]["reward_weights"] is None
    assert first["setup"]["run_config"]["red_count"] == 4
    if task == "damage":
        assert first["red_return"] == pytest.approx(-first["target_damage"])


def test_blue_uses_same_pre_step_observation_when_red_callable_changes(monkeypatch):
    example = _example()
    submitted = []

    def factory(**config):
        env = make_env(**config)
        step = env.step

        def capture(actions):
            submitted.append({a: np.asarray(v).copy() for a, v in actions.items()
                              if a.startswith("blue_")})
            return step(actions)

        env.step = capture
        return env

    monkeypatch.setattr(example, "make_env", factory)
    first = example.run_episode(dict(max_cycles=1), 11, example.red_intercept)
    np.random.default_rng(91).random(100)
    second = example.run_episode(dict(max_cycles=1), 11, example.red_coast)
    for name in submitted[0]:
        np.testing.assert_array_equal(submitted[0][name], submitted[1][name])
    assert first["setup"]["opponent"] == second["setup"]["opponent"]


@pytest.mark.parametrize("model", ["particle", "UAV_fixedwing", "UAV_quadrotor"])
def test_navigation_is_bounded_and_closes_toward_asset(model):
    example = _example()
    env = make_env(env_agent_type=model, continuous=True, red_count=1,
                   blue_count=1, target_count=1, max_cycles=3)
    try:
        env.reset(seed=11)
        scale = env.effective_config.scene_scale
        speed = env.effective_config.preset.reference_speed
        points = [np.asarray(p) * scale for p in
                  ([-1500., -1500., 500.], [-1000., 0., 500.], [-300., 200., 500.])]
        _place(env, [(points[0], [0., 0., 0.]), (points[1], [speed, 0., 0.]),
                     (points[2], [0., 0., 0.])])
        initial = np.linalg.norm(points[2] - points[1])
        for step in range(3):
            obs = env._observations(env.simulation.get_observation(), env.agents)
            info = env._info("blue_0")
            action = example.blue_navigation(obs["blue_0"], info, context=env.effective_config)
            assert env.action_space("blue_0").contains(action)
            assert np.linalg.norm(action) <= 1.000001
            if step == 0:
                assert action[1] > 0  # Initially steer north; later feedback may brake.
            env.step({"red_0": np.zeros(3, np.float32), "blue_0": action})
        final = np.linalg.norm(np.asarray(env.simulation.targets[0].position)
                               - env.simulation.blue_agents[0].position)
        assert final < initial - 10 * scale
    finally:
        env.close()


def _attack_poses():
    return [([-1000., 1000., 100.], [35., 0., 0.])] * 2 + [
        ([350., y, 100.], [-120., 0., 0.]) for y in (-100., 0., 100.)
    ] + [([0., 0., 100.], [0., 0., 0.])]


@pytest.mark.parametrize("task", ["survival", "damage"])
def test_team_score_once_per_step_after_red_casualties(monkeypatch, task):
    example = _example()
    _fixture_factory(monkeypatch, example, _attack_poses())
    row = example.run_episode(dict(red_count=2, blue_count=3, target_count=1,
                                   task_mode=task, max_cycles=2), 11, example.red_coast)
    assert row["steps"] == 2
    assert row["global_terminated"] and not row["global_truncated"]
    assert row["target_damage"] == pytest.approx(3.)  # Three simultaneous full-strength hits.
    if task == "survival":
        assert row["outcome"] == "loss"
        assert row["termination_reason"] == "target_destroyed"
        assert row["red_return"] == -10.  # Red was removed one step before this terminal reward.
    else:
        assert row["red_return"] == -3.
        assert row["blue_return"] == 3.
        assert row["termination_reason"] == "blue_attackers_destroyed"
        assert example.summarize([row])["mean_target_damage"] == 3.


def test_damage_return_accumulates_increments_instead_of_cumulative_counters(monkeypatch):
    example = _example()
    poses = [([-1000., 1000., 100.], [35., 0., 0.])] * 2 + [
        ([250., -100., 100.], [-120., 0., 0.]),
        ([450., 100., 100.], [-120., 0., 0.]),
        ([0., 0., 100.], [0., 0., 0.])]
    _fixture_factory(monkeypatch, example, poses)
    row = example.run_episode(dict(red_count=2, blue_count=2, target_count=1,
                                   task_mode="damage", max_cycles=5), 11, example.red_coast)
    assert row["steps"] == 3  # Separate hits at steps 1 and 3; cumulative counters persist at step 2.
    assert row["target_damage"] == 2.
    assert row["red_return"] == -2.


def test_individual_casualty_at_sampling_horizon_is_truncation(monkeypatch):
    example = _example()
    _fixture_factory(monkeypatch, example, _attack_poses())
    row = example.run_episode(dict(red_count=2, blue_count=3, target_count=1,
                                   max_cycles=1), 11, example.red_coast)
    assert row["outcome"] == "truncated"
    assert not row["global_terminated"] and row["global_truncated"]
    assert row["termination_reason"] is None
    assert row["red_return"] == 0.
    completed = example.run_episode(dict(red_count=2, blue_count=3, target_count=1,
                                         max_cycles=2), 11, example.red_coast)
    summary = example.summarize([row, completed])
    assert summary["episodes"] == 2 and summary["losses"] == 1 and summary["truncations"] == 1
    assert summary["truncation_fraction_all_episodes"] == .5


def test_natural_win_at_horizon_takes_priority_and_reward_is_not_multiplied(monkeypatch):
    example = _example()
    _fixture_factory(monkeypatch, example, [
        ([0., -100., 100.], [35., 0., 0.]), ([0., 100., 100.], [35., 0., 0.]),
        ([100., 0., 100.], [-35., 0., 0.]), ([-2000., -1500., 100.], [0., 0., 0.])])
    row = example.run_episode(dict(red_count=2, blue_count=1, target_count=1,
                                   max_cycles=1), 11, example.red_coast)
    assert row["outcome"] == "win"
    assert row["global_terminated"] and not row["global_truncated"]
    assert row["red_return"] == 10.
    summary = example.summarize([row])
    assert summary == dict(episodes=1, wins=1, losses=0, truncations=0,
                           win_fraction_all_episodes=1., truncation_fraction_all_episodes=0.)


def test_cli_outputs_reproducible_complete_configuration_from_fresh_process(tmp_path):
    example = _example()
    path = Path(example.__file__)
    command = [sys.executable, "-B", str(path), "--seeds", "11", "12", "--horizon", "1",
               "--task", "damage", "--red-policy", "coast"]
    environment = dict(os.environ, PYTHONPATH=str(path.parents[1]))
    first = subprocess.run(command, cwd=tmp_path, env=environment, capture_output=True, text=True, check=True)
    second = subprocess.run(command, cwd=tmp_path, env=environment, capture_output=True, text=True, check=True)
    assert json.loads(first.stdout) == json.loads(second.stdout)
    payload = json.loads(first.stdout)
    assert payload["version"] == __version__
    assert payload["seeds"] == [11, 12]
    assert payload["summary"]["episodes"] == 2
    assert payload["summary"]["truncations"] == 2
    assert "wins" not in payload["summary"]
    assert payload["run_config"]["target_health"] == 2.
    assert payload["run_config"]["red_scouts"] == 0
    assert payload["effective_config"]["world_bounds"] == [[-2500., 2500.], [-2500., 2500.], [0., 2500.]]
    assert payload["opponent"]["id"] == "nearest-target-velocity-v1"
