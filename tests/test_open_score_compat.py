"""Open-SCORE public contract, checked against frozen upstream outputs."""
import copy
import importlib
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest


def api():
    assert importlib.util.find_spec("had_env.open_score_compat") is not None, "Open-SCORE compatibility API is missing"
    return importlib.import_module("had_env.open_score_compat")


@pytest.fixture
def reference():
    return json.loads((Path(__file__).parent / "data/open_score_reference.json").read_text(encoding="utf-8"))


def test_original_defaults_and_training_profile():
    package = api()
    env = package.make_open_score_env()
    assert env.fixed_scale is None
    assert env.get_env_info() == {
        "n_agents": 50, "n_entities": 112, "n_actions": 9, "entity_shape": 10,
        "state_shape": 1123, "obs_shape": 1120, "episode_limit": 100,
        "gt_mask_avail": False, "feature_layout": "had", "n_tasks": 12,
    }
    assert env.wrapper.command_interval == 5
    assert env.wrapper.gamma == .99
    assert env.wrapper.fold_wipeout_tail is True
    assert env.wrapper.shaping_coef == 0
    assert env.wrapper.shaping_range == 4000
    assert env.wrapper.blue_upper == "reactive"
    assert env.wrapper.blue_lower == "rush"
    train = package.make_open_score_env(profile="training", episode_limit=7, max_steps=99)
    assert train.get_env_info()["n_entities"] == 23
    assert train.episode_limit == train.wrapper.max_steps == 7
    assert train.wrapper.shaping_coef == 1
    custom = package.make_open_score_env(profile="training", pad=(4, 5, 2), shaping_coef=.2, decision_interval=3)
    assert custom.wrapper.command_interval == 3
    assert custom.wrapper.shaping_coef == .2
    assert (custom.n_agents, custom.n_blue, custom.n_targets) == (4, 5, 2)
    for item in (env, train, custom):
        item.close()


def test_seed_stream_and_mixed_scale_sampling_match_upstream(reference):
    env = api().make_open_score_env()
    results = []
    try:
        saved = env.get_rng_state()
        for _ in reference["reset_sequence"]:
            env.reset()
            results.append({"seed": env.wrapper.episode_seed, "scale": env.wrapper.scale.as_dict()})
        assert results == reference["reset_sequence"]
        env.set_rng_state(saved)
        for expected in results:
            env.reset()
            assert env.wrapper.episode_seed == expected["seed"]
            assert env.wrapper.scale.as_dict() == expected["scale"]
    finally:
        env.close()


@pytest.mark.parametrize("case_index", [0, 1])
def test_entities_rewards_masks_and_diagnostics_match_literal_upstream(reference, case_index):
    case = reference["episodes"][case_index]
    env = api().make_open_score_env(**case["kwargs"])
    try:
        entities, masks = env.reset(seed=case["seed"], evaluate=True, retain_trajectory=True)
        np.testing.assert_allclose(entities, case["initial_entities"], rtol=0, atol=1e-7)
        np.testing.assert_array_equal(masks["entity_mask"], case["initial_entity_mask"])
        np.testing.assert_array_equal(env.get_initial_agent_mask(), case["initial_agent_mask"])
        for step in case["steps"]:
            reward, done, info = env.step(step["actions"])
            assert reward == pytest.approx(step["reward"], abs=1e-9)
            assert done == step["done"]
            for key in ("terminated", "truncated", "bootstrap_mask", "folded_steps", "step", "episode_limit"):
                assert info[key] == step["info"][key]
            np.testing.assert_allclose(info["task_rewards"], step["info"]["task_rewards"], rtol=0, atol=1e-9)
            assert sum(info["task_rewards"]) == pytest.approx(reward, abs=1e-9)
            assert info["tasks_terminated"] == step["info"]["tasks_terminated"]
            assert env.get_task_masks()["hier_decision"].tolist() == step["hier_decision"]
        np.testing.assert_allclose(env.get_entities(), case["final_entities"], rtol=0, atol=1e-7)
        np.testing.assert_allclose(env.get_state(), case["final_state"], rtol=0, atol=1e-7)
        actual = env.episode_summary()
        for key, expected in case["summary"].items():
            if isinstance(expected, float):
                assert actual[key] == pytest.approx(expected, abs=1e-9), key
            else:
                assert actual[key] == expected, key
        assert env.get_trajectory() == case["trajectory"]
        assert env.save_replay() == case["trajectory"]
        assert env.get_stats() == actual
        assert env.get_agg_stats([actual]) == {}
    finally:
        env.close()


def test_padding_deaths_observations_and_task_masks():
    env = api().make_open_score_env(scale=(4, 4, 2), pad="train")
    try:
        env.reset(seed=175)
        native = env.wrapper.adapter.env
        native.red_agents[0].Health = 0
        native.blue_agents[1].Health = 0
        native.update_alive_agents()
        env.wrapper._refresh_features()
        absent = env.get_masks()["entity_mask"]
        assert absent.tolist() == [1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1]
        np.testing.assert_array_equal(env.get_agent_mask(), [0, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(env.get_initial_agent_mask(), [0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        assert not env.get_obs_agent(0).any()
        np.testing.assert_array_equal(env.get_obs_agent(1), env.get_entities().ravel())
        assert env.get_avail_agent_actions(0).tolist() == [1, 0, 0, 0, 0, 0, 0, 0, 0]
        assert env.get_avail_agent_actions(1).tolist() == [1] * 9
        obs_mask = env.get_masks()["obs_mask"]
        assert obs_mask[1, 2] == 0 and obs_mask[1, 0] == 1 and obs_mask[0, 1] == 1
        tasks = env.get_task_masks()
        assert tasks["task_mask"].tolist() == [0, 0, 1]
        assert tasks["entity2task_mask"][1].tolist() == [0, 0, 1]
        assert tasks["entity2task_mask"][20].tolist() == [0, 1, 1]
        assert tasks["entity2task_mask"][21].tolist() == [1, 0, 1]
        assert tasks["entity2task_mask"][0].tolist() == [1, 1, 1]
        assert env.get_policy_state().red[0].alive is False
        owned = env.get_entities()
        owned[:] = 99
        assert not np.any(env.get_entities() == 99)
    finally:
        env.close()


def test_wipeout_folding_and_shaping_match_upstream(reference):
    case = reference["folded"]
    env = api().make_open_score_env(**case["kwargs"])
    try:
        env.reset(seed=case["seed"], evaluate=True, retain_trajectory=True)
        native = env.wrapper.adapter.env
        for agent in native.red_agents:
            agent.Health = 0
        target = np.asarray(native.targets[0].position)
        for i, agent in enumerate(native.blue_agents):
            agent.position = (target + np.array([700 + i * 200, i * 250, 0])).tolist()
            agent.velocity = [-60., 0., 0.]
        native.update_alive_agents()
        env.wrapper._refresh_features()
        reward, done, info = env.step([0] * env.n_agents)
        assert reward == pytest.approx(case["reward"], abs=1e-9)
        assert done and info["terminated"] and not info["truncated"]
        assert info["bootstrap_mask"] == 0
        assert info["folded_steps"] == case["info"]["folded_steps"]
        np.testing.assert_allclose(info["task_rewards"], case["info"]["task_rewards"], rtol=0, atol=1e-9)
        assert env.get_trajectory() == case["trajectory"]
        assert env.episode_summary()["D"] == case["summary"]["D"]
    finally:
        env.close()


def test_nv1_coverage_plan_and_actions_match_upstream(reference):
    package = api()
    rules = importlib.import_module("had_env.open_score_compat.rules")
    env = package.make_open_score_env(scale=(6, 4, 2), pad="train")
    try:
        env.reset(seed=177)
        policy = rules.CoveragePolicy()
        executor = rules.CoverageExecutor(policy=policy)
        grouping = policy.act(env.get_policy_state())
        assert json.loads(json.dumps(policy.last_plan)) == reference["nv1"]["plan"]
        assert grouping.to_dict() == reference["nv1"]["grouping"]
        actions = executor.act(env.wrapper.adapter, grouping)
        assert {str(key): value for key, value in actions.items()} == reference["nv1"]["actions"]
        assert len(set(policy.last_plan["assignment"].values())) == 4
        assert len(policy.last_plan["assignment"]) == 6
    finally:
        env.close()


def test_rng_isolation_and_wrapper_snapshot_replay():
    package = api()
    random.seed(921)
    np.random.seed(922)
    python_rng = copy.deepcopy(random.getstate())
    numpy_rng = copy.deepcopy(np.random.get_state())
    torch_module = sys.modules.get("torch")
    torch_rng = None if torch_module is None else torch_module.random.get_rng_state().clone()
    env = package.make_open_score_env(scale=(4, 4, 1), pad="train", max_steps=5)
    try:
        env.reset(seed=199, evaluate=True, retain_trajectory=True)
        for actual, expected in zip(np.random.get_state(), numpy_rng):
            np.testing.assert_equal(actual, expected)
        snapshot = env.wrapper.snapshot()
        result = env.step([0] * 10)
        entities = env.get_entities()
        np.random.random(3)
        latest = np.random.get_state()
        env.wrapper.restore(snapshot)
        for actual, expected in zip(np.random.get_state(), latest):
            np.testing.assert_equal(actual, expected)
        restored = env.step([0] * 10)
        assert restored == result
        np.testing.assert_array_equal(env.get_entities(), entities)
        assert random.getstate() == python_rng
        np.random.set_state(numpy_rng)
        assert sys.modules.get("torch") is torch_module
        if torch_module is not None:
            assert torch_module.equal(torch_module.random.get_rng_state(), torch_rng)
    finally:
        env.close()


@pytest.mark.parametrize("kwargs", [
    {"spatial_dim": 3}, {"continuous": True}, {"continuous_actions": True},
    {"task_mode": "survival"}, {"name": "uav"}, {"entity_scheme": False},
    {"env_agent_type": "UAV_fixedwing"}, {"env_agent_type": "UAV_quadrotor"},
    {"env_agent_action_type": "actuator"}, {"env_agent_action_type": "position"},
])
def test_unsupported_interfaces_are_rejected(kwargs):
    package = api()
    with pytest.raises(ValueError, match="2D|discrete|damage|HAD|entity scheme"):
        package.make_open_score_env(**kwargs)


def test_action_mapping_rejects_vertical_controls():
    package = api()
    assert package.PLANAR_NATIVE_IDS.tolist() == [0, 2, 5, 8, 11, 16, 19, 22, 25]
    np.testing.assert_array_equal(package.native_actions_to_planar([0, 2, 25]), [0, 1, 8])
    with pytest.raises(ValueError, match="3D"):
        package.native_actions_to_planar([1])
