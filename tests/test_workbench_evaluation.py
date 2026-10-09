from dataclasses import replace

import pytest

from skyhad_workbench.evaluation import EvaluationPlan, evaluate_policies, summarize_evaluation
from skyhad_workbench.protocols import CUSTOM_PROTOCOL, ProtocolSpec, ScenarioSpec, scenario_family


def test_split_contract_distinguishes_new_openings_from_structural_ood():
    train = scenario_family(31, 0, split="train")
    validation = scenario_family(31, 0, split="validation")
    test_id = scenario_family(31, 0, split="test")
    test_ood = scenario_family(31, 0, split="test", distribution="ood", cells=((12, 8),))
    plan = EvaluationPlan((train,), (validation,), (test_id, test_ood))
    assert EvaluationPlan.from_dict(plan.to_dict()) == plan
    assert train.opening_seed != test_id.opening_seed
    with pytest.raises(ValueError, match="OOD"):
        EvaluationPlan((train,), (), (replace(test_id, distribution="ood", scenario_id=""),))
    with pytest.raises(ValueError, match="opening"):
        EvaluationPlan((train,), (), (replace(train, split="test", scenario_id="other"),))


def test_native_v5_semantics_cannot_be_silently_relabelled():
    with pytest.raises(ValueError, match="new registered"):
        ProtocolSpec(firing_control="learned")
    with pytest.raises(ValueError, match="protocol_id"):
        ScenarioSpec(max_steps=1)
    custom = ScenarioSpec(max_steps=1, protocol_id=CUSTOM_PROTOCOL)
    with pytest.raises(ValueError, match="relabelled"):
        custom.episode_spec()


def test_evaluation_models_and_effective_config_are_structural_conditions():
    from skyhad_workbench.evaluation import scenario_signature
    from had_env.config import EnvConfig
    train = ScenarioSpec(protocol_id=CUSTOM_PROTOCOL, split="train",
                         target_positions=((-100., -50., 50.), (-100., 50., 50.)))
    held_out = replace(train, split="test", opening_seed=91, opponent_seed=92, scenario_id="")
    # Omitted defaults and explicitly written defaults describe the same physics.
    explicit_defaults = replace(held_out, env_config=EnvConfig(spatial_dim=3).to_dict(), scenario_id="")
    assert scenario_signature(held_out) == scenario_signature(explicit_defaults)
    EvaluationPlan((train,), (), (explicit_defaults,))
    for changes in ({"env_agent_type": "UAV_quadrotor"},
                    {"env_config": {"fire_range": 100.}}):
        changed = replace(held_out, **changes, scenario_id="")
        assert scenario_signature(changed) != scenario_signature(train)
        with pytest.raises(ValueError, match="ID must match"):
            EvaluationPlan((train,), (), (changed,))
        ood = replace(changed, distribution="ood", scenario_id="")
        EvaluationPlan((train,), (), (ood,))


def test_evaluation_target_health_matches_runtime_config_and_explicit_precedence():
    from had_env.config import initial_health
    from skyhad_workbench.evaluation import scenario_signature
    from skyhad_workbench.session import SimulationSession
    train = ScenarioSpec(protocol_id=CUSTOM_PROTOCOL, split="train", env_config={"target_health": 2.})
    held_out = replace(train, split="test", opening_seed=91, opponent_seed=92, scenario_id="")
    changed = replace(held_out, env_config={"target_health": 3.}, scenario_id="")
    with SimulationSession(train, record=False) as baseline, SimulationSession(changed, record=False) as other:
        assert baseline.env.target_health == 2.
        assert other.env.target_health == 3.
    assert scenario_signature(changed) != scenario_signature(train)
    with pytest.raises(ValueError, match="ID must match"):
        EvaluationPlan((train,), (), (changed,))
    EvaluationPlan((train,), (), (replace(changed, distribution="ood", scenario_id=""),))
    omitted = replace(held_out, env_config=None, scenario_id="")
    explicit_default = replace(omitted, target_health=initial_health, scenario_id="")
    configured_default = replace(omitted, env_config={"target_health": initial_health}, scenario_id="")
    assert scenario_signature(omitted) == scenario_signature(explicit_default) == scenario_signature(configured_default)
    explicit_override = replace(changed, target_health=2., scenario_id="")
    with SimulationSession(explicit_override, record=False) as override:
        assert override.env.target_health == 2.
    assert scenario_signature(explicit_override) == scenario_signature(train)
    EvaluationPlan((train,), (), (explicit_override,))


@pytest.mark.parametrize("config, explicit", [
    (None, {"horizon_policy": "red_win"}),
    ({"horizon_policy": "draw"}, {"horizon_policy": "red_win"}),
    (None, {"plane_altitude": 100.}),
    ({"scene_scale": .5}, {"plane_altitude": 50.}),
])
def test_equivalent_horizon_and_altitude_signatures_match_real_transitions(config, explicit):
    from had_env.grouping.actions import rule_grouping
    from skyhad_workbench.evaluation import scenario_signature
    from skyhad_workbench.session import SimulationSession
    omitted = ScenarioSpec(protocol_id=CUSTOM_PROTOCOL, red_count=2, blue_count=1,
                           max_steps=2, command_interval=1, env_config=config)
    written = replace(omitted, **explicit, scenario_id="")
    with SimulationSession(omitted) as baseline, SimulationSession(written) as other:
        assert baseline.env.horizon_policy == other.env.horizon_policy == "red_win"
        assert baseline.env.plane_altitude == other.env.plane_altitude
        assert baseline.env.effective_config == other.env.effective_config
        assert baseline.state.to_dict() == other.state.to_dict()
        action = rule_grouping(baseline.state)
        expected, actual = baseline.step(action), other.step(action)
        assert actual.state.to_dict() == expected.state.to_dict()
        assert actual.info == expected.info
        assert baseline.episode.frames[-1]["entities"] == other.episode.frames[-1]["entities"]
    assert scenario_signature(omitted) == scenario_signature(written)


@pytest.mark.parametrize("explicit", [{"horizon_policy": "red_win"}, {"plane_altitude": 100.}])
@pytest.mark.parametrize("distribution", ["id", "ood"])
def test_equivalent_settings_cannot_reuse_training_openings(explicit, distribution):
    train = ScenarioSpec(protocol_id=CUSTOM_PROTOCOL, split="train")
    reused = replace(train, **explicit, split="test", distribution=distribution, scenario_id="")
    with pytest.raises(ValueError, match="exact opening"):
        EvaluationPlan((train,), (), (reused,))


@pytest.mark.parametrize("explicit", [{"horizon_policy": "red_win"}, {"plane_altitude": 100.}])
def test_equivalent_settings_require_id_for_new_openings(explicit):
    train = ScenarioSpec(protocol_id=CUSTOM_PROTOCOL, split="train")
    held_out = replace(train, **explicit, split="test", opening_seed=91, opponent_seed=92, scenario_id="")
    with pytest.raises(ValueError, match="OOD must hold out"):
        EvaluationPlan((train,), (), (replace(held_out, distribution="ood", scenario_id=""),))
    EvaluationPlan((train,), (), (held_out,))


@pytest.mark.parametrize("changes", [{"horizon_policy": "draw"}, {"horizon_policy": "blue_win"},
                                    {"plane_altitude": 200.}])
def test_different_effective_horizon_and_altitude_require_ood(changes):
    from skyhad_workbench.evaluation import scenario_signature
    from skyhad_workbench.session import SimulationSession
    train = ScenarioSpec(protocol_id=CUSTOM_PROTOCOL, split="train", spatial_dim=2,
                         red_count=2, blue_count=1, max_steps=1, command_interval=1)
    held_out = replace(train, **changes, split="test", opening_seed=91, opponent_seed=92, scenario_id="")
    with SimulationSession(held_out, record=False) as session:
        assert session.env.horizon_policy == changes.get("horizon_policy", "red_win")
        assert session.env.plane_altitude == changes.get("plane_altitude", 100.)
        assert session.step().done
    assert scenario_signature(held_out) != scenario_signature(train)
    with pytest.raises(ValueError, match="ID must match"):
        EvaluationPlan((train,), (), (held_out,))
    EvaluationPlan((train,), (), (replace(held_out, distribution="ood", scenario_id=""),))


def test_cross_seed_means_are_equal_weighted_and_missing_pairs_are_explicit():
    def row(method, seed, scenario, win):
        return dict(method_id=method, training_seed=seed, scenario_id=scenario,
                    split="test", distribution="id", success_native=win,
                    opening_seed=1, opponent_seed=2)
    rows = [row("new", 1, "a", True), row("new", 2, "a", False),
            row("new", 2, "b", False), row("rule", 1, "a", False)]
    summary = summarize_evaluation(rows)
    new = next(group for group in summary["groups"] if group["method_id"] == "new")
    assert new["mean_seed_success_rate"] == .5  # Not the pooled 1/3.
    assert not new["balanced_scenario_coverage"] and new["across_seed_t95"] is None
    assert summary["paired"][0]["pairs"] == 1
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_evaluation(rows + rows[:1])


def test_small_paired_evaluation_uses_identical_openings_and_saves_records(tmp_path):
    scenarios = [ScenarioSpec(protocol_id=CUSTOM_PROTOCOL, max_steps=1, red_count=2, blue_count=1)]
    result = evaluate_policies({"rule": "rule", "grand": "grand"}, scenarios,
                               training_seeds=(1, 2), record_dir=tmp_path)
    assert len(result["episodes"]) == 4
    assert result["paired"][0]["pairs"] == 2
    assert all(row["physical_steps"] == 1 for row in result["episodes"])
    assert len(list(tmp_path.glob("*.json.gz"))) == 4
    contents = {path: path.read_bytes() for path in tmp_path.glob("*.json.gz")}
    with pytest.raises(FileExistsError, match="new output"):
        evaluate_policies({"rule": "rule"}, scenarios, training_seeds=(1,), record_dir=tmp_path)
    assert {path: path.read_bytes() for path in contents} == contents
