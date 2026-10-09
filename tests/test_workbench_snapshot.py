import copy
from dataclasses import make_dataclass
import json
import pickle

import numpy as np
import pytest

from had_env.grouping.actions import rule_grouping
from skyhad_workbench.recording import load_episode, serializable
from skyhad_workbench.session import SimulationSession, branch_from_snapshot
from skyhad_workbench.snapshot import decode_snapshot, encode_snapshot


def test_json_snapshot_preserves_types_rng_and_exact_future_transitions():
    with SimulationSession() as session:
        session.step()
        before = pickle.dumps(session.snapshot())
        ambient = pickle.dumps(np.random.get_state())
        document = json.loads(json.dumps(encode_snapshot(session.snapshot()), allow_nan=False))
        restored = decode_snapshot(document)
        assert type(restored["physical"].entity_states[0]["Health"]) is type(session.snapshot()["physical"].entity_states[0]["Health"])
        assert pickle.dumps(np.random.get_state()) == ambient
        env = branch_from_snapshot(restored)
        try:
            assert pickle.dumps(session.snapshot()) == before
            while not session.done:
                action = rule_grouping(session.state)
                reference = env.step(action)
                actual = session.step(action)
                assert reference[0].to_dict() == actual.state.to_dict()
                assert reference[3] == actual.info
        finally:
            env.close()


def test_initial_snapshot_and_branch_of_recorded_branch_resume_exactly(tmp_path):
    with SimulationSession() as session:
        initial = decode_snapshot(session.episode.metadata["initial_snapshot"])
        assert initial["physical"].step_count == 0
        session.step()
        with session.branch(728) as branch:
            branch.run(max_decisions=2)
            path = branch.save(tmp_path / "branch.json.gz")
            loaded = load_episode(path)
            decision = loaded.decisions[1]
            with SimulationSession.from_snapshot(decision["snapshot"], loaded.metadata["scenario"]) as reopened:
                assert serializable(reopened.state.to_dict()) == decision["state"]
                actual = reopened.step(decision["selected_plan"])
                assert actual.state.step == decision["step"] + decision["delta"]
                assert reopened.episode.frames[-1]["entities"] == loaded.frame_at(actual.state.step)["entities"]
                with reopened.branch(934) as nested:
                    assert nested.state.to_dict() == reopened.state.to_dict()
                    assert decode_snapshot(nested.episode.metadata["initial_snapshot"])["physical"].step_count == reopened.state.step


def test_snapshot_rejects_source_changes_executable_types_and_tampering():
    with SimulationSession() as session:
        document = encode_snapshot(session.snapshot())
    wrong = copy.deepcopy(document["source_identity"])
    wrong["behavior_hash"] = "different"
    with pytest.raises(ValueError, match="sources differ"):
        decode_snapshot(document, current_identity=wrong)
    broken = copy.deepcopy(document)
    broken["payload"]["kind"] = "pickle"
    with pytest.raises(ValueError, match="hash mismatch"):
        decode_snapshot(broken)
    from skyhad_workbench.snapshot import _hash
    broken["payload_hash"] = _hash(broken["payload"])
    with pytest.raises(ValueError, match="Unregistered"):
        decode_snapshot(broken)


def test_snapshot_rejects_malformed_shapes_before_restore():
    with SimulationSession() as session:
        snapshot = session.snapshot()
        snapshot["physical"].entity_states[0]["position"] = [1., 2.]
        encoded = encode_snapshot(snapshot)
    with pytest.raises(ValueError, match="xyz"):
        decode_snapshot(encoded)


def test_snapshot_cannot_encode_an_unregistered_class_with_a_trusted_name():
    # Name equality alone must never authorize a plugin's class for restoration.
    impostor = make_dataclass("HADStage3Snapshot", [("value", int)])
    with pytest.raises(ValueError, match="Unregistered snapshot dataclass"):
        encode_snapshot({"physical": impostor(123)})


@pytest.mark.parametrize("model", ["UAV_fixedwing", "UAV_quadrotor"])
def test_uav_portable_snapshot_restores_identical_rigid_trajectories_and_rejects_config_mismatch(model):
    from skyhad_workbench.protocols import ScenarioSpec, CUSTOM_PROTOCOL
    scenario = ScenarioSpec(red_count=2, blue_count=2, max_steps=4, command_interval=1,
                            protocol_id=CUSTOM_PROTOCOL, env_agent_type=model)
    with SimulationSession(scenario) as session:
        session.step()
        document = json.loads(json.dumps(encode_snapshot(session.snapshot())))
        restored = decode_snapshot(document)
        row = restored["physical"].entity_states[0]
        assert row["rigid_state"].shape == (13,)
        with SimulationSession.from_snapshot(document, scenario) as branch:
            while not session.done:
                action = rule_grouping(session.state)
                session.step(action)
                branch.step(action)
                assert branch.episode.frames[-1]["entities"] == session.episode.frames[-1]["entities"]
        restored["effective_config"]["scene_scale"] *= 2
        with pytest.raises(ValueError, match="configuration|config"):
            decode_snapshot(encode_snapshot(restored))


def test_uav_collision_snapshot_keeps_dead_native_and_rigid_velocities_synchronized():
    from skyhad_workbench.protocols import ScenarioSpec, CUSTOM_PROTOCOL
    scenario = ScenarioSpec(red_count=2, blue_count=2, max_steps=5, command_interval=1,
                            protocol_id=CUSTOM_PROTOCOL, env_agent_type="UAV_fixedwing")
    with SimulationSession(scenario) as session:
        world = session.env.adapter.env
        world.agents[0].reset([-15., 0., 100.], [30., 0., 0.])
        world.agents[1].reset([-500., 500., 100.], [30., 0., 0.])
        world.agents[2].reset([15., 0., 100.], [-30., 0., 0.])
        world.agents[3].reset([500., -500., 100.], [-30., 0., 0.])
        session.step()
        assert world.agents[0].Health == world.agents[2].Health == 0
        assert any(e["kind"] == "collision" for e in world.last_physics_events)
        assert not session.done
        document = encode_snapshot(session.snapshot())
        restored = decode_snapshot(document)
        for index in (0, 2):
            state = restored["physical"].entity_states[index]
            np.testing.assert_array_equal(state["rigid_state"][3:6], [0., 0., 0.])
            np.testing.assert_array_equal(state["velocity"], [0., 0., 0.])
        with SimulationSession.from_snapshot(document, scenario) as branch:
            action = rule_grouping(session.state)
            session.step(action)
            branch.step(action)
            assert branch.episode.frames[-1]["entities"] == session.episode.frames[-1]["entities"]
