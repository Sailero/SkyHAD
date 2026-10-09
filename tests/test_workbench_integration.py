import copy
import json
from pathlib import Path
import pickle
import time

import numpy as np
import pytest

from skyhad_workbench.agent_session import FlightScenarioSpec, FlightSession
from skyhad_workbench.cli import main
from skyhad_workbench.identity import source_identity
from skyhad_workbench.playback import display_frame
from skyhad_workbench.recording import ReplayEpisode, load_episode


def test_installed_cli_defaults_write_to_calling_directory_not_site_packages(tmp_path, monkeypatch):
    from skyhad_workbench.cli import _output_root
    caller = tmp_path / "experiment"
    caller.mkdir()
    monkeypatch.chdir(caller)
    installed = tmp_path / "site-packages" / "skyhad_workbench" / "cli.py"
    assert _output_root(installed) == caller
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "pyproject.toml").write_text("[project]\nname = 'skyhad-workbench'\n")
    assert _output_root(checkout / "skyhad_workbench" / "cli.py") == checkout


def test_native_heterogeneous_two_sided_actions_and_branch():
    session = FlightSession(FlightScenarioSpec(red_attackers=2, blue_attackers=2, red_scouts=1,
                                               blue_disturbers=1, max_steps=4))
    try:
        before = pickle.dumps(session.snapshot())
        with pytest.raises(ValueError):
            session.step({"red": [0, 0, 0], "blue": [99, 0, 0]})
        assert before == pickle.dumps(session.snapshot())
        session.step({"red": [0, 0, 0], "blue": [0, 0, 0]})
        child = session.branch()
        try:
            first, second = session.run(), child.run()
            assert first.frames == second.frames
            assert first.metadata["truncated"] and not first.metadata["terminated"]
            assert first.metadata["success_native"] is None
            assert len(first.frames) == 5
            assert {e["role"] for e in first.frames[0]["entities"]} >= {"Attack", "Scout", "Disturb"}
            assert first.frames[1]["actions"]["red"] and first.frames[1]["actions"]["blue"]
            with pytest.raises(RuntimeError):
                session.branch()
        finally:
            child.close()
    finally:
        session.close()
    with pytest.raises(RuntimeError):
        session.step()


@pytest.mark.parametrize("model", ["UAV_fixedwing", "UAV_quadrotor"])
@pytest.mark.parametrize("mode", ["acceleration", "actuator", "position"])
def test_native_uav_modes_record_rigid_state_and_branch_exactly(model, mode):
    session = FlightSession(FlightScenarioSpec(red_attackers=1, blue_attackers=1, max_steps=3,
                            env_agent_type=model, env_agent_action_type=mode, action_mode="continuous_native"))
    try:
        row = session.recorder.episode.frames[0]["entities"][0]
        assert row["env_agent_type"] == model and len(row["rigid_state"]) == 13
        assert session.recorder.episode.metadata["effective_config"]["env_agent_action_type"] == mode
        session.step()
        child = session.branch()
        try:
            assert session.run().frames == child.run().frames
        finally:
            child.close()
    finally:
        session.close()


def test_cli_selects_spatial_model_and_native_box_mode():
    from skyhad_workbench.cli import parser, _scenario
    args = parser().parse_args(["record", "--native", "--agent-type", "UAV_quadrotor",
                              "--agent-action-type", "actuator"])
    value = _scenario(args)
    assert value["spatial_dim"] == 3
    assert value["env_agent_type"] == "UAV_quadrotor"
    assert value["env_agent_action_type"] == "actuator"
    assert value["action_mode"] == "continuous_native"
    planar = _scenario(parser().parse_args(["record", "--native", "--spatial-dim", "2"]))
    assert planar["spatial_dim"] == 2


def test_planar_native_box_accepts_xyz_or_xy_acceleration():
    session = FlightSession(FlightScenarioSpec(red_attackers=1, blue_attackers=1, spatial_dim=2,
                                              action_mode="continuous_native"))
    try:
        session.step({"red": [[.2, .1, 0]], "blue": [[.1, -.1]]})
        assert session.env.red_agents[0].velocity[2] == 0
    finally:
        session.close()


def test_cli_configuration_file_selects_model_and_action_without_default_overrides(tmp_path):
    from skyhad_workbench.cli import parser, _scenario
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"env_agent_type": "UAV_quadrotor", "env_agent_action_type": "actuator", "spatial_dim": 3}))
    scenario = _scenario(parser().parse_args(["record", "--native", "--config", str(path)]))
    assert scenario["env_agent_type"] == "UAV_quadrotor"
    assert scenario["env_agent_action_type"] == "actuator"
    assert scenario["action_mode"] == "continuous_native"


def test_two_policies_receive_precommit_observations_without_engine_access():
    seen = []
    class Policy:
        def act(self, obs):
            assert {"observation", "agent_ids", "alive_mask"} <= set(obs)
            assert not {"env", "engine", "rng"} & set(obs)
            seen.append(copy.deepcopy(obs))
            obs["observation"].clear()
            return [0] * len(obs["agent_ids"])
    session = FlightSession(FlightScenarioSpec(red_attackers=1, blue_attackers=1), red_policy=Policy(), blue_policy=Policy())
    try:
        before = session.observations()
        session.step()
        assert seen == [before["red"], before["blue"]]
    finally:
        session.close()


def test_native_stochastic_policy_owns_memory_and_all_ambient_random_streams():
    import random
    torch = pytest.importorskip("torch")
    from skyhad_workbench.rng import random_state
    class Stochastic:
        def __init__(self):
            self.calls = 0

        def act(self, observation):
            self.calls += 1
            return [int(torch.randint(0, 27, ()).item() + random.randrange(27) + np.random.randint(27)) % 27
                    for _ in observation["agent_ids"]]
    policy = Stochastic()
    before = random_state()
    session = FlightSession(FlightScenarioSpec(red_attackers=1, blue_attackers=1, max_steps=3), red_policy=policy)
    child = session.branch()
    try:
        assert session.run().frames == child.run().frames
        assert policy.calls == 0
        after = random_state()
        assert before["python"] == after["python"]
        assert np.array_equal(before["numpy"][1], after["numpy"][1])
        assert torch.equal(before["torch"], after["torch"])
    finally:
        session.close()
        child.close()


def test_display_interpolation_never_changes_health_events_or_original_recording():
    a = dict(step=0, entities=[dict(id=0, side="red", alive=True, position=[0, 0, 0], health=1.)], events=[])
    b = dict(step=1, entities=[dict(id=0, side="red", alive=True, position=[10, 0, 0], health=.2)], events=[{"kind": "damage"}])
    episode = ReplayEpisode({}, [a, b], [])
    result = display_frame(episode, .5)
    assert result["entities"][0]["position"] == [5, 0, 0]
    assert result["entities"][0]["health"] == 1. and result["events"] == []
    assert episode.frames[0] == a
    episode.sparse = True
    assert display_frame(episode, .5) == a


def test_uav_playback_slerps_attitude_and_keeps_recorded_rigid_state_immutable():
    q0 = [1., 0., 0., 0.]
    q1 = [0., 0., 0., 1.]
    def frame(step, x, quaternion):
        return dict(step=step, entities=[dict(id=0, side="red", alive=True, position=[x, 0, 0],
                    attitude=quaternion, rigid_state=[x, 0, 0, 2, 0, 0, *quaternion, 0, 0, 1])])
    episode = ReplayEpisode({}, [frame(0, 0, q0), frame(1, 10, q1)], [])
    original = copy.deepcopy(episode.to_dict())
    row = display_frame(episode, .5)["entities"][0]
    np.testing.assert_allclose(row["attitude"], [2**-.5, 0, 0, 2**-.5], atol=1e-15)
    np.testing.assert_allclose(row["rigid_state"][6:10], row["attitude"])
    assert row["rigid_state"][:3] == [5., 0., 0.]
    assert episode.to_dict() == original
    # Opposite quaternion signs represent the same orientation.
    episode.frames[1]["entities"][0]["attitude"] = [-1., 0., 0., 0.]
    np.testing.assert_allclose(display_frame(episode, .5)["entities"][0]["attitude"], q0)


def test_viewer_edits_change_provenance_without_claiming_new_physics(tmp_path):
    source = tmp_path / "skyhad_workbench"
    source.mkdir(parents=True)
    (source / "gui.py").write_text("VIEW = 1")
    (source / "session.py").write_text("STATE = 1")
    first = source_identity(tmp_path)
    (source / "gui.py").write_text("VIEW = 2")
    second = source_identity(tmp_path)
    assert first["behavior_hash"] == second["behavior_hash"]
    assert first["viewer_hash"] != second["viewer_hash"]
    assert first["source_hash"] != second["source_hash"]
    (source / "session.py").write_text("STATE = 2")
    assert second["behavior_hash"] != source_identity(tmp_path)["behavior_hash"]
    before_entry = source_identity(tmp_path)
    (tmp_path / "make_env.py").write_text("FACTORY_PROTOCOL = 1")
    assert before_entry["behavior_hash"] != source_identity(tmp_path)["behavior_hash"]
    render = tmp_path / "had_env" / "render"
    render.mkdir(parents=True)
    (render / "glyphs.py").write_text("GLYPH = 1")
    before_render = source_identity(tmp_path)
    (render / "glyphs.py").write_text("GLYPH = 2")
    assert before_render["behavior_hash"] == source_identity(tmp_path)["behavior_hash"]
    assert before_render["viewer_hash"] != source_identity(tmp_path)["viewer_hash"]
    for module in ("agent_session.py", "cli.py", "live.py"):
        before_behavior = source_identity(tmp_path)
        (source / module).write_text("BEHAVIOR = 1")
        assert before_behavior["behavior_hash"] != source_identity(tmp_path)["behavior_hash"]


def test_render_payload_and_literal_palette_do_not_hide_physical_changes(tmp_path):
    core = tmp_path / "had_env"
    core.mkdir()
    simulation = core / "simulation.py"
    original = '''class Simulation:
    def step(self, action):
        return action + 1

    def _render_information(self):
        return {'label': 'old'}
'''
    simulation.write_text(original)
    config = core / "config.py"
    config.write_text("RedColor = (255, 0, 0)\nScreenLength = 800\nInterval = 1\n")
    first = source_identity(tmp_path)
    simulation.write_text(original.replace("'old'", "'new'"))
    config.write_text("RedColor = (200, 0, 0)\nScreenLength = 900\nInterval = 1\n")
    visual = source_identity(tmp_path)
    assert visual["source_hash"] != first["source_hash"]
    assert visual["behavior_hash"] == first["behavior_hash"]
    simulation.write_text(original + "\n    def _render_metadata(self):\n        return {'step': 1}\n")
    assert source_identity(tmp_path)["behavior_hash"] == visual["behavior_hash"]
    simulation.write_text(original.replace('action + 1', 'action + 2'))
    assert source_identity(tmp_path)["behavior_hash"] != visual["behavior_hash"]
    simulation.write_text(original)
    config.write_text("RedColor = (200, 0, 0)\nScreenLength = 900\nInterval = 2\n")
    assert source_identity(tmp_path)["behavior_hash"] != visual["behavior_hash"]
    # Even a presentation-named assignment stays behavioral if it calls code.
    config.write_text("RedColor = change_physics()\nScreenLength = 900\nInterval = 1\n")
    assert source_identity(tmp_path)["behavior_hash"] != visual["behavior_hash"]
    config.write_text("RedColor = (200, 0, 0)\nScreenLength = 900\nInterval = 1\n")
    for extra in ("    def close(self):\n        self.closed = True\n",
                  "    def reset(self):\n        self.state = 1\n"):
        simulation.write_text(original + extra)
        assert source_identity(tmp_path)["behavior_hash"] != visual["behavior_hash"]
    simulation.write_text(original)
    config.write_text("RedColor = (200, 0, 0)\nScreenLength = 900\nInterval = 1\nMODEL_PRESETS = {'speed': 12}\n")
    assert source_identity(tmp_path)["behavior_hash"] != visual["behavior_hash"]


def test_flight_preserves_direct_native_rewards_observations_and_fixed_roster():
    from had_env.simulation import Simulation
    session = FlightSession(FlightScenarioSpec(red_attackers=2, blue_attackers=2,
                                              red_scouts=1, blue_disturbers=1, max_steps=3))
    try:
        assert type(session.env) is Simulation
        session.env.red_agents[0].Health = 0
        native = copy.deepcopy(session.env)
        native_result = native.step([[0., 0., 0.]] * len(native.agents))
        assert len(native_result) == 6
        transition = session.step()
        assert transition.rewards == native_result[3]
        assert transition.observations['red']['observation'] + transition.observations['blue']['observation'] == native_result[0]
        assert len(transition.observations['red']['agent_ids']) == 3
        assert transition.observations['red']['alive_mask'][0] is False
        assert session.env.red_agents[0].Id in transition.observations['red']['agent_ids']
    finally:
        session.close()


def test_cli_records_loadable_complete_episode_and_exports_figures(tmp_path):
    recording = tmp_path / "episode.json.gz"
    assert main(["record", "--red", "2", "--blue", "2", "--output", str(recording)]) == 0
    episode = load_episode(recording)
    assert episode.metadata["complete"] and len(episode.frames) == episode.max_step + 1
    for suffix in ("png", "svg", "pdf"):
        target = tmp_path / f"figure.{suffix}"
        assert main(["export", str(recording), str(target)]) == 0
        assert target.stat().st_size > 1000
        assert json.loads(Path(str(target)+".json").read_text())["recording_metadata"]["scenario"]


def test_video_decodes_with_requested_dimensions_and_final_frame(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    import imageio.v2 as imageio
    from skyhad_workbench.export import export_video
    session = FlightSession(FlightScenarioSpec(red_attackers=1, blue_attackers=1, max_steps=2))
    try:
        episode = session.run()
    finally:
        session.close()
    # A saved continuation can start at a nonzero physical step; no invented lead-in.
    for frame in episode.frames:
        frame["step"] += 5
        frame["sim_time"] += 5
    target = export_video(episode, tmp_path / "episode.mp4", fps=2, steps_per_second=2, width=640, height=480)
    with imageio.get_reader(target) as reader:
        frames = list(reader.iter_data())
    assert len(frames) == 3 and frames[-1].shape == (480, 640, 3)
    assert not np.array_equal(frames[0], frames[-1])


def test_cli_saved_decision_can_branch_again_without_reconstructing_unknown_history(tmp_path):
    from skyhad_workbench.protocols import ScenarioSpec
    from skyhad_workbench.session import SimulationSession
    original = tmp_path / "original.json.gz"
    with SimulationSession(ScenarioSpec()) as session:
        episode = session.run()
        session.save(original)
    step = episode.decisions[1]["step"]
    first = tmp_path / "first.json.gz"
    second = tmp_path / "second.json.gz"
    main(["branch", str(original), "--step", str(step), "--policy", "grand", "--output", str(first)])
    main(["branch", str(first), "--step", str(step), "--policy", "rule", "--output", str(second)])
    assert load_episode(first).frames[0]["entities"] == episode.frame_at(step)["entities"]
    assert load_episode(second).frames[0]["entities"] == episode.frame_at(step)["entities"]
    assert load_episode(second).metadata["complete"]


def test_cli_explicit_split_evaluation_writes_records_and_refuses_overwrite(tmp_path):
    from skyhad_workbench.protocols import scenario_family
    from skyhad_workbench.evaluation import EvaluationPlan
    plan = EvaluationPlan(
        train=(scenario_family(10, 0, split="train", cells=((2, 2),)),),
        validation=(), test=(scenario_family(11, 0, split="test", cells=((2, 2),)),))
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan.to_dict()))
    destination = tmp_path / "evaluation"
    assert main(["evaluate", str(path), "--output", str(destination)]) == 0
    rows = json.loads((destination / "episodes.json").read_text())
    assert len(rows) == 2 and {r["method_id"] for r in rows} == {"rule", "grand"}
    assert all(load_episode(r["recording"]).metadata["complete"] for r in rows)
    with pytest.raises((ValueError, FileExistsError)):
        main(["evaluate", str(path), "--output", str(destination)])


def _wait_message(controller, wanted, timeout=20):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        rows = controller.poll()
        errors = [r for r in rows if r["kind"] == "error"]
        assert not errors, errors
        selected = [r for r in rows if r["kind"] == wanted]
        if selected:
            return selected[-1]
        assert controller.process.is_alive(), f"Worker exited {controller.process.exitcode}"
        time.sleep(.02)
    raise AssertionError(f"No {wanted} received from owned debug worker")


def test_spawn_debug_worker_pauses_and_advances_one_physical_step_only():
    from skyhad_workbench.live import LiveController
    from skyhad_workbench.protocols import ScenarioSpec
    controller = LiveController(ScenarioSpec(red_count=2, blue_count=2).to_dict())
    try:
        initial = _wait_message(controller, "episode")
        assert initial["episode"]["frames"][0]["step"] == 0
        controller.send("step")
        first = _wait_message(controller, "frame")
        assert first["frame"]["step"] == 1
        time.sleep(.12)
        assert not [r for r in controller.poll() if r["kind"] == "frame"]
        controller.send("step")
        assert _wait_message(controller, "frame")["frame"]["step"] == 2
    finally:
        controller.close()
    assert not controller.process.is_alive()
