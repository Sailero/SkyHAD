"""Headless rendering lifecycle and physics isolation checks."""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import pytest
pygame = pytest.importorskip("pygame")

from had_env.simulation import Simulation
from make_env import make_env
from had_env.config import ScreenLength, ScreenWidth, ScreenHeight


@pytest.fixture
def env():
    instance = Simulation(2, 1, 1)
    instance.reset(seed=4, evaluate=True)
    yield instance
    instance.close()
    pygame.quit()


def test_rgb_is_cached_stable_and_requires_no_window(env, monkeypatch):
    loads = []
    load = pygame.image.load
    def counted(path):
        loads.append(path)
        return load(path)
    monkeypatch.setattr(pygame.image, "load", counted)
    first = env.render_rgb_array()
    player = env._rgb_player
    second = env.render_rgb_array()
    assert env._rgb_player is player
    assert len(loads) == 2
    assert first.shape == (int(ScreenWidth + ScreenHeight), int(ScreenLength), 3)
    assert first.dtype == np.uint8
    np.testing.assert_array_equal(first, second)
    assert not np.shares_memory(first, second)
    assert pygame.display.get_surface() is None


def test_window_reuses_resources_and_terminal_frame_never_busy_waits(env, monkeypatch):
    calls = []
    set_mode = pygame.display.set_mode
    def counted(*args, **kwargs):
        calls.append(args)
        return set_mode(*args, **kwargs)
    monkeypatch.setattr(pygame.display, "set_mode", counted)
    def forbidden_timer():
        raise AssertionError("Rendering must not busy-wait on result timers")
    monkeypatch.setattr(pygame.time, "get_ticks", forbidden_timer)
    env.blue_agents[0].Health = 0
    assert env.render()
    player = env._display_player
    assert env.render()
    assert env._display_player is player and len(calls) == 1
    assert env.physics_step_count == 0


@pytest.mark.parametrize("event", [pygame.event.Event(pygame.QUIT),
                                  pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE)])
def test_window_quit_returns_control_without_reopening_or_ending_python(env, event):
    assert env.render()
    pygame.event.post(event)
    assert env.render() is False
    assert pygame.display.get_surface() is None
    assert env.render() is False
    env.close()
    env.close()
    # Offscreen export remains available even after interactive close.
    assert env.render_rgb_array().shape[2] == 3
    env.reset(seed=4)
    assert env.render()


def test_render_does_not_mutate_physics_or_random_generators(env):
    before = [(list(a.position), list(a.velocity), a.Health) for a in env.entities]
    rng_before = repr(env.np_random.bit_generator.state)
    for _ in range(3):
        env.render_rgb_array()
        env.render()
    after = [(list(a.position), list(a.velocity), a.Health) for a in env.entities]
    assert before == after
    assert rng_before == repr(env.np_random.bit_generator.state)
    assert env.physics_step_count == 0


def test_uav_glyphs_distinguish_models_and_orient_from_attitude():
    from had_env.render.render import DisplayPlayer
    pygame.init()
    surface = pygame.Surface((800, 1000), pygame.SRCALPHA)
    player = DisplayPlayer(surface)
    row = dict(position=[.5, .5, .5], velocity=[100, 0, 0], type="Attack", alive=True,
               env_agent_type="UAV_fixedwing", attitude=[1., 0., 0., 0.])
    def pixels(value):
        surface.fill((0, 0, 0, 0))
        player.draw_agents(surface, [value], "Red")
        return pygame.surfarray.array3d(surface).copy()
    forward = pixels(row)
    turned = pixels({**row, "attitude": [2**-.5, 0., 0., 2**-.5]})
    quad = pixels({**row, "env_agent_type": "UAV_quadrotor"})
    assert not np.array_equal(forward, turned)
    assert not np.array_equal(forward, quad)
    assert row["attitude"] == [1., 0., 0., 0.]
    pygame.quit()


def test_render_and_reset_lifecycle(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    env = make_env(red_count=1, blue_count=1, render_mode="rgb_array")
    with pytest.raises(RuntimeError):
        env.step({})
    env.reset(seed=1)
    before = env.state()
    frame = env.render()
    assert frame.dtype == np.uint8 and frame.ndim == 3 and frame.shape[-1] == 3
    np.testing.assert_array_equal(env.state(), before)
    env.close()
    with pytest.raises(RuntimeError):
        env.step({})
    env.reset(seed=1)
    assert env.render().shape == frame.shape
    env.close()


def _agent_pixels(row):
    from had_env.render.render import DisplayPlayer
    pygame.font.init()
    surface = pygame.Surface((800, 1000), pygame.SRCALPHA)
    player = DisplayPlayer(surface)
    player.draw_agents(surface, [row], "Red")
    return pygame.surfarray.array3d(surface).copy()


@pytest.mark.parametrize("model", ["particle", "UAV_fixedwing", "UAV_quadrotor"])
def test_native_altitude_increases_upward(model):
    row = dict(position=[.5, .5, .2], velocity=[1, 0, 0], type="Attack", alive=True,
               env_agent_type=model, attitude=[1., 0., 0., 0.])
    low = _agent_pixels(row)[:, 800:]
    high = _agent_pixels({**row, "position": [.5, .5, .8]})[:, 800:]
    assert np.nonzero(high.any(axis=2))[1].mean() < np.nonzero(low.any(axis=2))[1].mean()


def test_native_target_altitude_increases_upward():
    from had_env.render.render import DisplayPlayer
    pygame.font.init()
    surface = pygame.Surface((800, 1000), pygame.SRCALPHA)
    player = DisplayPlayer(surface)
    heights = []
    for altitude in (.2, .8):
        surface.fill((0, 0, 0, 0))
        player.draw_targets(surface, [dict(position=[.5, .5, altitude], alive=True)])
        pixels = pygame.surfarray.array3d(surface)[:, 800:]
        heights.append(np.nonzero(pixels.any(axis=2))[1].mean())
    assert heights[1] < heights[0]


def test_quad_nose_makes_quarter_turn_visible_and_follows_attitude():
    row = dict(position=[.5, .5, .5], velocity=[-100, 0, 0], type="Attack", alive=True,
               env_agent_type="UAV_quadrotor", attitude=[1., 0., 0., 0.])
    first = _agent_pixels(row)[:, :800]
    turned = _agent_pixels({**row, "attitude": [2**-.5, 0, 0, 2**-.5]})[:, :800]
    assert not np.array_equal(first, turned)
    from had_env.render.glyphs import aircraft_geometry
    nose = aircraft_geometry(row)["nose"]
    assert nose[0] > 0 and abs(nose[1]) < 1e-10
    nose = aircraft_geometry({**row, "attitude": [2**-.5, 0, 0, 2**-.5]})["nose"]
    assert abs(nose[0]) < 1e-10 and nose[1] < 0


@pytest.mark.parametrize("model", ["UAV_fixedwing", "UAV_quadrotor"])
def test_uav_roles_remain_visible_without_labels(model):
    row = dict(position=[.5, .5, .5], velocity=[1, 0, 0], alive=True,
               env_agent_type=model, attitude=[1., 0., 0., 0.])
    images = [_agent_pixels({**row, "type": role}) for role in ("Attack", "Disturb", "Scout")]
    assert all(not np.array_equal(a, b) for a, b in ((images[0], images[1]), (images[1], images[2])))


def test_bank_changes_cue_without_changing_forward_heading():
    row = dict(position=[.5, .5, .5], velocity=[1, 0, 0], type="Scout", alive=True,
               env_agent_type="UAV_fixedwing", attitude=[1., 0., 0., 0.])
    banked = {**row, "attitude": [2**-.5, 2**-.5, 0, 0]}
    assert not np.array_equal(_agent_pixels(row), _agent_pixels(banked))
    from had_env.render.glyphs import attitude_cues, aircraft_geometry
    assert attitude_cues(banked)["bank"] == pytest.approx(np.pi / 2)
    nose_up = {**row, "attitude": [2**-.5, 0, -2**-.5, 0]}
    assert attitude_cues(nose_up)["nose_up_pitch"] == pytest.approx(np.pi / 2)
    assert aircraft_geometry(nose_up, "xz")["nose"][1] < 0
    assert aircraft_geometry(nose_up, "xy")["head_on"]
    assert attitude_cues({"velocity": [1, 0, 1]})["bank"] is None
    assert attitude_cues({"attitude": [0, 0, 0, 0]})["nose_up_pitch"] is None


def test_native_hud_uses_effective_bounds_and_raw_damage():
    from had_env.config import EnvConfig
    config = EnvConfig(env_agent_type="UAV_quadrotor", task_mode="damage")
    instance = Simulation(1, 1, 1, effective_config=config, spatial_dim=3, task_mode="damage")
    try:
        instance.reset(seed=4)
        target = instance.targets[0]
        target.step_damage, target.cumulative_damage = .25, 3.5
        instance.blue_agents[0].Health = 0
        instance.physics_step_count = 7
        metadata = instance._render_metadata()
        assert metadata["step"] == 7
        assert metadata["world_bounds"] == instance.world_bounds.tolist()
        assert metadata["live_counts"] == {"red": 1, "blue": 0}
        assert metadata["target_damage"] == 3.5
        assert metadata["step_target_damage"] == .25
        assert metadata["targets"][0]["health"] == target.initial_health
        assert len(instance._render_information()) == 3
        with_hud = instance.render_rgb_array()
        from had_env.render.render import DisplayPlayer
        surface = pygame.Surface((800, 1000))
        surface.fill(__import__("had_env.config", fromlist=["SurfaceColor"]).SurfaceColor)
        DisplayPlayer(surface).draw(instance._render_information(), 0)
        assert not np.array_equal(with_hud, np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2)))
    finally:
        instance.close()


@pytest.mark.parametrize("model", ["UAV_fixedwing", "UAV_quadrotor"])
def test_render_preserves_full_uav_state_rng_and_next_transition(model):
    import copy
    from had_env.config import EnvConfig
    config = EnvConfig(env_agent_type=model, task_mode="damage")
    rendered = Simulation(1, 1, 1, effective_config=config, spatial_dim=3, task_mode="damage")
    control = Simulation(1, 1, 1, effective_config=config, spatial_dim=3, task_mode="damage")
    def state(instance):
        return copy.deepcopy([(a.position, a.velocity, getattr(a, "rigid_state", None),
                               a.Health, getattr(a, "cumulative_damage", None)) for a in instance.entities])
    try:
        rendered.reset(seed=45)
        control.reset(seed=45)
        before = repr(state(rendered))
        rng = repr(rendered.np_random.bit_generator.state)
        observation = rendered.get_global_state().copy()
        rendered.render_rgb_array()
        assert rendered.render()
        assert repr(state(rendered)) == before
        assert repr(rendered.np_random.bit_generator.state) == rng
        np.testing.assert_array_equal(rendered.get_global_state(), observation)
        assert rendered.physics_step_count == 0
        action = np.zeros((2, 3))
        rendered.step(action)
        control.step(action)
        assert repr(state(rendered)) == repr(state(control))
        np.testing.assert_array_equal(rendered.get_global_state(), control.get_global_state())
    finally:
        rendered.close()
        control.close()


def test_team_palette_has_light_background_contrast_and_grayscale_separation():
    from had_env.config import RedColor, BlueColor, SurfaceColor
    def luminance(color):
        rgb = np.asarray(color[:3]) / 255
        linear = np.where(rgb <= .04045, rgb/12.92, ((rgb+.055)/1.055)**2.4)
        return float(linear @ [0.2126, 0.7152, 0.0722])
    red, blue, background = map(luminance, (RedColor, BlueColor, SurfaceColor))
    assert abs(red-blue) >= .07
    assert min((background+.05)/(red+.05), (background+.05)/(blue+.05)) >= 3
