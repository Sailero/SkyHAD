"""View-mode routing through every interface, with a fake 3D renderer (no graphics required)."""
import copy
import sys
import types

import numpy as np
import pytest

from had_env import make_env


class Fake3DRenderer:
    def __init__(self, simulation):
        self.simulation = simulation
        self.closed = False
        self.frames = 0
    def render(self):
        self.frames += 1
        return not self.closed
    def render_rgb_array(self):
        self.frames += 1
        return np.full((48, 64, 3), 123, np.uint8)
    def close(self):
        self.closed = True


def physical(env):
    if hasattr(env, 'simulation'):
        return env.simulation
    if hasattr(env, 'parallel_env'):
        return env.parallel_env.simulation
    return env.adapter.env


@pytest.fixture
def fake_3d(monkeypatch):
    monkeypatch.setitem(sys.modules, 'had_env.render.native3d', types.SimpleNamespace(Native3DRenderer=Fake3DRenderer))


@pytest.mark.parametrize('api', ['parallel', 'mpe', 'grouping'])
def test_each_interface_routes_3d_and_switches_without_mutating_state(api, fake_3d):
    options = dict(api=api, env_agent_type='UAV_fixedwing', render_mode='rgb_array')
    env = make_env(**options)
    try:
        env.reset(seed=41)
        sim = physical(env)
        before = [(list(e.position), list(e.velocity), e.Health) for e in sim.entities]
        rng = copy.deepcopy(sim.np_random.bit_generator.state)
        env.set_view_mode('3d')
        image = env.render()
        assert image.shape == (48, 64, 3) and image.dtype == np.uint8
        renderer = sim._scene3d_renderer
        env.render()
        assert sim._scene3d_renderer is renderer
        assert sim._view_mode == '3d'
        assert before == [(list(e.position), list(e.velocity), e.Health) for e in sim.entities]
        assert sim.np_random.bit_generator.state == rng and sim.physics_step_count == 0
        env.set_view_mode('projections')
        assert renderer.closed and sim._scene3d_renderer is None
        assert sim._view_mode == 'projections'
        assert before == [(list(e.position), list(e.velocity), e.Health) for e in sim.entities]
        env.set_view_mode('3d')
        env.render()
        replacement = sim._scene3d_renderer
        env.close()
        assert replacement.closed
    finally:
        env.close()


def test_missing_3d_packages_give_install_guidance(monkeypatch):
    monkeypatch.setenv('QT_QPA_PLATFORM', 'xcb')
    monkeypatch.setenv('DISPLAY', ':0')
    for name in ('PySide6', 'PySide6.QtWidgets'):
        monkeypatch.setitem(sys.modules, name, None)
    env = make_env(env_agent_type='UAV_quadrotor', render_mode='rgb_array')
    try:
        env.reset(seed=3)
        env.set_view_mode('3d')
        with pytest.raises(RuntimeError, match=r'\[viewer3d\]'):
            env.render()
        assert env.metadata['render_modes'] == ['human', 'rgb_array']
    finally:
        env.close()


@pytest.mark.parametrize('mode', ['split', '2d', None, 'software'])
def test_invalid_view_mode_has_no_side_effects(mode):
    env = make_env(env_agent_type='UAV_quadrotor')
    try:
        with pytest.raises(ValueError, match='projections.*3d'):
            env.set_view_mode(mode)
        assert env.simulation._view_mode == 'projections'
    finally:
        env.close()


def test_particle_explicit_3d_request_is_clear():
    env = make_env()
    try:
        with pytest.raises(ValueError, match='UAV'):
            env.set_view_mode('3d')
    finally:
        env.close()


def test_grouping_restored_physical_state_can_render_without_an_extra_reset(fake_3d):
    original = make_env(api='grouping', env_agent_type='UAV_fixedwing')
    restored = make_env(api='grouping', env_agent_type='UAV_fixedwing', render_mode='rgb_array')
    try:
        original.reset(seed=31)
        saved = original.snapshot()
        restored.restore(saved)
        restored.set_view_mode('3d')
        assert restored.render().shape == (48, 64, 3)
    finally:
        original.close()
        restored.close()
