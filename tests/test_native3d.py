"""Read-only simulation adapter and genuine native OpenGL rendering."""
import copy
import importlib
import os
import random
import subprocess
import sys

import numpy as np
import pytest

from had_env.config import EnvConfig
from had_env.simulation import Simulation


def adapter():
    assert importlib.util.find_spec('had_env.render.native3d') is not None, 'Native 3D adapter missing'
    return importlib.import_module('had_env.render.native3d')


def simulation(model='UAV_fixedwing'):
    config = EnvConfig(env_agent_type=model, spatial_dim=3)
    value = Simulation(2, 1, 1, effective_config=config)
    value.reset(seed=7)
    return value


def test_native_adapter_import_does_not_load_graphics_packages():
    adapter()
    result = subprocess.run([sys.executable, '-c',
        "import sys; import had_env.render.native3d; assert not any(k.startswith(('vispy', 'PySide6')) for k in sys.modules)"],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('platform', ['offscreen', 'minimal'])
def test_unsupported_qt_platform_fails_before_loading_graphics(platform):
    result = subprocess.run([sys.executable, '-c',
        "import os, sys; from had_env.render.native3d import Native3DRenderer\n"
        f"os.environ['QT_QPA_PLATFORM'] = {platform!r}\n"
        "try:\n Native3DRenderer(object())\n"
        "except RuntimeError as exc:\n assert 'OpenGL' in str(exc) and '2D' in str(exc)\n"
        "else:\n raise AssertionError('Unsupported Qt platform accepted')\n"
        "assert not any(k.startswith(('PySide6', 'vispy')) for k in sys.modules)"],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_linux_without_display_fails_before_loading_graphics():
    result = subprocess.run([sys.executable, '-c',
        "import os, sys; from had_env.render.native3d import Native3DRenderer\n"
        "sys.platform = 'linux'\n"
        "for key in ('DISPLAY', 'WAYLAND_DISPLAY', 'QT_QPA_PLATFORM'):\n os.environ.pop(key, None)\n"
        "try:\n Native3DRenderer(object())\n"
        "except RuntimeError as exc:\n assert 'display' in str(exc).lower() and '2D' in str(exc)\n"
        "else:\n raise AssertionError('Headless Linux accepted')\n"
        "assert not any(k.startswith(('PySide6', 'vispy')) for k in sys.modules)"],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_simulation_frame_uses_raw_truth_global_ids_and_damage_without_mutation():
    sim = simulation()
    try:
        sim.blue_agents[0].Health = 0
        sim.targets[0].position = [-130, 240, 450]
        sim.targets[0].step_damage = .25
        sim.targets[0].cumulative_damage = 1.5
        before = [(list(e.position), list(e.velocity), e.Health,
                   copy.deepcopy(getattr(e, 'attitude', None))) for e in sim.entities]
        rng = copy.deepcopy(sim.np_random.bit_generator.state)
        frame = adapter().simulation_frame(sim)
        assert [e['id'] for e in frame['entities']] == [0, 1, 2, 3]
        assert [e['side'] for e in frame['entities']] == ['red', 'red', 'blue', 'targets']
        assert frame['entities'][2]['alive'] is False
        assert frame['entities'][3]['position'] == [-130, 240, 450]
        assert frame['entities'][0]['attitude'] == sim.red_agents[0].attitude
        assert frame['entities'][0]['rigid_state'][6:10] == sim.red_agents[0].attitude
        assert frame['step'] == 0 and frame['sim_time'] == 0
        assert frame['metadata']['target_damage'] == 1.5
        assert frame['metadata']['step_target_damage'] == .25
        assert frame['metadata']['world_bounds'] == sim.world_bounds.tolist()
        assert rng == sim.np_random.bit_generator.state
        assert before == [(list(e.position), list(e.velocity), e.Health,
                          copy.deepcopy(getattr(e, 'attitude', None))) for e in sim.entities]
        frame['entities'][0]['position'][0] = 99999
        assert sim.red_agents[0].position[0] != 99999
    finally:
        sim.close()


def test_particle_frame_keeps_unknown_orientation_and_target_actual_height():
    sim = simulation('particle')
    try:
        rows = adapter().simulation_frame(sim)['entities']
        assert rows[0]['attitude'] is None
        assert rows[0]['rigid_state'] is None
        assert rows[-1]['position'][2] == sim.targets[0].position[2]
    finally:
        sim.close()


@pytest.mark.parametrize('model', ['UAV_fixedwing', 'UAV_quadrotor'])
def test_native_renderer_human_rgb_and_close_are_read_only(model):
    pytest.importorskip('PySide6')
    pytest.importorskip('vispy')
    sim = simulation(model)
    before = sim.get_global_state(normalization=False).copy()
    rng = copy.deepcopy(sim.np_random.bit_generator.state)
    numpy_rng = np.random.get_state()
    python_rng = random.getstate()
    try:
        try:
            renderer = adapter().Native3DRenderer(sim)
        except RuntimeError as exc:
            if os.environ.get('SKYHAD_REQUIRE_3D'):
                raise
            if any(word in str(exc).lower() for word in ('context', 'opengl', 'framebuffer')):
                pytest.skip(str(exc))
            raise
        try:
            first = renderer.render_rgb_array()
            second = renderer.render_rgb_array()
            assert first.dtype == np.uint8 and first.shape == (700, 1000, 3)
            assert np.unique(first.reshape(-1, 3), axis=0).shape[0] > 30
            assert not np.shares_memory(first, second)
            assert renderer.render() is True
            assert renderer.view.layers['labels'] is True
            assert not renderer.view.layers['groups']
            np.testing.assert_array_equal(before, sim.get_global_state(normalization=False))
            assert rng == sim.np_random.bit_generator.state
            after_rng = np.random.get_state()
            assert numpy_rng[0] == after_rng[0] and numpy_rng[2:] == after_rng[2:]
            np.testing.assert_array_equal(numpy_rng[1], after_rng[1])
            assert python_rng == random.getstate()
            assert sim.physics_step_count == 0
            renderer.view.close()
            assert renderer.render() is False
            renderer.close()
            renderer.close()
        finally:
            renderer.close()
    finally:
        sim.close()
