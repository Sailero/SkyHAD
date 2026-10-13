"""The generated edition retains minimal 2D and excludes full-only features."""
import importlib.util
from pathlib import Path

import pytest

import had_env


def test_core_retains_minimal_2d_without_full_features_or_workbench():
    assert importlib.util.find_spec('had_env.render') is not None
    assert importlib.util.find_spec('skyhad_workbench') is None
    for name in ('had_env.render.scene3d', 'had_env.render.native3d', 'had_env.render.video'):
        assert importlib.util.find_spec(name) is None
    package = Path(had_env.__file__).parent
    assert (package / 'resources/target.png').is_file()
    env = had_env.make_env(red_count=1, blue_count=1, max_cycles=1)
    try:
        assert env.metadata['render_modes'] == ['human', 'rgb_array']
        env.reset(seed=17)
        assert len(env.step({a: 0 for a in env.agents})) == 5
    finally:
        env.close()


@pytest.mark.parametrize('mode', ['human', 'rgb_array'])
def test_core_accepts_2d_render_configuration_without_importing_pygame(mode):
    env = had_env.make_env(render_mode=mode)
    env.close()


def test_core_rejects_full_only_3d_with_actionable_guidance():
    env = had_env.make_env(env_agent_type='UAV_fixedwing')
    try:
        env.set_view_mode('projections')
        with pytest.raises(ValueError, match='[Ff]ull'):
            env.set_view_mode('3d')
    finally:
        env.close()
