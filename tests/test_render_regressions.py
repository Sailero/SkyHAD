"""Geometric/status regressions that cannot be proved by nonempty images."""
import math

import numpy as np
import pytest

pygame = pytest.importorskip('pygame')
from had_env.render.render import DisplayPlayer
from had_env.render.glyphs import aircraft_geometry


def player_and_surface():
    pygame.font.init()
    surface = pygame.Surface((800, 1000), pygame.SRCALPHA)
    return DisplayPlayer(surface), surface


def test_target_sprite_is_centered_on_world_coordinate():
    player, surface = player_and_surface()
    player.target_image = pygame.Surface((9, 9), pygame.SRCALPHA)
    player.target_image.fill((255, 255, 255))
    player.draw_targets(surface, [dict(position=[.5, .5, .5], alive=True)])
    points = np.argwhere(np.all(pygame.surfarray.array3d(surface) == 255, axis=2))
    np.testing.assert_allclose(points.mean(axis=0), player.project_position([.5, .5, .5]), atol=1)


def test_xz_target_destruction_changes_appearance():
    player, surface = player_and_surface()
    frames = []
    for alive in (True, False):
        surface.fill((0, 0, 0, 0))
        player.draw_targets(surface, [dict(position=[.5, .5, .5], alive=alive)])
        _, y = player.project_position([.5, .5, .5], 'xz')
        frames.append(pygame.surfarray.array3d(surface)[:, int(y)-10:int(y)+11].copy())
    assert not np.array_equal(*frames)


@pytest.mark.parametrize('roll', [math.pi, math.pi / 2])
def test_directed_bank_cues_distinguish_inversion_and_sign(roll):
    player, surface = player_and_surface()
    frames = []
    for angle in (0 if roll == math.pi else -roll, roll):
        surface.fill((0, 0, 0, 0))
        player.draw_agents(surface, [dict(position=[.5, .5, .5], velocity=[1, 0, 0],
            type='Attack', alive=True, env_agent_type='UAV_fixedwing',
            attitude=[math.cos(angle/2), math.sin(angle/2), 0, 0])], 'Red')
        frames.append(pygame.surfarray.array3d(surface).copy())
    assert not np.array_equal(*frames)


def test_panels_use_equal_metric_scale_with_custom_bounds():
    player, _ = player_and_surface()
    bounds = [[-1000, 1000], [-250, 250], [0, 1500]]
    player.configure_bounds(bounds)
    origin = np.array([.5, .5, .5])
    for projection, axis in [('xy', 1), ('xz', 2)]:
        start = np.asarray(player.project_position(origin, projection))
        xstep, ystep = origin.copy(), origin.copy()
        xstep[0] += 1 / 2000
        ystep[axis] += 1 / (bounds[axis][1] - bounds[axis][0])
        dx = np.asarray(player.project_position(xstep, projection)) - start
        dy = np.asarray(player.project_position(ystep, projection)) - start
        assert abs(dx[0]) == pytest.approx(abs(dy[1]))
        assert dy[1] < 0


def test_boundary_glyphs_do_not_bleed_into_headers_or_other_panels():
    player, surface = player_and_surface()
    row = dict(position=[.5, 1, 0], velocity=[1, 0, 0], type='Attack', id=3,
               alive=True, env_agent_type='UAV_fixedwing', attitude=[1, 0, 0, 0])
    player.draw_agents(surface, [row], 'Red')
    mask = pygame.surfarray.array_alpha(surface) > 0
    allowed = np.zeros(mask.shape, dtype=bool)
    for rect in player.panel_rects.values():
        allowed[rect.left:rect.right, rect.top:rect.bottom] = True
    assert not np.any(mask & ~allowed)


@pytest.mark.filterwarnings('ignore:invalid value encountered in scalar divide:RuntimeWarning:had_env.observations')
def test_degenerate_axis_has_safe_display_normalization():
    from had_env.config import EnvConfig
    from had_env.simulation import Simulation
    config = EnvConfig(env_agent_type='UAV_fixedwing', world_bounds=[[-1000,1000],[-1000,1000],[40,40]],
        target_region=[[-900,-800],[-300,300],[40,40]], spawn_altitude=[40,40], plane_altitude=40)
    sim = Simulation(1, 1, 1, effective_config=config, spatial_dim=3)
    try:
        sim.reset(seed=4)
        assert all(np.isfinite(row['position']).all() for group in sim._render_information() for row in group)
        assert sim.render_rgb_array().shape == (1000, 800, 3)
    finally:
        sim.close()


@pytest.mark.parametrize('role', ['Scout', 'Disturb'])
def test_particle_side_projection_keeps_horizontal_velocity_direction(role):
    player, surface = player_and_surface()
    images = []
    rect = player.panel_rects['xz']
    for vx in (-1, 1):
        surface.fill((0,0,0,0))
        player.draw_agents(surface, [dict(position=[.5,.5,.5], velocity=[vx,0,0],
            type=role, alive=True, env_agent_type='particle')], 'Red')
        images.append(pygame.surfarray.array3d(surface)[:,rect.top:rect.bottom].copy())
    assert not np.array_equal(*images)


def test_projected_uav_nose_agrees_with_metric_displacement():
    player, _ = player_and_surface()
    bounds = [[-1000,1000],[-250,250],[0,1500]]
    player.configure_bounds(bounds)
    q = [math.cos(math.pi/8), 0, -math.sin(math.pi/8), 0]
    row = dict(env_agent_type='UAV_fixedwing', attitude=q)
    center = player.project_position([.5,.5,.5], 'xz')
    nose = np.asarray(aircraft_geometry(row, 'xz', center=center)['nose']) - center
    projected = np.asarray(player.project_position([.5+1/2000,.5,.5+1/1500], 'xz')) - center
    np.testing.assert_allclose(nose / np.linalg.norm(nose), projected / np.linalg.norm(projected), atol=1e-12)


def test_collapsed_altitude_keeps_visible_entities_in_a_plane_panel():
    player, surface = player_and_surface()
    player.configure_bounds([[-1000,1000],[-1000,1000],[40,40]])
    player.draw_agents(surface, [dict(position=[.5,.5,.5], velocity=[1,0,0], type='Attack',
        env_agent_type='UAV_fixedwing', attitude=[1,0,0,0], alive=True)], 'Red')
    rect = player.panel_rects['xz']
    assert np.any(pygame.surfarray.array_alpha(surface)[:,rect.top:rect.bottom])
