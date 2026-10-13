"""Native Pygame adapter; all state-to-picture work stays optional."""
import numpy as np
from had_env.config import Interval
from .style import ScreenLength, ScreenWidth, ScreenHeight, SurfaceColor


def render_information(simulation):
    bounds = np.asarray(simulation.world_bounds, float)
    span = bounds[:, 1] - bounds[:, 0]
    def row(entity, aircraft=False):
        normalized = np.divide(np.asarray(entity.position)-bounds[:, 0], span,
                               out=np.full(3, .5), where=span > 0)
        result = dict(position=np.clip(normalized, 0., 1.).tolist(), id=entity.Id,
                      health=float(entity.Health), max_health=float(entity.initial_health),
                      alive=bool(entity.Health > 0))
        if aircraft:
            attitude = getattr(entity, 'attitude', None)
            result.update(velocity=list(entity.velocity), type=entity.Type,
                          env_agent_type=simulation.env_agent_type,
                          attitude=None if attitude is None else np.asarray(attitude).tolist())
        return result
    return ([row(t) for t in simulation.targets],
            [row(a, True) for a in simulation.red_agents],
            [row(a, True) for a in simulation.blue_agents])


def render_metadata(simulation):
    return dict(step=simulation.physics_step_count, sim_time=simulation.physics_step_count * Interval,
        task_mode=simulation.task_mode, env_agent_type=simulation.env_agent_type,
        world_bounds=simulation.world_bounds.tolist(),
        live_counts={'red': sum(a.Health > 0 for a in simulation.red_agents),
                     'blue': sum(a.Health > 0 for a in simulation.blue_agents)},
        targets=[dict(id=t.Id, health=float(t.Health), max_health=float(t.initial_health))
                 for t in simulation.targets],
        step_target_damage=simulation.step_target_damage, target_damage=simulation.target_damage)


def render_rgb_array(simulation, include_result=True):
    import pygame
    from .render import DisplayPlayer
    if not pygame.font.get_init():
        pygame.font.init()
    if simulation._rgb_player is None:
        simulation._rgb_player = DisplayPlayer(pygame.Surface((ScreenLength, ScreenWidth + ScreenHeight)))
    player = simulation._rgb_player
    player.screen.fill(SurfaceColor)
    player.draw(render_information(simulation), simulation.is_terminal() if include_result else 0,
                render_metadata(simulation))
    return np.transpose(pygame.surfarray.array3d(player.screen), (1, 0, 2)).copy()


def render(simulation):
    import pygame
    from .render import DisplayPlayer
    if simulation._display_player is None or not pygame.display.get_init():
        pygame.display.init()
        if not pygame.font.get_init():
            pygame.font.init()
        simulation._display_player = DisplayPlayer(pygame.display.set_mode((ScreenLength, ScreenWidth + ScreenHeight)))
    player = simulation._display_player
    player.screen.fill(SurfaceColor)
    player.update(render_information(simulation), simulation.is_terminal(), render_metadata(simulation))
    if not player.running:
        simulation.close()
        return False
    return True
