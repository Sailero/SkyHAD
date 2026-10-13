"""Physical Simulation with native reset/step tuples for grouping consumers."""
import numpy as np

from had_env.config import (
    EnvDim,
    HorizonPolicy,
    reward_attack_single,
    reward_boundary,
    reward_disturb_single,
    reward_episode,
    reward_scout_single,
)
from had_env._rendering import has_native_renderer, has_3d_renderer, native_backend
from had_env.world import World
from had_env.initialization import reset_random, reset_uniform
from had_env.observations import observations, alive_entities, global_state
from had_env.tasks import rewards, agent_done, entity_info, survival_outcome, termination_reason, task_info


class Simulation(World):
    """Own one fixed-order entity set and its episode/render lifecycle."""
    def __init__(self, red_attack_n, blue_attack_n, target_n,
                 red_scout_n=0, red_disturb_n=0, blue_scout_n=0, blue_disturb_n=0,
                 task_type='Training', target_region=None, seed=None,
                 red_spawn_annulus=None, blue_spawn_x=None, spawn_altitude=None,
                 fire_range=None, horizon_policy=None, reward_config=None,
                 task_mode='survival', target_health=None, spatial_dim=2, plane_altitude=None,
                 target_initialization='random', target_positions=None, effective_config=None):
        from had_env.config import EnvConfig
        effective_config = effective_config or EnvConfig(spatial_dim=spatial_dim, task_mode=task_mode,
            plane_altitude=plane_altitude, target_region=target_region, red_spawn_annulus=red_spawn_annulus,
            blue_spawn_x=blue_spawn_x, spawn_altitude=spawn_altitude, fire_range=fire_range)
        super().__init__(red_scout_n, red_disturb_n, red_attack_n,
                         blue_scout_n, blue_disturb_n, blue_attack_n,
                         target_n, task_mode=task_mode, target_health=target_health,
                         spatial_dim=spatial_dim, plane_altitude=plane_altitude,
                         effective_config=effective_config)

        self.area = self.world_bounds
        self.target_region = np.asarray(
            self.effective_config.target_region if target_region is None else target_region,
            dtype=np.float64,
        )
        self.red_spawn_annulus = tuple(self.effective_config.red_spawn_annulus if red_spawn_annulus is None else red_spawn_annulus)
        self.blue_spawn_x = tuple(self.effective_config.blue_spawn_x if blue_spawn_x is None else blue_spawn_x)
        self.spawn_altitude = tuple(self.effective_config.spawn_altitude if spawn_altitude is None else spawn_altitude)
        self.fire_range = float(self.effective_config.fire_range if fire_range is None else fire_range)
        if not np.isfinite(self.fire_range) or not 0 < self.fire_range <= self.attack_distance[0]:
            raise ValueError("fire_range must be positive and no larger than the full-damage radius")
        for entity in self.entities:
            entity.fire_range = self.fire_range
        self.horizon_policy = str(HorizonPolicy if horizon_policy is None else horizon_policy)
        if self.horizon_policy not in ("red_win", "draw", "blue_win"):
            raise ValueError("horizon_policy must be red_win, draw, or blue_win")
        self.reward_config = {
            "disturb_single": reward_disturb_single,
            "scout_single": reward_scout_single,
            "attack_single": reward_attack_single,
            "boundary": reward_boundary,
            "episode": reward_episode,
        }
        if reward_config:
            self.reward_config.update(reward_config)
        if self.target_region.shape != (3, 2):
            raise ValueError("target_region must contain [low, high] for x, y, z")
        world_bounds = self.world_bounds
        if np.any(self.target_region[:, 0] > self.target_region[:, 1]):
            raise ValueError("target_region lower bounds must not exceed upper bounds")
        if np.any(self.target_region[:, 0] < world_bounds[:, 0]) or np.any(
            self.target_region[:, 1] > world_bounds[:, 1]
        ):
            raise ValueError("target_region must stay inside AeroPoint")
        if target_initialization not in ('random', 'fixed'):
            raise ValueError("target_initialization must be random or fixed")
        self.target_initialization = 'fixed' if target_positions is not None else target_initialization
        self.fixed_target_positions = None
        if self.target_initialization == 'fixed':
            if target_positions is None:
                positions = np.tile(self.target_region.mean(axis=1), (target_n, 1))
                positions[:, 1] = np.linspace(*self.target_region[1], target_n + 2)[1:-1]
            else:
                positions = np.asarray(target_positions, dtype=np.float64).copy()
            if positions.shape != (target_n, EnvDim) or not np.isfinite(positions).all():
                raise ValueError("target_positions must contain one finite [x, y, z] per target")
            if self.spatial_dim == 2:
                positions[:, 2] = self.plane_altitude
            if np.any(positions < world_bounds[:, 0]) or np.any(positions > world_bounds[:, 1]):
                raise ValueError("target_positions must stay inside AeroPoint")
            self.fixed_target_positions = positions
        self.np_random = np.random.default_rng(seed)

        self._display_player = None
        self._rgb_player = None
        self._scene3d_renderer = None
        self._view_mode = 'projections'
        self._render_closed = False

        self.task_type = task_type

    def reset(self, evaluate=False, seed=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        if self.task_type == 'Normal Showcase':
            reset_uniform(self)
        else:
            reset_random(self, evaluate=evaluate)

        observation_n = self.get_observation(is_relative_observation=True)
        world_alive = self.get_world_alive()
        global_state = self.get_global_state()
        return observation_n, world_alive, global_state

    def step(self, action_n):
        self.step_physics(action_n)

        obs_n = self.get_observation(is_relative_observation=True)
        world_alive_n = self.get_world_alive()
        global_state = self.get_global_state()
        reward_n = self.get_reward()
        done_n = self.get_done()
        info = self.get_info()

        return obs_n, world_alive_n, global_state, reward_n, done_n, info

    def step_physics(self, action_n):
        # Validate the complete batch before collision checks or any mutation.
        try:
            actions = np.asarray(action_n)
        except (TypeError, ValueError) as error:
            raise ValueError("actions must be a real finite batch") from error
        width = 4 if self.env_agent_action_type == 'actuator' else 3
        if actions.shape != (len(self.agents), width) or actions.dtype.kind not in 'biuf' or not np.isfinite(actions).all():
            raise ValueError(f"actions must have shape {(len(self.agents), width)} and finite numeric values")
        if self.env_agent_action_type == 'actuator':
            low = 0. if self.env_agent_type == 'UAV_quadrotor' else -1.
            if np.any(actions < low) or np.any(actions > 1.):
                raise ValueError("actuator commands are outside normalized bounds")
            flying = actions.astype(float).tolist()
        elif self.env_agent_action_type == 'position':
            if np.any(actions < self.world_bounds[:, 0]) or np.any(actions > self.world_bounds[:, 1]):
                raise ValueError("position targets must remain within world bounds")
            if self.spatial_dim == 2 and np.any(actions[:, 2] != self.plane_altitude):
                raise ValueError("planar position commands must use plane_altitude")
            if self.dynamics is not None:
                flying = actions.astype(float).tolist()
            else:
                flying = []
                for agent, target in zip(self.agents, actions):
                    displacement = target - np.asarray(agent.position)
                    desired = displacement * min(agent.vMax/max(np.linalg.norm(displacement), 1e-12), .5)
                    accel = 2.*(desired - np.asarray(agent.velocity))
                    accel *= min(agent.aMax/max(np.linalg.norm(accel), 1e-12), 1.)
                    flying.append(accel.tolist())
        else:
            if self.spatial_dim == 2:
                actions = actions.copy()
                actions[:, 2] = 0
            flying = [(np.array(actions[i]) * self.agents[i].aMax).tolist() for i in range(len(actions))]
            if not np.isfinite(flying).all():
                raise ValueError("scaled accelerations must be finite")
        super().step(flying)

    @property
    def control_space(self):
        from gymnasium import spaces
        if self.env_agent_action_type == 'position':
            low = self.world_bounds[:, 0].astype(np.float32)
            high = self.world_bounds[:, 1].astype(np.float32)
            if self.spatial_dim == 2:
                low[2] = high[2] = self.plane_altitude
            return spaces.Box(low, high, dtype=np.float32)
        if self.env_agent_action_type == 'actuator':
            return spaces.Box(0. if self.env_agent_type == 'UAV_quadrotor' else -1., 1., (4,), np.float32)
        return spaces.Box(-1., 1., (self.spatial_dim,), np.float32)

    def get_observation(self, normalization=True, is_relative_observation=True):
        return observations(self, normalization, is_relative_observation)

    def get_world_alive(self):
        return alive_entities(self)

    def get_global_state(self, normalization=True):
        return global_state(self, normalization)

    def get_reward(self):
        return rewards(self)

    def get_done(self):
        return agent_done(self)

    def get_info(self):
        return entity_info(self)

    def is_terminal(self):
        return survival_outcome(self)

    def is_episode_done(self):
        return termination_reason(self) is not None

    def task_info(self):
        return task_info(self)

    def _render_information(self):
        return native_backend().render_information(self)

    def _render_metadata(self):
        return native_backend().render_metadata(self)

    def render_rgb_array(self, include_result=True):
        """Return the selected view without changing physical state."""
        return native_backend().render_rgb_array(self, include_result)

    def set_view_mode(self, mode):
        """Select basic 2D projections or the full edition's UAV 3D scene."""
        if mode not in ('projections', '3d'):
            raise ValueError("view mode must be 'projections' or '3d'")
        if mode == '3d':
            if self.env_agent_type not in ('UAV_fixedwing', 'UAV_quadrotor'):
                raise ValueError('3D rendering is supported for UAV_fixedwing and UAV_quadrotor.')
            if not has_3d_renderer():
                raise ValueError('3D rendering requires the full edition on the main branch; core retains basic 2D rendering.')
        if mode != self._view_mode:
            self.close()
            self._view_mode = mode
            self._render_closed = False

    def render(self):
        """Return False when rendering is unavailable or the window is closed."""
        if self._render_closed or not has_native_renderer():
            return False
        return native_backend().render(self)

    def close(self):
        """Release this environment's render resources without ending Python."""
        if self._display_player is not None:
            self._display_player.close_window()
        self._display_player = None
        self._rgb_player = None
        if self._scene3d_renderer is not None:
            self._scene3d_renderer.close()
            self._scene3d_renderer = None
        self._render_closed = True
