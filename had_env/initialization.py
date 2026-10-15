"""Episode placement using only the environment's own random generator."""
import numpy as np
from had_env.config import AttackRatio, DisturbRatio, EnvDim, PlanarSpawnSeparation, ScoutRatio


def reset_random(simulation):
    """Random assets, Red around a random asset, Blue on the far side of the scene."""
    reset_targets(simulation)
    for agent in simulation.agents:
        agent.reset(*random_pose(simulation, agent.Color))
    separate_spawns(simulation)
    simulation.reset_episode_state()


def reset_uniform(simulation):
    """Evenly spaced lines per role, in the same order used to create the entity slots."""
    reset_targets(simulation)
    agent_id = 0
    for color, counts in (
        ('Red', (simulation.red_scout_n, simulation.red_disturb_n, simulation.red_attack_n)),
        ('Blue', (simulation.blue_scout_n, simulation.blue_disturb_n, simulation.blue_attack_n)),
    ):
        for count, ratio in zip(counts, (ScoutRatio, DisturbRatio, AttackRatio)):
            for type_agent_id in range(count):
                position, velocity = uniform_pose(simulation, type_agent_id, count, ratio, color)
                simulation.agents[agent_id].reset(position, velocity)
                agent_id += 1
    separate_spawns(simulation)
    simulation.reset_episode_state()


def _spawn_spacing(simulation):
    return max(float(PlanarSpawnSeparation)*simulation.scene_scale, simulation.collision_distance + 1e-6)


def reset_targets(simulation):
    if simulation.target_initialization == 'fixed':
        for target, position in zip(simulation.targets, simulation.fixed_target_positions):
            target.reset(position.tolist(), [0.0] * EnvDim)
        return
    placed = []
    spacing = _spawn_spacing(simulation)
    for target in simulation.targets:
        for _ in range(4096):
            point = simulation.np_random.uniform(simulation.target_region[:simulation.spatial_dim, 0],
                                                 simulation.target_region[:simulation.spatial_dim, 1]).tolist()
            if simulation.spatial_dim == 2:
                point.append(simulation.plane_altitude)
            if simulation.spatial_dim == 3 or all(np.linalg.norm(np.asarray(point) - previous) > spacing for previous in placed):
                break
        else:
            raise ValueError("Target region cannot provide the required planar spawn separation")
        target.reset(point, [0.0] * EnvDim)
        placed.append(np.asarray(point))


def separate_spawns(simulation):
    """Resample aircraft that open closer than the spawn spacing to an already placed entity.

    This prevents collisions at the first step; it does not prevent later collisions.
    """
    spacing = _spawn_spacing(simulation)
    placed = [np.asarray(target.position) for target in simulation.targets]
    for agent in simulation.agents:
        point, velocity = agent.position, agent.velocity
        for _ in range(4096):
            if all(np.linalg.norm(np.asarray(point) - previous) > spacing for previous in placed):
                break
            point, velocity = random_pose(simulation, agent.Color)
        else:
            raise ValueError("Spawn regions cannot provide the required separation")
        if point is not agent.position:
            agent.reset(point, velocity)
        placed.append(np.asarray(agent.position))


def uniform_pose(simulation, type_id, type_agent_n, ratio, color):
    length_interval = (simulation.area[1][1] - simulation.area[1][0]) / (type_agent_n + 1)
    agent_y = length_interval * (type_id + 1) + simulation.area[1][0]
    width = simulation.area[0][1] - simulation.area[0][0]
    speed = simulation.effective_config.preset.min_speed
    if color == 'Blue':
        agent_x, initial_velocity = simulation.area[0][1] - width * ratio, [-speed, 0.0, 0.0]
    else:
        agent_x, initial_velocity = simulation.area[0][0] + width * ratio, [speed, 0.0, 0.0]
    agent_z = (simulation.plane_altitude if simulation.spatial_dim == 2 else
               float(simulation.np_random.uniform(simulation.spawn_altitude[0], simulation.spawn_altitude[1])))
    return [agent_x, agent_y, agent_z], initial_velocity


def sample_speed(simulation):
    preset = simulation.effective_config.preset
    sample = simulation.np_random.uniform(low=preset.min_speed, high=preset.max_speed, size=simulation.spatial_dim)
    return float(np.linalg.norm(sample) / np.sqrt(simulation.spatial_dim))


def sample_direction(simulation):
    direction = np.zeros(EnvDim, dtype=np.float64)
    direction[:simulation.spatial_dim] = simulation.np_random.normal(size=simulation.spatial_dim)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        direction = np.zeros(EnvDim, dtype=np.float64)
        direction[0] = 1.0
        return direction
    return direction / norm


def random_pose(simulation, color):
    altitude = (simulation.plane_altitude if simulation.spatial_dim == 2 else
                float(simulation.np_random.uniform(simulation.spawn_altitude[0], simulation.spawn_altitude[1])))
    if color == 'Red':
        # Area-uniform annulus around a random asset, rejecting points outside the world.
        target = simulation.targets[int(simulation.np_random.integers(0, max(len(simulation.targets), 1)))]
        inner, outer = simulation.red_spawn_annulus
        for _ in range(4096):
            radius = float(np.sqrt(simulation.np_random.uniform(inner ** 2, outer ** 2)))
            angle = float(simulation.np_random.uniform(0.0, 2.0 * np.pi))
            random_x = float(target.position[0] + radius * np.cos(angle))
            random_y = float(target.position[1] + radius * np.sin(angle))
            if (simulation.area[0][0] <= random_x <= simulation.area[0][1]
                    and simulation.area[1][0] <= random_y <= simulation.area[1][1]):
                break
        else:
            raise ValueError("Red spawn annulus cannot provide an in-bounds position after 4096 attempts")
    else:
        random_x = float(simulation.np_random.uniform(*simulation.blue_spawn_x))
        random_y = float(simulation.np_random.uniform(simulation.area[1][0], simulation.area[1][1]))
    position = [random_x, random_y, float(np.clip(altitude, simulation.area[2][0], simulation.area[2][1]))]
    if simulation.env_agent_type == 'UAV_fixedwing':
        # Level 30 m/s trim heading east for Red and west for Blue.
        return position, [30. if color == 'Red' else -30., 0., 0.]
    if simulation.env_agent_type == 'UAV_quadrotor':
        return position, [0., 0., 0.]
    return position, (sample_direction(simulation) * sample_speed(simulation)).tolist()
