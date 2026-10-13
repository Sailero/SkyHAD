"""Episode placement; random draws retain their native order."""
import numpy as np
from had_env.config import AttackRatio, DisturbRatio, EnvDim, PlanarSpawnSeparation, ScoutRatio


def reset_random(simulation, evaluate=False):
    # Target layout is independent of the agents' random/uniform reset mode.
    reset_targets(simulation)

    for agent_id in range(len(simulation.agents)):
        initial_position, initial_velocity = random_pose(simulation, simulation.agents[agent_id].Color, evaluate=evaluate)
        simulation.agents[agent_id].reset(initial_position, initial_velocity)

    separate_spawns(simulation, evaluate=evaluate)
    simulation.reset_episode_state()


def reset_uniform(simulation):
    """Place each role in the same order used to create the entity slots."""
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


def reset_targets(simulation):
    if simulation.target_initialization == 'fixed':
        for target, position in zip(simulation.targets, simulation.fixed_target_positions):
            target.reset(position.tolist(), [0.0] * EnvDim)
        return
    placed = []
    spacing = max(float(PlanarSpawnSeparation)*simulation.scene_scale, simulation.collision_distance + 1e-6)
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


def separate_spawns(simulation, evaluate=False):
    """Reject close planar openings using only this environment's RNG.

    The 3D path and its random draws remain unchanged. This prevents initial
    overlaps from height projection; it does not prevent later collisions.
    """
    if simulation.spatial_dim != 2 and simulation.env_agent_type == "particle":
        return
    spacing = max(float(PlanarSpawnSeparation)*simulation.scene_scale, simulation.collision_distance + 1e-6)
    placed = [np.asarray(target.position) for target in simulation.targets]
    for agent in simulation.agents:
        point, velocity = agent.position, agent.velocity
        for _ in range(4096):
            if all(np.linalg.norm(np.asarray(point) - previous) > spacing for previous in placed):
                break
            point, velocity = random_pose(simulation, agent.Color, evaluate=evaluate)
        else:
            raise ValueError("Spawn regions cannot provide the required planar separation")
        if point is not agent.position:
            agent.reset(point, velocity)
        placed.append(np.asarray(agent.position))


def uniform_pose(simulation, type_id, type_agent_n, ratio, color):

    length_interval = (simulation.area[1][1] - simulation.area[1][0]) / (type_agent_n + 1)
    agent_y = length_interval * (type_id + 1) + simulation.area[1][0]

    if color == 'Blue':
        agent_x = simulation.area[0][1] - (simulation.area[0][1] - simulation.area[0][0]) * ratio
        initial_velocity = [-simulation.effective_config.preset.min_speed] + [0.0] * (EnvDim - 1)
    else:
        agent_x = (simulation.area[0][1] - simulation.area[0][0]) * ratio + simulation.area[0][0]
        initial_velocity = [simulation.effective_config.preset.min_speed] + [0.0] * (EnvDim - 1)
    agent_z = (simulation.plane_altitude if simulation.spatial_dim == 2 else
               float(simulation.np_random.uniform(simulation.spawn_altitude[0], simulation.spawn_altitude[1])))
    return [agent_x, agent_y, agent_z], initial_velocity


def sample_speed(simulation):
    sample = simulation.np_random.uniform(low=simulation.effective_config.preset.min_speed, high=simulation.effective_config.preset.max_speed, size=simulation.spatial_dim)
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


def random_pose(simulation, color, evaluate=False):
    altitude = (simulation.plane_altitude if simulation.spatial_dim == 2 else
                float(simulation.np_random.uniform(simulation.spawn_altitude[0], simulation.spawn_altitude[1])))
    if color == 'Red':
        target = simulation.targets[int(simulation.np_random.integers(0, max(len(simulation.targets), 1)))]
        inner, outer = simulation.red_spawn_annulus
        for _ in range(4096 if simulation.spatial_dim == 2 else 1):
            radius = float(np.sqrt(simulation.np_random.uniform(inner ** 2, outer ** 2)))
            angle = float(simulation.np_random.uniform(0.0, 2.0 * np.pi))
            random_x = float(target.position[0] + radius * np.cos(angle))
            random_y = float(target.position[1] + radius * np.sin(angle))
            if simulation.spatial_dim == 3:
                random_x = float(np.clip(random_x, simulation.area[0][0], simulation.area[0][1]))
                random_y = float(np.clip(random_y, simulation.area[1][0], simulation.area[1][1]))
                break
            if (simulation.area[0][0] <= random_x <= simulation.area[0][1]
                    and simulation.area[1][0] <= random_y <= simulation.area[1][1]):
                break
        else:
            raise ValueError("Red spawn annulus cannot provide an in-bounds planar position after 4096 attempts")
    else:
        low, high = simulation.blue_spawn_x
        if not evaluate:
            low, high = simulation.blue_spawn_x
        random_x = float(simulation.np_random.uniform(low, high))
        random_y = float(simulation.np_random.uniform(simulation.area[1][0], simulation.area[1][1]))
    random_z = float(np.clip(altitude, simulation.area[2][0], simulation.area[2][1]))
    velocity = (sample_direction(simulation) * sample_speed(simulation)).tolist()
    if simulation.env_agent_type == 'UAV_fixedwing':
        velocity = [30. if color == 'Red' else -30., 0., 0.]
    elif simulation.env_agent_type == 'UAV_quadrotor':
        velocity = [0., 0., 0.]
    return [random_x, random_y, random_z], velocity
