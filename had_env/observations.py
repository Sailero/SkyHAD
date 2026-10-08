"""Fixed-order entity observations, alive masks and critic state."""
import numpy as np
from had_env.config import EnvDim, OBS_ENTITY_DIM
from had_env.geometry import world_diagonal


def observations(simulation, normalization=True, is_relative_observation=True):
    """Return per-agent rows of [rel_pos, rel_vel, health, alive, side flags].

    Dead entities are zeroed. ``simulation.entity_mask`` holds the matching alive bits.
    """
    observation_n = []
    mask_n = []
    diagonal = world_diagonal(simulation.world_bounds)
    speed = 2.*simulation.effective_config.preset.max_speed
    for agent in simulation.agents:
        observation = []
        mask = []
        agent_pos = np.asarray(agent.position, dtype=np.float64)
        agent_vel = np.asarray(agent.velocity, dtype=np.float64)
        for other in simulation.entities:
            if other.Id == agent.Id and is_relative_observation:
                continue
            alive = float(other.Health > 0)
            if alive <= 0:
                observation.append([0.0] * OBS_ENTITY_DIM)
                mask.append(0.0)
                continue
            if is_relative_observation:
                pos = np.asarray(other.position, dtype=np.float64) - agent_pos
                vel = np.asarray(other.velocity, dtype=np.float64) - agent_vel
                if normalization:
                    pos = pos / diagonal
                    vel = vel / speed
            else:
                pos = np.asarray(other.position, dtype=np.float64)
                vel = np.asarray(other.velocity, dtype=np.float64)
                if normalization:
                    pos = np.array([(pos[dim] - simulation.world_bounds[dim][0]) / (simulation.world_bounds[dim][1] - simulation.world_bounds[dim][0])
                                    for dim in range(EnvDim)])
                    vel = vel / speed
            row = pos.tolist() + vel.tolist() + [
                float(other.Health),
                1.0,
                1.0 if other.Color == "Red" else 0.0,
                1.0 if other.Color == "Blue" else 0.0,
                1.0 if other.Color == "Entity" else 0.0,
            ]
            observation.append(row)
            mask.append(1.0)
        observation_n.append(observation)
        mask_n.append(mask)
    simulation.entity_mask = mask_n
    return observation_n


def alive_entities(simulation):
    world_alive = [agent.Health > 0 for agent in simulation.entities]
    return world_alive


def global_state(simulation, normalization=True):
    """Flatten absolute entity rows for a centralized critic; dead rows are zero."""
    global_state = []
    speed = 2.*simulation.effective_config.preset.max_speed
    for entity in simulation.entities:
        is_alive = float(entity.Health > 0)
        flags = [
            1.0 if entity.Color == "Red" else 0.0,
            1.0 if entity.Color == "Blue" else 0.0,
            1.0 if entity.Color == "Entity" else 0.0,
        ]
        if not is_alive:
            global_state.extend([0.0] * OBS_ENTITY_DIM)
            continue
        pos = list(entity.position)
        vel = list(entity.velocity)
        if normalization:
            pos = [(pos[dim] - simulation.world_bounds[dim][0]) / (simulation.world_bounds[dim][1] - simulation.world_bounds[dim][0]) for dim in range(EnvDim)]
            vel = (np.asarray(vel, dtype=np.float64) / speed).tolist()
        global_state.extend(pos + vel + [float(entity.Health), is_alive] + flags)

    return global_state
