"""Survival outcomes and per-step target damage rewards."""
from had_env.config import reward_episode


def rewards(simulation):
    """Return team RealReward and per-agent [hit, target, enemy, episode] components."""

    reward_n = {
        "RealReward": {
            "Red": 0,
            "Blue": 0
        },
        "LatentReward": {
            "Red": [[0.0, 0.0, 0.0, 0.0] for _ in range(simulation.red_agent_n)],
            "Blue": [[0.0, 0.0, 0.0, 0.0] for _ in range(simulation.blue_agent_n)]
        }
    }

    if simulation.task_mode == 'damage':
        # Only this physical step's new target damage is rewarded. Reading
        # rewards or reaching termination never adds cumulative damage again.
        damage = simulation.step_target_damage
        reward_n['RealReward'] = {'Red': -damage, 'Blue': damage}
        return reward_n

    for i, agent in enumerate(simulation.red_agents):
        if agent.Health > 0:
            indiv_reward = agent.get_attack_reward(simulation.entities)
            for dim in range(3):
                reward_n["LatentReward"]["Red"][i][dim] = indiv_reward[dim]

    for i, agent in enumerate(simulation.blue_agents):
        if agent.Health > 0:
            indiv_reward = agent.get_attack_reward(simulation.entities)
            for dim in range(3):
                reward_n["LatentReward"]["Blue"][i][dim] = indiv_reward[dim]

    episode_scale = simulation.reward_config.get("episode", reward_episode)
    episode_reward = 0.0 if abs(episode_scale) < 1e-12 else episode_scale * survival_outcome(simulation)
    reward_n["RealReward"]["Red"] = episode_reward
    reward_n["RealReward"]["Blue"] = -episode_reward

    for i in range(simulation.red_agent_n):
        reward_n["LatentReward"]["Red"][i][3] = reward_n["RealReward"]["Red"]
    for i in range(simulation.blue_agent_n):
        reward_n["LatentReward"]["Blue"][i][3] = reward_n["RealReward"]["Blue"]

    return reward_n


def agent_done(simulation):
    """Survival ends dead slots individually; damage keeps every slot until completion."""
    is_terminal = simulation.is_episode_done()
    if simulation.task_mode == 'damage':
        # Fixed team slots keep receiving later team damage rewards; death
        # remains observable through each entity's health/alive mask.
        return [is_terminal] * len(simulation.agents)

    red_done = [is_terminal or (agent.Health <= 0) for agent in simulation.red_agents]

    blue_done = [is_terminal or (agent.Health <= 0) for agent in simulation.blue_agents]

    return red_done + blue_done


def entity_info(simulation):
    info = {"core_version": simulation.core_version, "physics_protocol": simulation.physics_protocol,
            **task_info(simulation)}
    for agent_id in range(len(simulation.targets)):
        info['target_%d' % agent_id] = {
            "position": simulation.targets[agent_id].get_position(),
            "health": float(simulation.targets[agent_id].Health),
            "is_alive": bool(simulation.targets[agent_id].Health > 0),
            "step_damage": float(simulation.targets[agent_id].step_damage),
            "cumulative_damage": float(simulation.targets[agent_id].cumulative_damage),
        }
    for agent_id in range(len(simulation.agents)):
        info['agent_%d' % agent_id] = {
            "velocity": simulation.agents[agent_id].get_velocity(),
            "position": simulation.agents[agent_id].get_position(),
            "is_alive": simulation.agents[agent_id].Health > 0,
            "Color": simulation.agents[agent_id].Color
        }
    return info


def survival_outcome(simulation):
    """Legacy survival outcome; use is_episode_done for task completion.

    A damage score is not a win/loss sign. Returning zero in that mode also
    keeps the native renderer from drawing the survival result banner.
    """
    if simulation.task_mode == 'damage':
        return 0
    reason = termination_reason(simulation)
    if reason == 'target_destroyed':
        return -1
    elif reason == 'blue_attackers_destroyed':
        return 1
    else:
        return 0


def termination_reason(simulation):
    if simulation.task_mode == 'survival':
        if any(target.Health < 1e-3 for target in simulation.targets):
            return 'target_destroyed'
        blue_eliminated = all(agent.Health < 1e-3 for agent in simulation.blue_agents
                              if agent.Type == 'Attack')
    else:
        blue_eliminated = all(agent.Health <= 0 for agent in simulation.blue_agents
                              if agent.Type == 'Attack')
    return 'blue_attackers_destroyed' if blue_eliminated else None


def task_info(simulation):
    """Expose task state without accumulating rewards on repeated reads.

    Per-target counters live in entity state so native deep copies and
    grouping entity snapshots retain the complete damage branch point.
    Survival returns remain governed by the existing reward contract.
    """
    by_target = {int(target.Id): float(target.cumulative_damage) for target in simulation.targets}
    total = float(sum(by_target.values()))
    reason = termination_reason(simulation)
    return {
        'task_mode': simulation.task_mode,
        'env_agent_type': simulation.env_agent_type,
        'env_agent_action_type': simulation.env_agent_action_type,
        'effective_config': simulation.effective_config.to_dict(),
        'world_bounds': simulation.world_bounds.tolist(),
        'scene_scale': simulation.scene_scale,
        'target_initialization': simulation.target_initialization,
        'spatial_dim': simulation.spatial_dim,
        'plane_altitude': simulation.plane_altitude,
        'attack_distance': list(simulation.attack_distance),
        'fire_range': simulation.fire_range,
        'step_target_damage': simulation.step_target_damage,
        'target_damage': total,
        'target_damage_by_target': by_target,
        'episode_returns': {'Red': -total, 'Blue': total} if simulation.task_mode == 'damage' else None,
        'episode_done': reason is not None,
        'termination_reason': reason,
    }
