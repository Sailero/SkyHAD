from had_env.agents.base import BaseAgent
from had_env.geometry import distance, distances_from
import numpy as np


class AttackAgent(BaseAgent):
    def __init__(self, color, id_, render_id):
        super(AttackAgent, self).__init__(color, id_, render_id)

        self.Type = 'Attack'
        self.IsFire = False

    def reset(self, initial_position, initial_velocity):
        super(AttackAgent, self).reset(initial_position, initial_velocity)
        self.IsFire = False

    def update_status(self, all_agents):
        super(AttackAgent, self).update_status(all_agents)

        if self.IsFire:
            self.Health = 0

    def get_function_action(self):
        return [self.IsFire]

    def set_function_action(self, is_fire):
        self.IsFire = is_fire

    def get_action(self):
        return super(AttackAgent, self).get_flying_action() + [self.IsFire]

    def own_observation(self, world):
        if self.Health > 0:
            others = [one for one in world if one is not self]
            separations = distances_from(self.get_position(), [one.get_position() for one in others])
            attack_range = self.fire_range
            return [one for one, separation in zip(others, separations)
                    if separation < attack_range and one.Health > 0]
        else:
            return []

    def choose_function_ruled_action(self, all_agents):
        fire_range = self.fire_range
        origin = self.get_position()
        if self.Color == 'Blue':
            return any(
                one.Color == 'Entity' and one.Health > 0 and distance(origin, one.get_position()) < fire_range
                for one in all_agents
            )
        if self.Color == 'Red':
            return any(
                one.Color == 'Blue' and one.Health > 0 and distance(origin, one.get_position()) < fire_range
                for one in all_agents
            )
        return False

    def calculate_normalized_distance_to_targets(self, targets):
        """Return the nearest live target distance divided by the world diagonal."""
        if len(targets) == 0:
            return 0

        min_distance = float('inf')
        for target in targets:
            if target.Health > 0:
                dist = distance(self.get_position(), target.get_position())
                min_distance = min(min_distance, dist)

        if min_distance == float('inf'):
            return 0

        d_max = np.sqrt(sum([(self.world_bounds[i][1] - self.world_bounds[i][0])**2 for i in range(3)]))
        return min_distance / d_max

    def calculate_normalized_distance_to_color(self, agents, target_color):
        """Return nearest live opponent distance, with the native Blue offset."""
        target_agents = [a for a in agents if a.Color == target_color and a != self and a.Health > 0]
        if len(target_agents) == 0:
            return 0.0

        min_distance = float('inf')
        for target_agent in target_agents:
            dist = distance(self.get_position(), target_agent.get_position())
            min_distance = min(min_distance, dist)

        d_max = np.sqrt(sum([(self.world_bounds[i][1] - self.world_bounds[i][0]) ** 2 for i in range(3)]))

        if self.Color == "Blue":
            return min_distance / d_max - 1
        else:
            return min_distance / d_max

    def calculate_health_damage(self, agents):
        """Sum the native geometric hit-shaping signal over live opponents."""

        enemy_agents = [a for a in agents if a.Color != self.Color and a.Color != 'Entity' and a.Health > 0]
        if len(enemy_agents) == 0:
            return 0.0

        total_damage = 0.0
        for enemy in enemy_agents:
            dist = distance(self.get_position(), enemy.get_position())
            if dist <= self.attack_distance[1]:
                damage_ratio = max(0, 1 - dist / self.attack_distance[1])
                total_damage += damage_ratio

        return total_damage

    def get_attack_reward(self, world):
        """Return [hit, target, enemy] shaping; tasks adds the shared episode reward."""
        agents = [agent for agent in world if hasattr(agent, 'Color') and agent.Color in ['Red', 'Blue']]
        targets = [target for target in world if hasattr(target, 'Color') and target.Color == 'Entity']

        r_hit = self.calculate_health_damage(agents)

        r_target = -self.calculate_normalized_distance_to_targets(targets)

        if self.Color == 'Red':
            r_enemy = -self.calculate_normalized_distance_to_color(agents, 'Blue')
        else:
            r_enemy = self.calculate_normalized_distance_to_color(agents, 'Red')

        return [r_hit, r_target, r_enemy]
