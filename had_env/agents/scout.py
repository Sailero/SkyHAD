from had_env.agents.base import BaseAgent
from had_env.geometry import within_sector_area, boundary_loss
from had_env.config import ScoutAngleMax, ScoutDistanceMax, reward_scout_single


class ScoutAgent(BaseAgent):
    def __init__(self, color, id_, render_id):
        super(ScoutAgent, self).__init__(color, id_, render_id)

        self.Type = 'Scout'

    def get_function_action(self):
        return []

    def set_function_action(self, action):
        pass

    def get_action(self):
        return self.get_flying_action()

    def own_observation(self, world):
        if self.Health > 0:
            return [one for one in world if
                    within_sector_area(self.get_position(), one.get_position(), self.get_velocity(),
                                       ScoutAngleMax, ScoutDistanceMax*self.scene_scale) and one.Health > 0]
        else:
            return []

    def get_reward(self, world):
        scout_obs = self.own_observation(world)
        return reward_scout_single * len(scout_obs) - boundary_loss(self.get_position(), bounds=self.world_bounds, scene_scale=self.scene_scale)

    def choose_function_ruled_action(self, all_agents):
        return []
