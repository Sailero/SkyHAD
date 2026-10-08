import numpy as np
from had_env.agents.base import BaseAgent
from had_env.geometry import within_sector_area, disturb_intensity_ratio, boundary_loss
from had_env.config import DisturbAngleMax, DisturbDistanceMax, DisturbIntensity, reward_disturb_single


class DisturbAgent(BaseAgent):
    def __init__(self, color, id_, render_id):
        super(DisturbAgent, self).__init__(color, id_, render_id)

        self.Type = 'Disturb'
        self.IsDisturb = False

    def reset(self, initial_position, initial_velocity):
        super(DisturbAgent, self).reset(initial_position, initial_velocity)
        self.IsDisturb = False

    def get_function_action(self):
        return [self.IsDisturb]

    def set_function_action(self, is_disturb):
        self.IsDisturb = is_disturb

    def get_action(self):
        return super(DisturbAgent, self).get_flying_action() + [self.IsDisturb]

    def own_observation(self, world):
        if self.Health > 0:
            return [one for one in world if
                    within_sector_area(self.get_position(), one.get_position(), self.get_velocity(),
                                       DisturbAngleMax, DisturbDistanceMax*self.scene_scale) and one.Health > 0]
        else:
            return []

    def get_reward(self, world):
        if self.IsDisturb and self.Health > 0:
            DisturbIntensityRatio = 0
            for one in world:

                if one.Color == 'Entity':
                    continue

                if one.Color == self.Color:
                    DisturbIntensityRatio -= disturb_intensity_ratio(self.get_position(), one.get_position(),
                                                                     self.get_velocity(), self.scene_scale)
                else:
                    DisturbIntensityRatio += disturb_intensity_ratio(self.get_position(), one.get_position(),
                                                                     self.get_velocity(), self.scene_scale)

            return reward_disturb_single * DisturbIntensityRatio * DisturbIntensity - boundary_loss(self.get_position(), bounds=self.world_bounds, scene_scale=self.scene_scale)
        else:
            return 0


    def choose_function_ruled_action(self, all_agents):
        own_obs = self.own_observation(all_agents)

        RedDisturbRatio = [disturb_intensity_ratio(self.get_position(), one.get_position(), self.get_velocity(), self.scene_scale)
                           for one in own_obs if one.Color == 'Red']
        BlueDisturbRatio = [disturb_intensity_ratio(self.get_position(), one.get_position(), self.get_velocity(), self.scene_scale)
                            for one in own_obs if one.Color == 'Blue']

        IsDisturb = False

        if self.Color == 'Red':
            IsDisturb = np.sum(RedDisturbRatio) < np.sum(BlueDisturbRatio)
        elif self.Color == 'Blue':
            IsDisturb = np.sum(RedDisturbRatio) >= np.sum(BlueDisturbRatio)

        return IsDisturb
