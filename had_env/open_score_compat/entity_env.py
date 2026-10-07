"""The standalone Open-SCORE ALMA entity-environment contract for HAD."""
from __future__ import annotations

# Adapted from Open-SCORE open_score/envs/entity_env.py and utils/seeding.py; native HAD physics remains in had_env.

import copy
import numpy as np

from .features import (ENTITY_DIM, N_ACTIONS, masks_from_entity_mask,
                       available_actions, task_masks)
from .wrapper import HADWrapper
from .scales import ScaleSampler, as_scale


def split_seeds(seed, worker_id=0):
    streams = np.random.SeedSequence([int(seed), int(worker_id)]).spawn(3)
    values = [int(stream.generate_state(1)[0]) for stream in streams]
    return {"train_seed": int(seed), "environment_seed": values[0],
            "scale_seed": values[1], "evaluation_seed": values[2]}

class EpisodeSeedStream:
    def __init__(self, seed):
        self.rng = np.random.default_rng(int(seed))

    def next(self):
        # Keep all training initial conditions outside frozen validation/anchor IDs.
        return int(self.rng.integers(100_000, 2 ** 31 - 1))

    def state_dict(self):
        return self.rng.bit_generator.state

    def load_state_dict(self, state):
        self.rng.bit_generator.state = state

class HADEntityEnv:
    def __init__(self, seed=0, worker_id=0, train_dist="mixed_le10", scale=None, config=None,
                 max_steps=100, episode_limit=None, blue_upper="reactive", blue_lower="rush",
                 entity_scheme=True, gamma=0.99, **kwargs):
        if not entity_scheme:
            raise ValueError("HAD cross-scale experiments require the entity scheme")
        seeds = split_seeds(seed, worker_id)
        self.scale_sampler = ScaleSampler(kwargs.pop("scale_seed", seeds["scale_seed"]), train_dist)
        self.episode_seeds = EpisodeSeedStream(kwargs.pop("environment_seed", seeds["environment_seed"]))
        self.fixed_scale = as_scale(config if config is not None else scale) if config is not None or scale is not None else None
        self.episode_limit = int(max_steps if episode_limit is None else episode_limit)
        self.wrapper = HADWrapper(scale=self.fixed_scale or (8, 8, 2), max_steps=self.episode_limit,
                                  blue_upper=blue_upper, blue_lower=blue_lower, gamma=gamma, **kwargs)
        self.n_agents = self.wrapper.n_red
        self.n_blue = self.wrapper.n_blue
        self.n_targets = self.wrapper.n_targets
        self.n_entities = self.wrapper.n_entities
        self.n_actions = N_ACTIONS
        self.train_dist = train_dist

    def reset(self, seed=None, config=None, evaluate=False, retain_trajectory=False,
              test=False, **kwargs):
        episode_seed = self.episode_seeds.next() if seed is None else int(seed)
        scale = as_scale(config) if config is not None else self.fixed_scale or self.scale_sampler.sample()
        self.wrapper.reset(episode_seed, scale, evaluate=evaluate or test,
                           retain_trajectory=retain_trajectory, **kwargs)
        return self.get_entities(), self.get_masks()

    def step(self, actions):
        return self.wrapper.step(actions)

    def get_entities(self):
        return self.wrapper.entities.copy()

    def get_entity_size(self):
        return ENTITY_DIM

    def get_masks(self):
        return masks_from_entity_mask(self.wrapper.entity_mask)

    def get_agent_mask(self):
        return 1 - self.wrapper.entity_mask[:self.n_agents].copy()

    def get_initial_agent_mask(self):
        """One denotes initial padding, while deaths remain graph nodes."""
        mask = np.ones(self.n_agents, dtype=np.uint8)
        mask[:self.wrapper.scale.N_R] = 0
        return mask

    def get_avail_actions(self):
        return available_actions(self.wrapper.entity_mask, self.n_agents)

    def get_avail_agent_actions(self, agent_id):
        return self.get_avail_actions()[agent_id]

    def get_total_actions(self):
        return N_ACTIONS

    def get_state(self):
        env = self.wrapper.adapter.env
        fractions = np.asarray([sum(a.Health > 0 for a in env.red_agents) / len(env.red_agents),
                                sum(a.Health > 0 for a in env.blue_agents) / len(env.blue_agents),
                                1 - self.wrapper.adapter.step_count / self.episode_limit], dtype=np.float32)
        return np.concatenate((self.wrapper.entities.reshape(-1), fractions))

    def get_state_size(self):
        return self.n_entities * ENTITY_DIM + 3

    def get_obs(self):
        table = self.get_entities()
        return np.repeat(table.reshape(1, -1), self.n_agents, axis=0) * self.get_agent_mask()[:, None]

    def get_obs_agent(self, agent_id):
        return self.get_obs()[agent_id]

    def get_obs_size(self):
        return self.n_entities * ENTITY_DIM

    def get_env_info(self, args=None):
        return {"n_agents": self.n_agents, "n_entities": self.n_entities, "n_actions": N_ACTIONS,
                "entity_shape": ENTITY_DIM, "state_shape": self.get_state_size(),
                "obs_shape": self.get_obs_size(), "episode_limit": self.episode_limit,
                "gt_mask_avail": False, "feature_layout": "had",
                "n_tasks": self.wrapper.n_tasks()}

    def get_task_masks(self):
        masks = task_masks(self.wrapper.entities, self.wrapper.entity_mask, self.wrapper.scale.K,
                           self.n_agents, self.n_blue, self.n_targets,
                           subtask_set=self.wrapper.subtask_set)
        masks["hier_decision"] = np.asarray([int(self.wrapper.hier_decision)], dtype=np.uint8)
        return masks

    def episode_summary(self):
        return self.wrapper.episode_summary()

    def get_stats(self):
        return self.episode_summary() if self.wrapper.adapter is not None else {}

    def get_agg_stats(self, stats):
        return {}

    def get_policy_state(self):
        return self.wrapper.get_policy_state()

    def get_trajectory(self):
        return copy.deepcopy(self.wrapper.diagnostics.trajectory)

    def get_rng_state(self):
        return {"scale": copy.deepcopy(self.scale_sampler.state_dict()),
                "episode": copy.deepcopy(self.episode_seeds.state_dict())}

    def set_rng_state(self, state):
        self.scale_sampler.load_state_dict(state["scale"])
        self.episode_seeds.load_state_dict(state["episode"])

    def save_replay(self):
        return self.get_trajectory()

    def close(self):
        self.wrapper.close()

