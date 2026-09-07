"""Simultaneous multi-agent interfaces over the unchanged native HAD world.

The default API follows PettingZoo ParallelEnv. MPEEnv is a small fixed-list
adapter for legacy consumers; it does not change physical stepping or reward.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping
from threading import RLock

import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from pettingzoo import ParallelEnv

from had_env.actions import ACCELERATION_PRIMITIVES
from had_env.core.config import Interval
from had_env.scenarios.defense import Scenario


_LEGACY_RNG_LOCK = RLock()


class HADParallelEnv(ParallelEnv):
    """Both teams submit actions from the same pre-step observations.

    Observation rows retain native HAD normalization, in fixed world order
    excluding self: relative position (3), relative velocity (3), alive (1).
    They include all entities; scout visibility is not newly imposed.

    By default scalar reward equals the agent's team's ``RealReward``. Set an
    explicit four-element ``reward_weights`` to take a weighted sum of native
    ``LatentReward=[hit, target, enemy, episode]`` instead. Both native values
    are always available in infos. The episode component already includes
    RealReward and must not be added a second time.

    A killed agent receives its final native transition and is then removed
    from ``agents``; its fixed entity slot remains observable by survivors.
    ``max_cycles`` is a sampling truncation and never awards a new win.
    """

    metadata = {"name": "had_defense_v1", "render_modes": ["human", "rgb_array"],
                "render_fps": 30, "is_parallelizable": True}

    def __init__(self, scenario=None, *, max_cycles=100, continuous=False,
                 render_mode=None, reward_weights=None, seed=None, record_events=False):
        if type(max_cycles) is not int or max_cycles < 1:
            raise ValueError("max_cycles must be a positive integer")
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError("render_mode must be None, 'human', or 'rgb_array'")
        if not isinstance(continuous, bool):
            raise ValueError("continuous must be a boolean")
        self.scenario = scenario or Scenario()
        self.max_cycles, self.continuous, self.render_mode = max_cycles, continuous, render_mode
        self.reward_weights = None
        if reward_weights is not None:
            weights = np.asarray(reward_weights, dtype=np.float64)
            if weights.shape != (4,) or not np.isfinite(weights).all():
                raise ValueError("reward_weights must contain four finite values")
            self.reward_weights = weights.copy()
        self.world = self.scenario.make_world(seed=seed)
        self.world.record_events = bool(record_events)
        self.possible_agents = ([f"red_{i}" for i in range(self.scenario.red_count)]
                                + [f"blue_{i}" for i in range(self.scenario.blue_count)])
        self.agent_name_mapping = {name: i for i, name in enumerate(self.possible_agents)}
        self.entity_names = self.possible_agents + [f"target_{i}" for i in range(self.scenario.target_count)]
        shape = (len(self.entity_names) - 1, 7)
        # Legacy normalization is not guaranteed to be inside [-1, 1].
        self.observation_spaces = {a: spaces.Box(-np.inf, np.inf, shape, np.float32)
                                   for a in self.possible_agents}
        self.action_spaces = {a: (spaces.Box(-1., 1., (3,), np.float32) if continuous
                                  else spaces.Discrete(len(ACCELERATION_PRIMITIVES)))
                              for a in self.possible_agents}
        self.state_space = spaces.Box(-np.inf, np.inf, (len(self.entity_names) * 7,), np.float32)
        self.agents = []
        self.num_cycles, self.outcome_red = 0, 0
        self._has_reset, self._closed = False, False
        self.seed(seed)

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def seed(self, seed=None):
        """Reseed only this environment and its spaces; call reset afterward."""
        self.np_random, value = seeding.np_random(seed)
        self.np_random_seed = int(value)
        self.world.np_random = self.np_random
        # One legacy anti-parallel rotation fallback uses np.random.random.
        self._legacy_random_state = np.random.RandomState(int(value) % (2**32)).get_state()
        children = np.random.SeedSequence(value).spawn(2 * len(self.possible_agents))
        for i, agent in enumerate(self.possible_agents):
            self.action_spaces[agent].seed(int(children[2*i].generate_state(1)[0]))
            self.observation_spaces[agent].seed(int(children[2*i+1].generate_state(1)[0]))
        self._has_reset = False
        return [int(value)]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        native_obs, _, _ = self.scenario.reset_world(self.world, options=options)
        self.agents = self.possible_agents.copy()
        self.num_cycles, self.outcome_red = 0, 0
        self._has_reset, self._closed = True, False
        obs = self._observations(native_obs, self.agents)
        infos = {a: self._info(a) for a in self.agents}
        if self.render_mode == "human":
            self.render()
        return obs, infos

    def _observations(self, values, agents):
        return {a: np.asarray(values[self.agent_name_mapping[a]], dtype=np.float32).copy()
                for a in agents}

    def _decode_action(self, agent, action):
        if not self.continuous:
            if isinstance(action, (bool, np.bool_)) or not self.action_space(agent).contains(action):
                raise ValueError(f"{agent} action must be an integer in 0..26")
            # tolist preserves the original discrete-control arithmetic path.
            return ACCELERATION_PRIMITIVES[int(action)].tolist()
        try:
            value = np.asarray(action)
        except (ValueError, TypeError) as error:
            raise ValueError(f"{agent} action must be a finite 3-vector in [-1,1]") from error
        if (value.shape != (3,) or value.dtype.kind not in "iuf"
                or not np.isfinite(value).all() or np.any(np.abs(value) > 1)):
            raise ValueError(f"{agent} action must be a finite 3-vector in [-1,1]")
        return value.tolist()

    def step(self, actions):
        if not self._has_reset or self._closed:
            raise RuntimeError("Call reset before stepping an uninitialized or closed environment")
        if not isinstance(actions, Mapping) or set(actions) != set(self.agents):
            raise ValueError("actions must contain exactly the current env.agents keys")
        if not self.agents:
            return {}, {}, {}, {}, {}
        acting = self.agents.copy()
        # Decode the complete joint action before touching physics or RNG.
        decoded = {a: self._decode_action(a, actions[a]) for a in acting}
        batch = [decoded.get(a, [0., 0., 0.]) for a in self.possible_agents]
        with _LEGACY_RNG_LOCK:
            ambient = np.random.get_state()
            np.random.set_state(self._legacy_random_state)
            try:
                self.world.step_physics(batch)
                native_obs = self.scenario.observation(self.world)
                native_rewards = self.scenario.reward(self.world)
            finally:
                self._legacy_random_state = np.random.get_state()
                np.random.set_state(ambient)
        self.num_cycles += 1
        self.outcome_red = self.scenario.done(self.world)
        terminations = {a: bool(self.outcome_red or self._entity(a).Health <= 0) for a in acting}
        truncations = {a: bool(self.num_cycles >= self.max_cycles and not terminations[a]) for a in acting}
        rewards, infos = {}, {}
        for a in acting:
            info = self._info(a, native_rewards)
            info["terminated"], info["truncated"] = terminations[a], truncations[a]
            rewards[a] = (info["RealReward"] if self.reward_weights is None else
                          float(np.dot(info["LatentReward"], self.reward_weights)))
            infos[a] = info
        observations = self._observations(native_obs, acting)
        self.agents = [a for a in acting if not (terminations[a] or truncations[a])]
        if self.render_mode == "human":
            self.render()
        return observations, rewards, terminations, truncations, infos

    def _entity(self, agent):
        return self.world.agents[self.agent_name_mapping[agent]]

    def _info(self, agent, rewards=None):
        entity = self._entity(agent)
        side = entity.Color
        index = self.agent_name_mapping[agent] - (self.scenario.red_count if side == "Blue" else 0)
        real = float(rewards["RealReward"][side]) if rewards else 0.
        latent = list(map(float, rewards["LatentReward"][side][index])) if rewards else [0.] * 4
        return {"entity_id": int(entity.Id), "side": side.lower(), "role": entity.Type.lower(),
                "alive": bool(entity.Health > 0), "health": float(entity.Health),
                "observation_entities": tuple(n for n in self.entity_names if n != agent),
                "RealReward": real, "LatentReward": latent, "outcome_red": self.outcome_red,
                "cycle": self.num_cycles, "sim_time": self.num_cycles * Interval,
                "events": copy.deepcopy(self.world.last_physics_events),
                "core_version": self.world.core_version, "physics_protocol": self.world.physics_protocol}

    def state(self):
        if not self._has_reset:
            raise RuntimeError("Call reset before requesting state")
        return np.asarray(self.world.get_global_state(), dtype=np.float32)

    def render(self):
        if not self._has_reset or self._closed:
            raise RuntimeError("Call reset before rendering")
        if self.render_mode == "rgb_array":
            return self.world.render_rgb_array()
        if self.render_mode == "human":
            self.world.render()
        return None

    def close(self):
        self.world.close()
        self._closed = True


class MPEEnv:
    """Fixed-order MPE-style list interface (not the old Gym package).

    ``reset`` returns obs_n; ``step`` returns obs_n, reward_n, done_n, info.
    Observations are flattened native rows. Dead slots are zeroed after their
    final transition; actions for already-dead slots are ignored. Discrete
    actions are integer indices, or an explicit length-27 one-hot vector.
    ``info['n']`` retains termination versus truncation for each fixed slot.
    """

    def __init__(self, env):
        self.parallel_env = env
        self.possible_agents = env.possible_agents.copy()
        self.n = len(self.possible_agents)
        self.action_space = [env.action_space(a) for a in self.possible_agents]
        self.observation_space = [spaces.flatten_space(env.observation_space(a)) for a in self.possible_agents]
        self.world, self.metadata = env.world, env.metadata
        self._done = {a: False for a in self.possible_agents}
        self._infos = {}

    @property
    def agents(self):
        return self.world.agents

    @property
    def unwrapped(self):
        return self.parallel_env

    def reset(self, seed=None, options=None):
        obs, self._infos = self.parallel_env.reset(seed=seed, options=options)
        self._done = {a: False for a in self.possible_agents}
        return [obs[a].reshape(-1) for a in self.possible_agents]

    def step(self, actions):
        if len(actions) != self.n:
            raise ValueError(f"MPE actions must have {self.n} fixed slots")
        joint = {}
        for a, value in zip(self.possible_agents, actions):
            if a not in self.parallel_env.agents:
                continue
            array = np.asarray(value)
            if not self.parallel_env.continuous and array.shape == (27,):
                if not np.all((array == 0) | (array == 1)) or np.sum(array) != 1:
                    raise ValueError("MPE vector actions must be explicit length-27 one-hot vectors")
                value = int(np.argmax(array))
            joint[a] = value
        obs, rewards, terms, truncs, infos = self.parallel_env.step(joint)
        rows, values, dones, details = [], [], [], []
        for i, a in enumerate(self.possible_agents):
            self._done[a] = self._done[a] or terms.get(a, False) or truncs.get(a, False)
            rows.append(obs[a].reshape(-1) if a in obs else np.zeros(self.observation_space[i].shape, np.float32))
            values.append(rewards.get(a, 0.))
            dones.append(self._done[a])
            info = infos.get(a, {**self._infos.get(a, {}), "inactive": True})
            if a not in infos:
                info = {**info, "RealReward": 0., "LatentReward": [0.] * 4, "events": []}
            self._infos[a] = info
            details.append(info)
        return rows, values, dones, {"n": details}

    def seed(self, seed=None):
        return self.parallel_env.seed(seed)

    def state(self):
        return self.parallel_env.state()

    def render(self, mode=None):
        if mode is not None:
            if mode not in self.metadata["render_modes"]:
                raise ValueError("unsupported render mode")
            self.parallel_env.render_mode = mode
        return self.parallel_env.render()

    def close(self):
        self.parallel_env.close()
