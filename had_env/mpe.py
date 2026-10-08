"""Fixed-order MPE-style lists over the simultaneous Parallel interface."""
import numpy as np
from gymnasium import spaces


class MPEEnv:
    """Fixed-order MPE-style list interface (not the old Gym package).

    ``reset`` returns obs_n; ``step`` returns obs_n, reward_n, done_n, info.
    Observations are flattened native rows. Survival dead slots are zeroed
    after their final transition. Damage dead slots immediately receive zero
    observations and keep their team reward stream until the world ends.
    Actions for already-dead slots are ignored. Discrete
    actions are integer indices, or an explicit one-hot vector matching the
    action space (9 entries in 2D, 27 in 3D).
    ``info['n']`` retains termination versus truncation for each fixed slot.
    """

    def __init__(self, env):
        self.parallel_env = env
        self.possible_agents = env.possible_agents.copy()
        self.n = len(self.possible_agents)
        self.action_space = [env.action_space(a) for a in self.possible_agents]
        self.observation_space = [spaces.flatten_space(env.observation_space(a)) for a in self.possible_agents]
        self.simulation, self.metadata = env.simulation, env.metadata
        self._done = {a: False for a in self.possible_agents}
        self._infos = {}

    @property
    def agents(self):
        return self.simulation.agents

    @property
    def unwrapped(self):
        return self.parallel_env

    def reset(self, seed=None, options=None):
        obs, self._infos = self.parallel_env.reset(seed=seed, options=options)
        self._done = {a: False for a in self.possible_agents}
        return [spaces.flatten(self.parallel_env.observation_space(a), obs[a]) for a in self.possible_agents]

    def step(self, actions):
        if len(actions) != self.n:
            raise ValueError(f"MPE actions must have {self.n} fixed slots")
        joint = {}
        for a, value in zip(self.possible_agents, actions):
            if a not in self.parallel_env.agents:
                continue
            if self.parallel_env.task_mode == "damage" and self.parallel_env._entity(a).Health <= 0:
                joint[a] = self.parallel_env._neutral_action(a) if self.parallel_env.continuous else 0
                continue
            array = np.asarray(value)
            action_count = self.parallel_env.action_space(a).n if isinstance(self.parallel_env.action_space(a), spaces.Discrete) else None
            if action_count is not None and array.shape == (action_count,):
                if not np.all((array == 0) | (array == 1)) or np.sum(array) != 1:
                    raise ValueError(f"MPE vector actions must be explicit length-{action_count} one-hot vectors")
                value = int(np.argmax(array))
            joint[a] = value
        obs, rewards, terms, truncs, infos = self.parallel_env.step(joint)
        rows, values, dones, details = [], [], [], []
        for i, a in enumerate(self.possible_agents):
            self._done[a] = self._done[a] or terms.get(a, False) or truncs.get(a, False)
            rows.append(spaces.flatten(self.parallel_env.observation_space(a), obs[a]) if a in obs else np.zeros(self.observation_space[i].shape, np.float32))
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
