"""Compare scientific transitions, including observations, RNG and rigid states."""
from dataclasses import asdict, is_dataclass
import hashlib
import json

import numpy as np

from had_env import make_env
from had_env.grouping.policies import RulePolicy


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return plain(asdict(value))
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def physics(env):
    simulation = env.simulation
    return {'entities': [{key: plain(getattr(entity, key)) for key in
                         ('position', 'velocity', 'Health', 'rigid_state', 'attitude', 'angular_velocity', 'IsFire', 'IsDisturb')
                         if hasattr(entity, key)} for entity in simulation.entities],
            'state': plain(env.state()), 'rng': plain(env.np_random.bit_generator.state)}


def signatures():
    # Grouping snapshots include ambient NumPy state for branch restoration.
    # Fix that diagnostic input as well as each environment's private seed.
    np.random.seed(20261013)
    signatures = {}
    for model in ('particle', 'UAV_fixedwing', 'UAV_quadrotor'):
        controls = ('acceleration', 'position') if model == 'particle' else ('acceleration', 'position', 'actuator')
        for control in controls:
            for task in ('survival', 'damage'):
                for dimension in ((2, 3) if model == 'particle' else (3,)):
                    for api in ('parallel', 'mpe'):
                        for seed in (17, 42):
                            env = make_env(api=api, env_agent_type=model, env_agent_action_type=control,
                                           task_mode=task, spatial_dim=dimension, red_count=3, blue_count=3,
                                           target_count=2, red_scouts=1, blue_disturbers=1,
                                           max_cycles=3, record_events=True)
                            native = env if api == 'parallel' else env.parallel_env
                            rows = []
                            try:
                                rows.append([plain(env.reset(seed=seed)), physics(native)])
                                for i, agent in enumerate(native.possible_agents):
                                    native.action_space(agent).seed(seed * 100 + i)
                                while native.agents:
                                    actions = {a: native.action_space(a).sample() for a in native.agents}
                                    supplied = actions if api == 'parallel' else [actions.get(a, 0) for a in native.possible_agents]
                                    rows.append([plain(supplied), plain(env.step(supplied)), physics(native)])
                            finally:
                                env.close()
                            name = f'{model}/{control}/{task}/{dimension}D/{api}/seed{seed}'
                            signatures[name] = hashlib.sha256(json.dumps(rows, sort_keys=True, allow_nan=True).encode()).hexdigest()
        for task in ('survival', 'damage'):
            for dimension in ((2, 3) if model == 'particle' else (3,)):
                for seed in (17, 42):
                    env = make_env(api='grouping', env_agent_type=model, task_mode=task, spatial_dim=dimension,
                                   red=3, blue=3, targets=2, max_steps=3, command_interval=2)
                    rows, policy = [], RulePolicy('rule')
                    try:
                        state = env.reset(seed=seed)
                        rows.append(plain(state))
                        while not env.done:
                            action = policy.act(state)
                            state, reward, done, info = env.step(action)
                            rows.append(plain([action, state, reward, done, info, env.adapter.snapshot()]))
                    finally:
                        env.close()
                    name = f'{model}/grouping/{task}/{dimension}D/seed{seed}'
                    signatures[name] = hashlib.sha256(json.dumps(rows, sort_keys=True, allow_nan=True).encode()).hexdigest()
    return signatures


if __name__ == '__main__':
    print(json.dumps(signatures(), sort_keys=True))
