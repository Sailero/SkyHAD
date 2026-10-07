"""Single public factory; explicit keyword values override configuration files."""
from dataclasses import fields
from had_env.config import EnvConfig, load_config
from had_env.environment import HADParallelEnv, MPEEnv
from had_env.scenarios.defense import Scenario


def make_env(scenario_name="defense", *, config=None, **kwargs):
    if scenario_name not in ("defense", "defense.py"):
        raise ValueError(f"Unknown scenario {scenario_name!r}; available: defense")
    values = load_config(config)
    values.update(kwargs)
    api = values.pop("api", "parallel")
    if api == "open_score":
        from had_env.open_score_compat import make_open_score_env
        if 'max_cycles' in values:
            values['max_steps'] = values.pop('max_cycles')
        names = ('red_count', 'blue_count', 'target_count')
        legacy = ('N_R', 'N_B', 'K')
        if any(k in values for k in names+legacy):
            roster = {old: values.pop(new, values.pop(old, default))
                      for new, old, default in zip(names, legacy, (4, 4, 2))}
            values['scale'] = roster
        return make_open_score_env(**values)
    if api == "grouping":
        from had_env.grouping.environment import KnownOpponentEnv
        values.setdefault('spatial_dim', 3)
        return KnownOpponentEnv(config=values)
    if api not in ("parallel", "mpe"):
        raise ValueError("api must be parallel, mpe, grouping or open_score")
    effective = EnvConfig.from_values(values)
    scenario_names = ("red_count", "blue_count", "target_count", "red_scouts", "red_disturbers",
                      "blue_scouts", "blue_disturbers", "initialization", "evaluate", "target_region",
                      "task_mode", "target_health", "spatial_dim", "target_initialization", "target_positions")
    runtime_names = ("max_cycles", "continuous", "render_mode", "reward_weights", "seed", "record_events")
    known = set(scenario_names) | set(runtime_names) | {f.name for f in fields(EnvConfig)}
    unknown = set(values)-known
    if unknown:
        raise ValueError(f"Unknown environment configuration: {sorted(unknown)}")
    scenario_values = {k: values[k] for k in scenario_names if k in values}
    scenario_values.update(spatial_dim=effective.spatial_dim, task_mode=effective.task_mode,
                           effective_config=effective)
    scenario = Scenario(**scenario_values)
    env = HADParallelEnv(scenario, **{k: values[k] for k in runtime_names if k in values})
    return MPEEnv(env) if api == "mpe" else env


def parallel_env(**kwargs):
    return make_env(api="parallel", **kwargs)
