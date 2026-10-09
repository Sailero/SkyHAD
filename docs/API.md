# SkyHAD environment API

The public entry points are `had_env.make_env`, `had_env.parallel_env`, and `from make_env import make_env`. One `had-env` distribution contains the native environment, a general grouping extension, and optional `skyhad_workbench` recording/desktop tools. It contains no learning algorithm or algorithm-specific adapter.

## Construction

```python
from had_env import make_env
from had_env.config import EnvConfig

env = make_env(config={"env_agent_type": "UAV_fixedwing", "task_mode": "damage"},
               env_agent_action_type="position", red_count=4, blue_count=4,
               target_count=2, max_cycles=100)
```

`make_env(scenario_name="defense", *, config=None, **kwargs)` accepts a mapping, UTF-8 JSON path, or immutable `EnvConfig`. Explicit keywords override the configuration. `env.effective_config` holds resolved model and geometry values. The available scenario is `defense` (`defense.py` is an accepted alias).

| API | Action submitted by the caller | Step result |
| --- | --- | --- |
| `parallel` | Dictionary for every current red and blue agent | Observations, rewards, terminations, truncations, infos |
| `mpe` | Fixed red-then-blue list | Observation list, reward list, done list, `{"n": infos}` |
| `grouping` | Defender target assignments and reserve | DecisionState, reward, done, info |

`api="open_score"` is unsupported in v4. Algorithm-side adapters belong in the algorithm repository.

| Configuration | Meaning/default |
| --- | --- |
| `red_count`, `blue_count`, `target_count` | Total team sizes and assets; native defaults 4/4/2 |
| `red_scouts`, `blue_scouts`, `red_disturbers`, `blue_disturbers` | Roles within team totals; remaining agents are attackers; blue needs an attacker |
| `env_agent_type` | `particle` (default), `UAV_fixedwing`, `UAV_quadrotor` |
| `env_agent_action_type` | `acceleration` (default), `position`, UAV-only `actuator` |
| `spatial_dim` | Factory default 3; particle can use 2, UAVs require 3 |
| `continuous` | Continuous acceleration intentions instead of discrete directions |
| `task_mode`, `target_health` | `survival` (default) or `damage`; asset health defaults to 2 |
| `max_cycles` | Native sampling horizon, default 100 |
| `world_bounds`, `target_region` | Three `[low, high]` rows in meters |
| `scene_scale` | Default geometry scales: particle 1, fixed-wing .4, quadrotor .16 |
| `red_spawn_annulus`, `blue_spawn_x`, `spawn_altitude` | Defender spawn radii, attacker x interval, aircraft altitude interval |
| `plane_altitude` | Fixed 2D altitude, default `100*scene_scale` |
| `fire_range` | Positive automatic-fire threshold, at most the full-damage radius |
| `initialization`, `evaluate` | Aircraft `random`/`uniform` layout; `evaluate` is retained for compatibility and does not change sampling |
| `target_initialization`, `target_positions` | Random/fixed assets; explicit positions select fixed layout |
| `reward_weights` | Optional four-component survival shaping weights; rejected for damage |
| `render_mode`, `record_events` | None/human/rgb_array and detailed physical events |

Scene scale affects default geometry and interaction distances, not aircraft mass, inertia, wing size, or rotor size. Explicit geometry values are final meters. For Parallel/MPE, `reset(seed=...)` reproduces the complete initial layout and action-space sampling streams; reset without a new seed continues the instance's existing initialization stream.

## Actions

| Model/control | Single-agent action space | Physical meaning |
| --- | --- | --- |
| 3D discrete acceleration | `Discrete(27)` | Zero intention and 26 unit directions |
| 2D particle discrete acceleration | `Discrete(9)` | Zero intention and eight planar directions |
| Continuous acceleration | `Box(-1,1,(spatial_dim,))` | Normalized intention passed to model dynamics/control |
| Fixed-wing actuator | `Box(-1,1,(4,))` | Elevator, aileron, rudder, throttle; +/-25-degree surfaces and throttle `(u+1)/2` |
| Quadrotor actuator | `Box(0,1,(4,))` | Front/right/rear/left rotor thrust fractions |
| Position | `Box(world_low,world_high,(3,))` | Absolute destination; 2D z is fixed |

Position commands move through a controller and never teleport. Fixed-wing aircraft fly through a destination and cannot hover. UAV acceleration intentions are converted by a controller; they are not instantaneous physical accelerations. A zero quadrotor actuator vector means zero thrust. Fixed-wing zero actuators mean zero surfaces and half throttle, not trim.

Parallel requires exactly the current `env.agents` keys and validates the complete living joint action before changing state. In survival, a dead agent leaves `env.agents` after its final transition. In damage, dead slots remain until the episode ends; their supplied actions are ignored and neutralized.

MPE uses fixed roster slots and accepts discrete indices or explicit one-hot vectors of the correct length. It flattens Dict observations using Gymnasium. Its `done` combines termination and truncation; the original flags remain in `info["n"]`. An inactive survival slot returns zero observations/reward after its final transition.

## Observations and critic features

Let A be the aircraft count, K the asset count, and E=A+K. Entity rows have 11 columns:

```text
relative_position[3], relative_velocity[3], health, alive, is_red, is_blue, is_target
```

Rows exclude self and otherwise preserve fixed entity order. Dead rows are zero. Positions are normalized by the world diagonal; velocities by twice the model preset's maximum speed: 240 m/s for particle, 60 m/s for fixed-wing, and 40 m/s for quadrotor. Rows are not hidden by Scout field of view, and they do not encode the Attack/Scout/Disturb role. `info["observation_entities"]` identifies row order.

| Model/control | Local observation | `state()` length |
| --- | --- | --- |
| Particle acceleration | Array `(E-1,11)` | 11E |
| Particle position | Dict: entities plus self_state `(6,)` | 11E |
| UAV, any control | Dict: entities plus self_state `(13,)` | 11E+13A+3 |

Particle self-state is metric position/velocity. UAV self-state is `[position3, world_velocity3, quaternion4, body_angular_velocity3]`; the unit quaternion is scalar-first and rotates body coordinates into world coordinates. Dead damage observers receive zero local observations.

The UAV critic feature vector appends each aircraft's rigid state and `[cycle, cumulative_target_damage, is_damage]`; dead rigid blocks are zero. These are the existing feature layouts, not a promise that `state()` exposes every variable of the mathematical Markov state. Particle critic features omit time/damage accumulators; external opponent memory is never automatically exposed. See the mathematical formulation for the full state.

## Rewards and episode endings

| Behavior | Survival | Damage |
| --- | --- | --- |
| Asset health | Decreases to destruction | Remains at initial finite health |
| Natural ending | Any asset health below 1e-3, or all blue attackers destroyed | All blue attackers health <=0 |
| Red team destroyed | Does not independently end the episode | Does not independently end the episode |
| Native red reward | `10*outcome_red` by default | Minus newly accumulated raw asset damage |
| `outcome_red` | Asset destroyed -1, blue attackers eliminated +1, otherwise 0 | 0 |

A physical step resolves synchronous collisions and damage using pre-step information. Attack-role agents fire automatically under the existing range rule and die on firing: red attacks blue Attack agents, blue attacks protected assets. Scout and Disturb roles retain their existing sensing/reward and sector-damage mechanisms. Automatic firing does not generate navigation actions.

Raw asset damage is measured before health clamping, counts simultaneous sources, and is not capped by the asset's health. `step_target_damage` is the current increment; `target_damage` is the cumulative sum. Damage episode returns are Red=-D, Blue=D, without discounting.

Every agent receives the full team reward. The optional survival `LatentReward=[hit,target,enemy,episode]` can be weighted explicitly; its last component already includes the terminal reward. Damage uses the raw team objective and rejects shaping weights.

Native `max_cycles` is a sampling truncation, not an automatic defender victory. A natural ending at the same step takes priority.

| Important info | Use |
| --- | --- |
| `agent_mask`, `alive` | Participation in the returned observation and next action |
| `entity_mask`, `action_mask` | Valid entity rows and discrete actions |
| `global_terminated`, `global_truncated` | Team ending versus sampling horizon |
| `bootstrap_mask` | `float(not global_terminated)`; death or sampling truncation need not zero the team critic |
| `cycle`, `sim_time`, `remaining_cycles` | Physical clock and sampling budget |
| `termination_reason` | target_destroyed / blue_attackers_destroyed / None |
| `events`, `effective_config` | Physical diagnostics and resolved geometry/model |

For defender MARL, actor losses use the participation mask from before the action. Team critics use global ending information; a death does not itself end the defender objective. Use the final actual state for time-limit bootstrapping.

## Grouping extension

`make_env(api="grouping", ...)` or `KnownOpponentEnv` provides upper-level assignments. Direct KnownOpponentEnv defaults to particle 2D (UAV 3D), 8 red/8 blue/2 assets, seed 0, horizon 100, command interval 5; the common factory defaults to 3D.

Grouping `reset()` reseeds from its saved seed on every call; `reset(seed=...)` updates that saved seed.

`Grouping(groups, reserve)` contains `Group(target, members)`: target is a stable local asset index, and members are global aircraft IDs. Every surviving defender must appear exactly once, in a nonempty group or reserve. The default group-size cap is None. Casualties prune existing relationships without automatically reassigning survivors.

The blue upper rule is reactive, balanced, or concentrated, and the lower navigation rule is rush. Defender flight actions come from RuleExecutor. Only acceleration intentions are supported in this interface; position and actuator learning use the native interfaces with an external blue actor.

A macro step advances at least one physical step and returns at an absolute command tick, a nonterminal casualty, or episode end. If no defender survives while the episode remains active, it continues autonomously to episode end instead of returning at casualty or command boundaries; blue commands still refresh at their usual opportunities, and `info["no_red_continuation"]` is True. `info["delta"]` records elapsed steps. Damage macro reward sums negative step damage without intra-macro discounting. Survival pays terminal defender success 1, otherwise 0, and its `horizon_policy=red_win/draw/blue_win` is a grouping convention, distinct from native sampling truncation. A discounting learner should account explicitly for macro duration.

In-memory `snapshot()` and `restore()` support independent continuation of the grouping environment under identical configuration. Portable recording, replay and exact branching belong to the optional `skyhad_workbench` tools in this same distribution. Exact continuation requires compatible scientific behavior and configuration; historical recordings remain viewable without granting cross-version continuation. See the [workbench commands](../README.md#optional-research-workbench).
