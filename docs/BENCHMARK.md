# Native defender evaluation

`examples/defender_evaluation.py` evaluates a Red callable through PettingZoo Parallel against one frozen deterministic Blue rule, `nearest-target-velocity-v1`. Blue pursues its nearest live asset; equal distances use fixed observation row order. Both teams act from the same pre-step observations, and only Red is replaced. Automatic firing, collisions, rewards and episode endings remain native.

```bash
python examples/defender_evaluation.py --model particle --task survival --seeds 11 12 13 --horizon 100 --red-policy intercept --output outputs/intercept.json
python examples/defender_evaluation.py --model particle --task survival --seeds 11 12 13 --horizon 100 --red-policy coast --output outputs/coast.json
```

The demonstration supports all three models (`particle`, `UAV_fixedwing`, `UAV_quadrotor`) with either `survival` or `damage`: **six acceleration cells**, using continuous normalized intentions, 4/4 Attack-only rosters, 2 assets and no reward shaping. `--spatial-dim 2` additionally supports planar particle motion; UAVs require 3D. These example choices are separate from the **16 native model/action/task combinations**, which also include position and UAV actuator controls. See the [native action table](API.md#actions).

Navigation converts public normalized relative rows back to meters and m/s, requests the model preset's reference velocity toward the destination, then bounds the velocity error by the preset acceleration limit. Particle own velocity comes from a stationary asset's relative velocity; UAVs supply metric self-state. Existing UAV controllers convert the intention to physical actuators. Fixed-wing aircraft cannot hover or instantly reverse. The Red `intercept` example follows the nearest Blue contact's two-second predicted position; `coast` submits zero acceleration and does not stop the aircraft.

To use a learned defender, import this example and replace its Red callable:

```python
from examples.defender_evaluation import run_episode

# Reset your policy memory/RNG before each episode. Return a float32 vector
# of length context.spatial_dim in [-1,1], using observation and public info.
def my_red(observation, info, *, context):
    return trained_actor(observation, info["agent_mask"])

row = run_episode({"env_agent_type": "UAV_quadrotor", "task_mode": "damage"},
                  seed=11, red_policy=my_red)
```

`context` is the immutable resolved `EnvConfig`. Policies must not mutate their inputs. Dead Damage slots have `agent_mask=0`; their actions are neutralized by the environment. Red receives no pending Blue action. Frozen Blue means identical code, parameters, observation contract and randomness; different Red trajectories can lead to different subsequent Blue actions. This example trains nothing and imposes no checkpoint/algorithm interface.

Each JSON result records ordered seeds, episode count, horizon, physical timestep, full resolved geometry/model and roster/runtime configuration, model navigation preset, package/core version, physics protocol, Blue identifier and Red demonstration parameters. Research reports should additionally identify the exact release/commit and external Red checkpoint, training seed and policy randomness.

**Survival:** one full Red team reward is counted per physical step, with the Blue sign reversed if all Red slots have retired. Individual casualties do not end evaluation. Results separate native wins (`outcome_red=1`), losses (`-1`) and sampling truncations; `win_fraction_all_episodes` and `truncation_fraction_all_episodes` both use **all episodes** as their denominator. Natural termination at the horizon takes priority. A horizon alone never awards a win.

**Damage:** report raw cumulative `target_damage`, its per-target counters and Red return `-target_damage`, with natural endings and truncations separate. Simultaneous raw damage can exceed asset health. Damage is not a win rate, and its sampling horizon affects exposure. Never sum replicated team rewards or counters across teammates, or combine the two tasks' scores.

For fair comparison, freeze Blue and the evaluated Red checkpoint before testing; use identical ordered seeds, horizon, roster, role counts, asset distribution, geometry, model/control and native rewards for every Red method. Independently seed/reset Red memory and RNG each episode. Keep training/validation openings separate from evaluation, publish every raw episode including truncations, and pair comparisons by seed within the same configuration. Report different models/tasks as separate experiment cells; their preset scale and dynamics differ. The three default seeds demonstrate reproducibility rather than statistical significance.

The workbench already provides explicit paired ID/OOD scenarios and uncertainty across training-seed means for **grouping Survival policies**:

```bash
skyhad-workbench evaluate examples/evaluation.json --policies rule grand --training-seeds 0 --output outputs/evaluation
```

That evaluator accepts grouping policies through `SimulationSession`, not this example's native Red callable, and rejects Damage win-rate summaries. Its command clock and horizon conventions belong to the grouping protocol. Reuse it for that scope rather than relabeling native results as the grouping benchmark.
