# SkyHAD

SkyHAD is an aerial attack-defense environment for multi-agent reinforcement learning. The red team defends stationary assets against the blue team. Choose particle, six-degree-of-freedom fixed-wing, or quadrotor dynamics; the collision, automatic attack, and task rules are shared.

The environment contains no learning algorithm. Its primary interface is PettingZoo Parallel; a fixed-list MPE adapter and a grouping interface are also available.

## Install and run

Python 3.10 or newer:

```bash
pip install -e .
python examples/quickstart.py
```

```python
from had_env import make_env

env = make_env(
    env_agent_type="UAV_quadrotor",
    env_agent_action_type="acceleration",
    task_mode="damage",
    red_count=4, blue_count=4, target_count=2,
    max_cycles=100,
)
observations, infos = env.reset(seed=42)
try:
    while env.agents:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, terminated, truncated, infos = env.step(actions)
finally:
    env.close()
```

This example samples both teams' actions. For defender training, replace the red actions with your decentralized policies and the blue actions with an explicitly supplied opponent. Automatic fire is part of the simulator; opponent navigation is not built into Parallel/MPE.

## Choose a task

| Setting | Options |
| --- | --- |
| `env_agent_type` | `particle`, `UAV_fixedwing`, `UAV_quadrotor` |
| `env_agent_action_type` | `acceleration`, `position`; UAVs also support `actuator` |
| `task_mode` | `survival`, `damage` |
| `api` | `parallel` (default), `mpe`, `grouping` |
| `spatial_dim` | 3 (factory default); particle also supports 2 |

These choices define **16 model/action/task combinations**. Position actions specify a destination for a controller; they do not teleport the aircraft. Acceleration actions for UAVs pass through a flight controller. Actuator actions control the physical inputs directly.

- **Survival:** defend every asset. An asset's destruction is a defender loss; eliminating all blue attackers is a defender win. The default native team reward is +10/-10 at the corresponding terminal transition.
- **Damage:** minimize accumulated raw damage to assets. Asset health stays fixed for accounting, while every hit adds damage. The red team receives minus the new damage each step; blue receives the opposite reward.

Each agent receives its team's full reward. Do not sum identical rewards over teammates to obtain the team score. Native `max_cycles` is a sampling truncation and does not award a win.

See [API details](docs/API.md) for observations, action units, termination masks, custom geometry, and initialization.

## Grouping and hierarchical decisions

```python
from had_env import make_env
from had_env.grouping.policies import RulePolicy

env = make_env(api="grouping", red=8, blue=8, targets=2,
               task_mode="damage", max_steps=100)
policy = RulePolicy("rule")  # Replace with your upper-level grouping policy.
try:
    state = env.reset(seed=42)
    while not env.done:
        state, reward, done, info = env.step(policy.act(state))
finally:
    env.close()
```

The action assigns surviving defenders to protected targets or a reserve. A lower-level executor produces their flight actions. This interface has built-in blue assignment and rush rules and returns at decision boundaries; `info["delta"]` is the number of physical steps advanced. It supports all three models with acceleration control. Grouping survival has its own terminal reward and horizon convention, documented in the API.

## Rendering and development

```bash
pip install -e ".[render]"  # Basic env.render(), without a desktop workbench.
pip install -e ".[test]"
python -m pytest -q
```

Set `render_mode="human"` for a window or `"rgb_array"` for an image. Rendering dependencies are optional; headless training uses only the base installation. Render tests skip when Pygame is absent.

A local `.venv/` is your Python installation, is ignored by Git, and is not part of the repository distribution.

## Read the environment

Start with the [quickstart](examples/quickstart.py), then follow `reset()` and `step()` through the environment interface, simulation, dynamics and interaction, task scoring, and observation construction. The `grouping/` package is a separate extension; reading the native environment does not require it.

```text
had_env/
  environment.py     # Parallel RL interface
  simulation.py      # Episode lifecycle and command conversion
  world.py           # Motion, collision and simultaneous interaction
  initialization.py  # Initial layouts
  observations.py    # Actor observations and critic features
  tasks.py           # Rewards and episode endings
  config.py          # Parameters and model presets
  agents/            # Attack, Scout and Disturb roles
  dynamics/          # Six-DOF models and flight controllers
  grouping/          # General hierarchical decisions and rule execution
  render/            # Optional basic rendering
examples/             # Minimal interaction examples
tests/                # Focused behavioral checks
docs/                 # API and mathematical formulation
```

The native route is `make_env -> HADParallelEnv -> Simulation -> World.step`, followed by task scoring and observation construction. `env.simulation` is the physical simulator; `env.simulation.entities` is its fixed-order entity list.

<details>
<summary>Why each shipped file exists</summary>

| File | Responsibility |
| --- | --- |
| `.gitattributes` | Normalize tracked text line endings across platforms. |
| `.gitignore` | Exclude virtual environments, caches, build output and experiment data. |
| `README.md` | Installation, minimal interaction, task selection and the code reading route. |
| `pyproject.toml` | Build metadata, dependency groups and pytest discovery. |
| `make_env.py` | Preserve the short public import used by existing environment consumers. |
| `docs/API.md` | Environment contracts, units, observations, rewards and hierarchical interface. |
| `docs/MODELING.md` | Defender task definitions and the mathematical model behind the simulator. |
| `docs/PROBLEM_FORMULATION.pdf` | The 36-page IEEE-style defender formulation of all 16 themes. |
| `docs/PROBLEM_FORMULATION.tex` | Editable mathematical source for the formulation PDF. |
| `docs/VALIDATION.md` | Test commands, behavior comparisons and release checkpoints. |
| `examples/quickstart.py` | A complete minimal Parallel reset/action/step/close loop. |
| `examples/grouping.py` | A minimal upper-level grouping interaction loop. |
| `had_env/__init__.py` | Public factory exports and software version. |
| `had_env/actions.py` | Stable 3D and planar discrete acceleration direction tables. |
| `had_env/config.py` | One location for immutable instance configuration, presets and physical defaults. |
| `had_env/factory.py` | Resolve configuration and select Parallel, MPE or grouping. |
| `had_env/scenario.py` | Validate roster/role choices and construct the physical simulation. |
| `had_env/environment.py` | Parallel spaces, joint-action validation, masks and lifecycle. |
| `had_env/mpe.py` | Adapt Parallel to fixed-order observations/actions and done lists. |
| `had_env/simulation.py` | Coordinate reset, command conversion, physical stepping and rendering. |
| `had_env/world.py` | Predict synchronized motion, resolve collisions, trigger fire and apply simultaneous damage. |
| `had_env/initialization.py` | Sample or construct aircraft and asset layouts with the instance RNG. |
| `had_env/observations.py` | Build local entity rows, participation masks and centralized features. |
| `had_env/tasks.py` | Define damage accounting, rewards and natural episode endings. |
| `had_env/geometry.py` | Shared distance, sector, damage falloff and synchronous collision mathematics. |
| `had_env/agents/__init__.py` | Declare the entity-role package. |
| `had_env/agents/base.py` | Entity health, particle motion, boundaries and rigid-state synchronization. |
| `had_env/agents/attack.py` | Attack-role behavior and survival auxiliary rewards. |
| `had_env/agents/scout.py` | Scout sector behavior and auxiliary rewards. |
| `had_env/agents/disturb.py` | Disturb sector behavior and auxiliary rewards. |
| `had_env/dynamics/__init__.py` | Select the rigid-body dynamics model. |
| `had_env/dynamics/control.py` | Quaternion rotations and shared flight-control mathematics. |
| `had_env/dynamics/rigid_body.py` | Newton-Euler evolution and RK4 substep integration. |
| `had_env/dynamics/fixedwing.py` | Fixed-wing aerodynamics, propeller, trim and flight controllers. |
| `had_env/dynamics/quadrotor.py` | Rotor force/moment mapping and quadrotor flight controllers. |
| `had_env/grouping/__init__.py` | Expose the general grouping action and decision-state types. |
| `had_env/grouping/domain.py` | Stable entity identities and validated target-group assignments. |
| `had_env/grouping/environment.py` | Advance upper-level actions to periodic or casualty decision boundaries. |
| `had_env/grouping/adapter.py` | Connect group assignments, low-level controls and the physical simulation. |
| `had_env/grouping/actions.py` | Construct and decode grouping actions. |
| `had_env/grouping/opponents.py` | Reactive, balanced and concentrated blue target-assignment rules. |
| `had_env/grouping/policies.py` | Simple example upper-level policies. |
| `had_env/grouping/rules.py` | Execute defender group assignments as low-level flight intentions. |
| `had_env/render/__init__.py` | Declare optional rendering without importing Pygame in headless mode. |
| `had_env/render/render.py` | Draw the native environment and return RGB frames. |
| `had_env/render/glyphs.py` | Aircraft/rotor vector shapes and attitude-aware icon geometry. |
| `had_env/resources/target.png` | Asset icon for native rendering. |
| `had_env/resources/target_dead.png` | Destroyed-asset icon for native rendering. |
| `tests/test_api.py` | Verify interface, configuration, observation, seeding and action-validation contracts. |
| `tests/test_dynamics.py` | Verify physical forces, control modes, rigid states and integration. |
| `tests/test_combat.py` | Verify collision, simultaneous damage, task rewards and episode endings. |
| `tests/test_grouping.py` | Verify grouping execution, seeded episodes and independent snapshot continuation. |
| `tests/test_render.py` | Verify optional rendering lifecycle and image output. |
| `tests/data/physics_reference.json` | Independent pre-refactor physical trajectory expectations. |
| `tests/data/grouping_extraction_validation.json` | Independent pre-refactor grouping episode expectations. |

</details>

## Mathematical formulation

[MODELING.md](docs/MODELING.md) explains the two tasks and the defender's Dec-POMDP. [PROBLEM_FORMULATION.pdf](docs/PROBLEM_FORMULATION.pdf) contains 36 pages in IEEE journal style: four pages of shared formulation and an index, then two pages for each of the 16 themes. The normal body text is 10pt; each theme references shared definitions and gives its model, control, observation and defender objective. Its [LaTeX source](docs/PROBLEM_FORMULATION.tex) is editable for a paper's Problem Formulation chapter. The formulation distinguishes the mathematical Markov state from the packed critic features returned by `state()`.

Reproducible checks and their results are recorded in [VALIDATION.md](docs/VALIDATION.md).

## Versions and tools

SkyHAD **v4.0.1** focuses on the environment and general hierarchical decisions. This patch revises the mathematical documentation and preserves v4.0.0 environment behavior. The Python distribution remains `had-env`; imports remain `had_env` and the root `make_env` entry point.

The full desktop viewer, recording, replay, exact branching, and export tools live in [SkyHAD-Workbench](https://github.com/Sailero/SkyHAD-Workbench). Its v1.0.0 release is pinned to SkyHAD v3.0.0. The environment v4 package has no desktop CLI or Qt dependency.

Open_Score-specific adapters have been removed. An algorithm project can implement its own adapter against SkyHAD's public interface.

The original complete release remains available at **v3.0.0**. The refactoring baseline is **baseline-v4-20261008**; stage tags preserve workbench extraction, algorithm-adapter removal, core simplification, documentation, and final validation.
