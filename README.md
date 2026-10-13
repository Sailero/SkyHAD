# SkyHAD 4.2

SkyHAD is an aerial attack-defense environment for multi-agent reinforcement learning. Red defends stationary assets against Blue. Particle, six-DOF fixed-wing and quadrotor models share collision, automatic attack, observation and task rules. No learning algorithm is included.

Choose **main** for the environment, default 2D projections, optional native 3D and video recording. Choose **core** for the same environment and minimal 2D rendering. Both branches use `had-env` / `had_env`, version 4.2.0, and retain the same physics and Parallel/MPE/grouping APIs. Keep separate virtual environments. See [edition selection](docs/EDITIONS.md).

## Install

Use Python 3.10-3.12. The base installation needs only NumPy, SciPy, Gymnasium and PettingZoo. Rendering dependencies are optional; no Torch is required.

```bash
git clone --branch main https://github.com/Sailero/SkyHAD.git SkyHAD-full
cd SkyHAD-full
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[render,viewer3d,record]"
python examples/quickstart.py
```

`render` adds Pygame for 2D; `viewer3d` adds PySide6 and VisPy; `record` adds ImageIO and its FFmpeg integration. Install only `.` for headless RL, or `.[render]` for 2D. Linux Qt/3D sessions need a usable OpenGL context and Qt's XCB runtime libraries. On Ubuntu:

```bash
sudo apt-get install -y libegl1 libxcb-cursor0 libxkbcommon-x11-0 libxcb-icccm4 libxcb-keysyms1
```

## Environment and rendering

```python
from had_env import make_env

env = make_env(env_agent_type="UAV_fixedwing", task_mode="survival",
               red_count=4, blue_count=4, max_cycles=100, render_mode="human")
# Default: minimal 2D projections. Full main can switch to native 3D:
env.set_view_mode("3d")
try:
    observations, infos = env.reset(seed=42)
    while env.agents:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, terminated, truncated, infos = env.step(actions)
finally:
    env.close()
```

Human mode renders through the environment's normal reset/step lifecycle. `env.set_view_mode("projections")` selects the default projections again. Use `render_mode="rgb_array"` and `env.render()` to obtain an RGB frame; leave `render_mode=None` for headless training. View changes do not change physics, observations, actions or RNG. Core supports 2D only and gives guidance when a full-only view is requested.

## Video from code

The full edition records frames directly from the environment:

```python
from had_env import make_env
from had_env.render.video import VideoRecorder

env = make_env(red_count=4, blue_count=4, max_cycles=100, render_mode="rgb_array")
try:
    env.reset(seed=42)
    with VideoRecorder(env, "outputs/episode.mp4", fps=10) as recorder:
        recorder.capture()
        while env.agents:
            env.step({agent: env.action_space(agent).sample() for agent in env.agents})
            recorder.capture()
finally:
    env.close()
```

Each capture writes one frame. `fps` controls video playback speed; the display target of 30 FPS does not assert physical real-time playback. See `examples/rendering.py` for native rendering/recording usage. There are no desktop workbench, recording/replay command tools or analysis panels.

## Tasks, controls and grouping

| Setting | Options |
| --- | --- |
| `env_agent_type` | `particle`, `UAV_fixedwing`, `UAV_quadrotor` |
| `env_agent_action_type` | `acceleration`, `position`; UAVs also support `actuator` |
| `task_mode` | `survival`, `damage` |
| `api` | `parallel` (default), `mpe`, `grouping` |
| `spatial_dim` | 3 (factory default); particle also supports 2 |

These choices provide sixteen native model/action/task themes. Fixed-wing aircraft cannot hover. Survival gives Red a loss when any defended asset is destroyed and a win when all Blue attackers are eliminated. Damage uses minus the new raw asset damage as Red's team reward. Every teammate receives the same team reward; count it once per step. `max_cycles` is a sampling truncation, not a win.

```python
from had_env import make_env
from had_env.grouping.policies import RulePolicy

env = make_env(api="grouping", red=8, blue=8, targets=2,
               task_mode="damage", max_steps=100)
policy = RulePolicy("rule")
try:
    state = env.reset(seed=42)
    while not env.done:
        state, reward, done, info = env.step(policy.act(state))
finally:
    env.close()
```

Pure environment evaluation is available in both editions:

```bash
python examples/defender_evaluation.py --model UAV_quadrotor --task damage --seeds 11 12 13 --horizon 100
```

Read the [API](docs/API.md), [modeling assumptions](docs/MODELING.md), [benchmark protocol](docs/BENCHMARK.md), mathematical [PDF](docs/PROBLEM_FORMULATION.pdf) and [validation record](docs/VALIDATION.md). The retained zero-span observation normalization issue is documented there.

## Tests and releases

```bash
python -m pip install -e ".[render,viewer3d,record,test]"
python -m pytest -q
```

Base-only tests skip optional render checks. Full CI additionally exercises real Linux OpenGL. Paired releases use `v4.2.0-full` and `v4.2.0-core`; existing historical tags remain available. The core provenance manifest identifies its exact full source commit and shared file hashes.
