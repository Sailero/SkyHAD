# SkyHAD core 4.2.0

The `core` branch contains the SkyHAD environment and its minimal 2D renderer. It shares the same physics, actions, observations, tasks and Parallel/MPE/grouping APIs as the full edition on `main`. Core omits native 3D, video recording, Qt/VisPy/ImageIO dependencies and full-only render examples. Neither edition contains a desktop workbench or desktop commands.

Both editions use `had-env` and `had_env`, version 4.2.0. Install them in separate environments.

```bash
git clone --branch core https://github.com/Sailero/SkyHAD.git SkyHAD-core
cd SkyHAD-core
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[render,test]"
python examples/quickstart.py
python examples/rendering.py
python -m pytest -q
```

Use Python 3.10-3.12. Base dependencies are NumPy, SciPy, Gymnasium and PettingZoo; no Torch is required. Install only `.` or `.[test]` for headless RL. The optional `render` extra adds Pygame. Core advertises `human` and `rgb_array` render modes and uses 2D projections by default.

```python
from had_env import make_env

env = make_env(env_agent_type="UAV_quadrotor", task_mode="damage",
               max_cycles=100, render_mode="human")
env.set_view_mode("projections")
try:
    observations, infos = env.reset(seed=42)
    while env.agents:
        env.step({agent: env.action_space(agent).sample() for agent in env.agents})
finally:
    env.close()
```

Use `render_mode="rgb_array"` and `env.render()` for a 2D RGB array, or `render_mode=None` for headless stepping. Select full main in its own environment for `env.set_view_mode("3d")` and code-level video recording.

There are sixteen model/action/task themes. Particle supports 2D/3D physical motion; both are shown with 2D projections in this edition. [API](docs/API.md), [modeling](docs/MODELING.md), [benchmark](docs/BENCHMARK.md) and the mathematical [PDF](docs/PROBLEM_FORMULATION.pdf) retain the scientific description. `examples/grouping.py` and `examples/defender_evaluation.py` provide environment-only examples and metrics.

`edition-manifest.json` is also installed as `had_env/edition_manifest.json`. It records the full source SHA, generator version, shared engine hashes and selected file hashes. Both branches share the exact same conditional CI workflow; core uses base/test dependencies and skips full-only Qt/GL/generation jobs. Releases pair `v4.2.0-full` with `v4.2.0-core`; older tags remain available. See [edition selection](docs/EDITIONS.md) and [validation](docs/VALIDATION.md). The known zero-span observation normalization issue remains shared.
