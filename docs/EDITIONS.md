# SkyHAD 4.2 editions

Both editions contain the RL environment and minimal 2D projections. Full `main` additionally contains native 3D visualization and code-level video recording. There are no desktop workbench, playback/branch commands or analysis panels in either edition. Environment benchmark metrics remain available in both.

| Capability | main / full | core |
| --- | --- | --- |
| Same sixteen model/control/task themes and physics | Yes | Yes |
| Parallel, MPE, grouping and benchmark metrics | Yes | Yes |
| Headless base install | Yes | Yes |
| Minimal 2D renderer and target icons | Optional Pygame | Optional Pygame |
| Native 3D rendering | Optional PySide6 + VisPy | Omitted |
| VideoRecorder from code | Optional ImageIO + FFmpeg | Omitted |
| Desktop workbench / desktop entry points | Omitted | Omitted |

Both editions are version **4.2.0**, distribution **had-env**, import **had_env**. Installing one replaces the other under the same distribution name, so keep separate virtual environments. No Torch is required.

```bash
git clone --branch main https://github.com/Sailero/SkyHAD.git SkyHAD-full
cd SkyHAD-full
python -m venv .venv
# Activate before installing.
python -m pip install -e ".[render,viewer3d,record,test]"
```

```bash
git clone --branch core https://github.com/Sailero/SkyHAD.git SkyHAD-core
cd SkyHAD-core
python -m venv .venv
# Activate this separate environment before installing.
python -m pip install -e ".[render,test]"
python examples/rendering.py
```

Activate with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in Windows PowerShell. Install only `.` for headless use. The base dependencies are NumPy, SciPy, Gymnasium and PettingZoo. Core has only the optional `render` and `test` extras. Full adds `viewer3d` and `record`; the old `viewer` extra and desktop scripts have been removed.

Ubuntu full 3D sessions also need the [Qt XCB runtime libraries](https://doc.qt.io/qt-6/linux-requirements.html) and a usable OpenGL context:

```bash
sudo apt-get install -y libegl1 libxcb-cursor0 libxkbcommon-x11-0 libxcb-icccm4 libxcb-keysyms1 libxcb-shape0 libxcb-glx0
```

CI adds Xvfb and software Mesa, checks the installed Qt plugin's shared-library closure, and probes QApplication/OpenGL before its required real-GL tests. Core needs none of these Qt prerequisites.

## Rendering from the environment

Create `make_env(render_mode="human")`; the normal environment lifecycle renders the default 2D projections. Both editions accept `env.set_view_mode("projections")`. Only full accepts `env.set_view_mode("3d")`, and missing full-only modules/dependencies produce actionable guidance. `render_mode="rgb_array"` returns the selected view through `env.render()`. Switching view mode never advances physics or changes the action/observation contract.

Full recording uses `had_env.render.video.VideoRecorder(env, path, fps=...)`, its context manager, `capture()` and `close()`; see main's `examples/rendering.py`. Core has a dedicated minimal 2D rendering example. Each capture is one frame; FPS determines playback speed and is not a physical real-time claim.

## Generation and provenance

Core is generated from an exact full commit. `tools/core_files.json` explicitly selects the shared engine, minimal 2D renderer and target icons, retained science documents/tests/fixtures, and environment examples. It omits `scene3d.py`, `native3d.py`, `video.py` and full-only examples/tests. New shared engine modules must be reviewed in the allowlist. Generation copies shared code directly and never strips mixed engine code.

```bash
python tools/generate_core.py --source . --ref <40-character-full-commit-SHA> --output ../SkyHAD-core-generated
python tools/verify_editions.py --source . --ref <40-character-full-commit-SHA> --output ../SkyHAD-edition-validation
```

Output must be new/empty and outside the source checkout; existing files are never deleted. For development only, use explicit `--working-tree` instead of `--ref`. Such exports record their HEAD and `source_mode: working-tree`, so they cannot be mistaken for immutable releases. The timestamp-free `edition-manifest.json`, also installed as `had_env/edition_manifest.json`, records source SHA, generator version, shared engine hashes and all selected input hashes. Repeated generation is byte-identical for identical inputs. The shared engine and copied 2D code remain byte-identical between editions.

## Shared CI and publishing

Main and core use the exact same `.github/workflows/tests.yml` path and contents. Edition detection requires no workbench and a 2D backend in either edition; core must additionally have its matching manifests and omit 3D/video modules. Core runs the Ubuntu/Windows Python 3.10/3.12 matrix with base/test dependencies. Full adds optional 2D/3D/video test dependencies, usable Linux OpenGL and installed-edition validation. Optional Pygame checks skip in the base-only core matrix; main's validator also checks installed core 2D rendering separately.

The installed-wheel/96-cell parity integration runs once in the dedicated editions job. The four OS/Python matrix cells run all other tests; local default `pytest` still includes the integration test. Publication requires the dedicated job and every matrix cell to pass.

The core publisher requires all full validation jobs and canonical main push detection. It checks artifact hashes and canonical workflow bytes against the exact tested full Git commit before any Git mutation, verifies main is still current, serializes updates and uses ordinary fast-forward-only pushes. Generated core commits are not merged back into main.

The design relies on GitHub's documented exception allowing a workflow file to be committed without the workflow scope when its exact path and contents already exist on another branch ([GitHub workflow scope documentation](https://docs.github.com/en/apps/oauth-apps/building-oauth-apps/scopes-for-oauth-apps#available-scopes)). No additional stored token is configured. Future workflow changes are copied from their already validated main commit. Default-token pushes do not trigger another push workflow, so generated core is validated before publication; the shared workflow still covers direct core maintenance pushes/PRs. Initial publication remains a release step after the CI gate and user authorization.

Releases pair `v4.2.0-full` with `v4.2.0-core`. Existing historical tags are preserved and tags are published only after validation.

## Retained scientific behavior

Physics, controllers, collision/fire/damage, observations, rewards, termination, seeded RNG streams and all environment interfaces are shared. Renderer selection does not improve a policy's scientific result. The known zero-span observation normalization issue can produce nonfinite normalized values for degenerate configured spans and remains identical in both editions pending a separately validated scientific/API change.
