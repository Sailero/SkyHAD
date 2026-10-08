# SkyHAD v4 validation

The v4 refactor changes repository organization and removes desktop and algorithm-specific adapters. It preserves the native motion, interaction, observation, reward and termination rules of v3.0.0. The immutable complete release is still available at `v3.0.0`.

## Reproduce the retained checks

From a fresh Python 3.10+ environment, at the repository root:

```bash
pip install -e ".[test]"
python -m pytest -q
python examples/quickstart.py
python examples/grouping.py
```

The base installation has no Pygame or Qt requirement. Render checks skip explicitly if Pygame is absent. To include basic rendering:

```bash
pip install -e ".[test,render]"
python -m pytest -q
```

The render tests set dummy SDL drivers themselves and exercise real RGB frames. No desktop workbench is needed. Tests are organized into five subjects: API, dynamics, combat/tasks, grouping, and optional rendering.

## Behavior preservation

Before changing the core, a separate v3.0.0 process captured **32 trajectories**: all 16 valid model/control/task combinations, each with seeds 17 and 42. Each episode used 3 defenders, 3 attackers, 2 assets, one red Scout, one blue Disturb, event recording, and a 24-step maximum. Seeded action-space samples supplied the same joint actions to both implementations.

The comparison passed for reset and every recorded step: actions, observations, centralized features, positions, velocities, rigid states, health, rewards, current/cumulative asset damage, physical events, termination and truncation. Numeric tolerance was `rtol=0, atol=1e-10`; discrete fields matched exactly. The one-off capture and comparison artifacts are retained outside the source distribution, rather than adding a second test framework to this repository.

The portable regression data in `tests/data/` provide ongoing checks. The current physical cases and all four grouping episode expectations were copied unchanged from the pre-refactor repository. Only unused historical records and metadata were removed. Expected trajectories were not regenerated from the refactored implementation.

## Recorded acceptance

The following checks ran on 2026-10-08 and 2026-10-09:

| Check | Result |
| --- | --- |
| Complete pre-refactor release | 182 passed; 2 optional Torch tests skipped |
| Retained environment suite, with Pygame | 102 passed |
| Fresh base-only runtime, without Pygame/Qt | 95 passed; 1 render module skipped |
| Paired v3/v4 seeded trajectories | 32/32 matched |
| Installed wheel: official PettingZoo Parallel API test | 16/16 themes passed |
| Installed wheel: native/MPE state, reward and done parity | 16/16 themes passed |
| Installed wheel: grouping episodes | 6/6 model/task pairs passed |
| Installed wheel: basic RGB rendering and packaged icons | 3/3 models passed; state and RNG unchanged |
| Mathematical PDF | 33 pages; 16 exact two-page themes; all pages rendered and inspected |
| Independent workbench v1.0.0, pinned to v3.0.0 | 66 passed; 2 optional Torch tests skipped |

The complete environment suite exposes upstream PettingZoo and Pygame deprecation warnings. The base-only run has no GUI dependency and imports no Pygame, Qt or Torch. The package was installed into a separate runtime and exercised outside the source checkout; source-only imports were not used as evidence of wheel installation.

Workbench checks include real offscreen Qt event loops and owned workers, recording, replay, exact saved branching, legacy v3 continuation, and PNG/SVG/PDF/MP4 export. Its independent runtime contains the original v3 environment wheel, so changing the main environment checkout does not change workbench behavior. Workbench tests and launcher are maintained in [SkyHAD-Workbench](https://github.com/Sailero/SkyHAD-Workbench).

These checks establish implementation consistency and interface behavior. They do not report trained-policy performance or aircraft parameter identification. The simulator's flight and boundary assumptions are stated in [MODELING.md](MODELING.md).

## Recovery checkpoints

| Tag | Saved stage |
| --- | --- |
| `v3.0.0` | Complete original environment and workbench |
| `baseline-v4-20261008` | Save before this simplification |
| `v4.0.0a1` | Desktop workbench extracted |
| `v4.0.0a2` | Open_Score-specific adapter removed |
| `v4.0.0a3` | Readable core and focused tests |
| `v4.0.0b1` | Environment-first README, API and packaging |
| `v4.0.0rc1` | English mathematical formulation and editable source |
| `v4.0.0` | Final integrated release |

The independent tool repository has its own `v1.0.0` tag. To inspect a prior environment version without changing your working checkout, use `git worktree add <new-directory> <tag>` and install it in a separate Python environment. Tags and normal Git history preserve all stages; no history was rewritten.
