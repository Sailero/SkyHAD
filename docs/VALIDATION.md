# SkyHAD v4 validation

The v4.0 refactor organized the readable environment core and separated the desktop tools. v4.1 reunites the complete optional workbench with that single core. Native motion, interaction, observation, reward and termination rules are preserved. The immutable original complete release remains available at `v3.0.0`; each historical acceptance below refers to its recorded version.

## Reproduce the retained checks

From a fresh Python 3.10-3.12 environment, at the repository root:

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

The core tests cover API, dynamics, combat/tasks, grouping and optional rendering. Render tests exercise real RGB frames with dummy SDL drivers. For the complete optional workbench checks, install `.[viewer,test]` and run the same pytest command; they additionally cover live Qt workers, recording/replay, portable and nested branches, evaluation, and figure/video export. Ubuntu Qt tests need the system `libegl1` runtime.

## Behavior preservation

Before changing the core, a separate v3.0.0 process captured **32 trajectories**: all 16 valid model/control/task combinations, each with seeds 17 and 42. Each episode used 3 defenders, 3 attackers, 2 assets, one red Scout, one blue Disturb, event recording, and a 24-step maximum. Seeded action-space samples supplied the same joint actions to both implementations.

The comparison passed for reset and every recorded step: actions, observations, centralized features, positions, velocities, rigid states, health, rewards, current/cumulative asset damage, physical events, termination and truncation. Numeric tolerance was `rtol=0, atol=1e-10`; discrete fields matched exactly. The one-off capture and comparison artifacts are retained outside the source distribution, rather than adding a second test framework to this repository.

The portable regression data in `tests/data/` provide ongoing checks. The current physical cases and all four grouping episode expectations were copied unchanged from the pre-refactor repository. Only unused historical records and metadata were removed. Expected trajectories were not regenerated from the refactored implementation.

## Recorded v4.0.0 acceptance

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

After integration into main, the full suite passed again: **102 passed, 3 upstream deprecation warnings in 15.74 seconds**. The final v4.0.0 wheel contains 37 package/resource files and is **78,295 bytes**, excluding third-party dependencies. Its packaged files match the merged source byte for byte. There are no desktop entry points, Qt/Torch dependencies, old core wrappers or algorithm-specific modules in the wheel.

The v4.0.0 repository shipped **55 files**; its source archive was approximately **0.53 MB**, including the mathematical PDF, editable source and retained tests. Local virtual environments, experiment output, build/cache files and the user-supplied reference article are not distributed. Validation artifacts are archived outside the source tree. All 33 formulation pages were rendered and checked for theme/page correspondence, formula overflow, column crossings and footer clearance; independent code/math review found no outstanding issues.

Workbench checks include real offscreen Qt event loops and owned workers, recording, replay, exact saved branching, legacy v3 continuation, and PNG/SVG/PDF/MP4 export. Its independent runtime contains the original v3 environment wheel, so changing the main environment checkout does not change workbench behavior. Workbench tests and launcher are maintained in [SkyHAD-Workbench](https://github.com/Sailero/SkyHAD-Workbench).

These checks establish implementation consistency and interface behavior. They do not report trained-policy performance or aircraft parameter identification. The simulator's flight and boundary assumptions are stated in [MODELING.md](MODELING.md).

## v4.0.1 documentation revision

The 2026-10-09 patch changes the mathematical source/PDF and their documentation references, plus the package version metadata. All dynamics, controls, interactions, observations, rewards, termination, interfaces and retained test data remain unchanged from v4.0.0. The independent workbench remains v1.0.0.

| Check | Result |
| --- | --- |
| Final PDF rebuilt from editable source | 36 A4 pages, IEEEtran journal layout, 10pt normal body |
| Theme map | Four overview pages; themes 1-16 occupy exactly two pages each, pages 5-36 |
| Artifact checks | All pages rendered and inspected; embedded fonts; resolved equation/page links; no overfull boxes, missing glyphs, cropping or overlaps |
| Mathematical preservation | Model parameters/controllers, S/A/O/T/R, reset laws and sampling semantics independently checked |
| Final retained environment suite with Pygame | 102 passed; 3 upstream deprecation warnings |

The revision separates shared laws from theme-specific instances and clarifies exact value conditioning for recurrent policies. It distinguishes a six-DOF vehicle from its twelve-dimensional state stored in thirteen values, including a unit quaternion. The original 33-page PDF and its validation evidence remain recoverable at v4.0.0. One-off compilation, rendering and review artifacts stay outside the repository.

## v4.0.2 formulation continuity

The 2026-10-09 correction treats each native theme as one paper-style section, with continuous subsection numbering and a single narrative from dynamics and control to defender information, task reward and policy optimization. It revises document structure and prose, plus release metadata. Environment behavior and the independent v1.0.0 workbench remain unchanged.

| Check | Result |
| --- | --- |
| Theme structure | Sixteen single numbered sections V-XX; one theme heading across each two-page section |
| Page map | 36 A4 pages: four overview pages and themes 1-16 on pages 5-36 |
| Continuity | Subsection letters continue across page boundaries; repeated page introductions and checklist narration consolidated |
| Mathematical preservation | Existing display equations, physical parameters, observation/reward definitions and task semantics retained and independently reviewed |
| PDF acceptance | Rebuilt from source; 10pt normal body; all pages rendered and inspected; embedded fonts and resolved references; no overflow or overlap |
| Retained core tests | 102 passed; 3 upstream deprecation warnings |

The v4.0.1 layout and all earlier validation evidence remain recoverable at their existing tags. Compilation and review artifacts stay outside the repository.

## v4.1.0 unified environment and workbench

The 2026-10-09 revision reunites the environment and the complete desktop tools in one distribution. `had_env` remains the single simulator; `skyhad_workbench` supplies optional live controls, recording, playback, saved branches, evaluation and figure/video export. Native flight sessions use the simulator's fixed-roster transition contract directly. Parallel, MPE and grouping interfaces retain their existing behavior. The mathematical TeX/PDF are byte-for-byte unchanged from v4.0.2: 36 pages with sixteen continuous theme sections.

Shared geometry now distinguishes fixed-wing and quadrotor vehicles in the native renderer, Qt viewer and figure export. Aircraft attitude cues use the recorded physical state. The particle geometry and simulation laws remain unchanged. Presentation-only differences do not invalidate exact branches; dynamics, lifecycle, effective configuration and policy changes remain strict compatibility boundaries. Original v3 recordings remain readable; exact continuation requires the original compatible simulator revision rather than bypassing that boundary.

| Check | Result |
| --- | --- |
| Final-version checkout before final portability/configuration regressions | 204 passed; 2 optional Torch tests skipped; 3 existing upstream deprecation warnings |
| CLI/package portability correction | 40 passed; 1 optional Torch skip; all eight help entries exercise strict cp1252 output |
| Portable branch/effective-evaluation correction | 62 passed; 1 optional Torch skip; real configured-health continuation and equivalent/different horizon/altitude cases |
| Fresh final base-only wheel | Installed and exercised outside the checkout with isolated Python; no Qt, Pygame, Matplotlib or Torch installed/imported |
| Wheel packaging | `had_env-4.1.0-py3-none-any.whl`, approximately 141 KiB excluding dependencies; both console aliases and MIT/Saileron license payload verified |
| Paired v4.0.2/v4.1.0 transitions | All 16 native combinations, two seeds each, plus 6 grouping model/task runs: 38/38 JSON payloads byte-identical |
| Comparison coverage | Reset/step actions, actor observations, critic features, entity/rigid states, health, rewards, asset damage, termination and truncation |
| Rendering acceptance | Native/Qt/figure views of all three models inspected, including light/dark, grayscale and dense views; SVG/PDF exports inspected |
| Native defender example | Six acceleration-control model/task cells; deterministic frozen Blue rule, replaceable Red navigation, task-specific metrics and explicit truncation denominator |
| Legacy records | Genuine compressed v3 recording loads; incompatible exact continuation rejects explicitly |

The frozen defender example is an evaluation starting point, not a trained MARL result or coverage claim for all sixteen native combinations. It distinguishes automatic firing from attacker navigation and does not introduce a universal scripted-attacker interface. Optional Torch paths were skipped because Torch is absent; no Torch validation is claimed.

Final review corrected two configuration boundaries without changing simulation laws: portable restoration now uses the same explicit/configured/default target-health precedence as construction, and grouping evaluation compares effective horizon and altitude instead of accepting equivalent spellings as structural OOD. Genuine mismatches remain rejected. Both regressions failed before their fixes; the final scoped review approved both corrections.

The original environment and independent-workbench histories were saved as verified Git bundles outside the repository before migration. Original ignored recordings and images were also copied and checked separately; generated QA output and those backups are not source-distribution content. The main-folder installation, hosted matrix and authorized old-folder removal are recorded below only after those actions complete.

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
| `v4.0.0` | Final integrated release; original 33-page formulation |
| `v4.0.1a1` | Four-page common formulation and sixteen theme instances |
| `v4.0.1rc1` | IEEE-style 36-page formulation and visual acceptance |
| `v4.0.1` | Final IEEE layout revision |
| `v4.0.2rc1` | Continuous theme sections and paper-style prose |
| `v4.0.2` | Verified formulation continuity release |
| `v4.1.0a1` | Complete workbench reunited with the single environment core |
| `v4.1.0a2` | Shared aircraft rendering and visual acceptance |
| `v4.1.0rc1` | Public benchmark example, MIT license, concise documentation and CI |

The independent tool repository has its own `v1.0.0` tag. To inspect a prior environment version without changing your working checkout, use `git worktree add <new-directory> <tag>` and install it in a separate Python environment. Tags and normal Git history preserve all stages; no history was rewritten.
