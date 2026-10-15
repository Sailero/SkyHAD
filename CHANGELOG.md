# Changelog

## 4.3.0 — 2026-10-15

SkyHAD is one package again. The separate `core` branch and its generation and publishing tools are retired; 3D rendering and video stay optional extras (`.[viewer3d]`, `.[record]`), and `pip install -e .` remains headless.

### Physics corrections (simulated results change)

- **Fixed-wing rudder sign.** The guidance law fed sideslip back with the wrong sign for this airframe (`Cn_dr < 0`), cancelling about three quarters of its directional stability. During a sustained 120° turn the peak sideslip falls from 2.5° to 0.7°. The feedforward now uses the trim sideslip instead of the trim heading.
- **Disturb role.** The jamming Gaussian used distance in metres with a 1 m width, so it never did measurable damage (about 1e-88 HP at 20 m). Distance is now measured in sector radii: an active disturber removes 0.024–0.040 HP per second from each UAV in its sector.
- **3D resets.** Red positions outside the world were clipped onto the walls, so 45% of defenders started on a boundary. They are now resampled, as in 2D, and every model applies the minimum spawn spacing.
- **Boundary shaping.** The Scout/Disturb wall penalty now decays across its 100 m band, `0.05·exp(-d/100λ)`, instead of `exp(-d)` with `d` in metres.
- **Degenerate worlds.** World bounds with a zero-extent axis are rejected; they produced non-finite normalized observations.
- `PHYSICS_PROTOCOL` is now `skyhad-4.3`, so grouping snapshots from earlier versions are rejected.

Unchanged: particle dynamics, combat rules, observation and state layouts, rewards, the quadrotor model and the fixed-wing aerodynamic model. Their parameters were re-checked against the BYU MAGICC Aerosonde file and Lee, Leok and McClamroch (arXiv:1003.2005v2, Sec. VII). Two of the three physics golden trajectories and all four grouping golden episodes are byte-identical; the mixed-role 3D trajectory was regenerated for the corrected reset and disturber.

### Performance

- All aircraft are integrated in one vectorized RK4 loop and swept collisions are computed for all pairs at once. A 4 vs 4 step is about 5.6× faster (fixed wing 2 → 12 steps/s, quadrotor 3 → 16 steps/s); trajectories match the previous per-aircraft integration to 1e-14.

### Rendering

- 2D: labels match agent names (`R0`, `B0`, `T0` instead of global entity IDs); aircraft outlines are projected with the full attitude, so bank and pitch are visible without the old R/P indicator ticks; 20 s trails; target fire-range rings; quadrotors drawn in the plus configuration used by the dynamics.
- 3D: larger symbols, ground grid, altitude drop lines, team-colored trails, target fire-range rings, a status line with task, time, alive counts and damage, and labels matching agent names.
- A missing optional package now raises an error with the matching `pip install` command.

### Cleanup

- Removed the dual-edition tooling (`tools/`, `docs/EDITIONS.md`, edition tests and CI jobs) and the internal validation log.
- The grouping adapter is now `GroupingAdapter`, without unused paths (`split_rush`, target-activity toggles, an unused controller class).
- `config.py` documents every tunable constant with units.

## Earlier releases

| Version | Date | Summary |
| --- | --- | --- |
| 4.2.0 | 2026-10-13 | Environment-only full/core editions with optional 3D view and video capture |
| 4.1.0 | 2026-10-09 | Environment and desktop workbench reunited |
| 4.0.0–4.0.2 | 2026-10-09 | Readable flat core and the 36-page mathematical formulation |
| 3.0.0 | 2026-10-08 | Original complete release with the desktop workbench |

Every earlier version remains available as a Git tag.
