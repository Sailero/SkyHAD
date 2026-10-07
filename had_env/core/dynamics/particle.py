"""Particle model compatibility boundary.

The historical HAD particle equations continue through
``had_env.core.agents.base.BaseAgent``. Selecting particle dynamics does not
route legacy trajectories through the new aircraft rigid-body integrator.
"""
