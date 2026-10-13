"""Optional, single-window 3D adapter for the environment's native renderer.

Importing this module needs no GUI packages. Requesting its window or RGB capture
requires ``had-env[viewer3d]`` and a usable native OpenGL display.
"""
from __future__ import annotations

import os
import sys

import numpy as np

from had_env.config import Interval


def simulation_frame(simulation):
    """Copy current physical truth, preserving native global IDs and raw metres."""
    entities = []
    for entity in simulation.entities:
        side = {'Red': 'red', 'Blue': 'blue', 'Entity': 'targets'}.get(entity.Color, str(entity.Color).lower())
        attitude = getattr(entity, 'attitude', None)
        rigid = getattr(entity, 'rigid_state', None)
        entities.append(dict(
            side=side, id=entity.Id, entity_id=entity.Id, role=entity.Type,
            position=np.asarray(entity.position, dtype=float).tolist(),
            velocity=np.asarray(entity.velocity, dtype=float).tolist(),
            health=float(entity.Health), max_health=float(entity.initial_health),
            alive=bool(entity.Health > 0), env_agent_type=simulation.env_agent_type,
            attitude=None if attitude is None else np.asarray(attitude, dtype=float).tolist(),
            rigid_state=None if rigid is None else np.asarray(rigid, dtype=float).tolist(),
            attack_range=float(simulation.attack_distance[-1]),
            step_damage=float(entity.step_damage), cumulative_damage=float(entity.cumulative_damage),
        ))
    step = int(simulation.physics_step_count)
    metadata = dict(
        step=step, sim_time=step * Interval, world_bounds=simulation.world_bounds.tolist(),
        env_agent_type=simulation.env_agent_type, task_mode=simulation.task_mode,
        live_counts={'red': sum(e.Health > 0 for e in simulation.red_agents),
                     'blue': sum(e.Health > 0 for e in simulation.blue_agents),
                     'targets': sum(e.Health > 0 for e in simulation.targets)},
        step_target_damage=float(simulation.step_target_damage),
        target_damage=float(simulation.target_damage),
    )
    return dict(entities=entities, step=step, sim_time=step * Interval, metadata=metadata)


class Native3DRenderer:
    """One lightweight scene window; no stepping, controls or recording state."""

    def __init__(self, simulation):
        qt_platform = os.environ.get('QT_QPA_PLATFORM', '').split(':', 1)[0].strip().lower()
        if qt_platform in {'offscreen', 'minimal'}:
            raise RuntimeError(
                f'Native 3D OpenGL rendering is unsupported with QT_QPA_PLATFORM={qt_platform}. '
                'Use the 2D projections renderer or a graphics-enabled display.')
        if sys.platform.startswith('linux') and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
            raise RuntimeError(
                'Native 3D OpenGL rendering requires a Linux display (DISPLAY or WAYLAND_DISPLAY). '
                'Use the 2D projections renderer or run with an X/Wayland display.')
        self.simulation = simulation
        self._closed = False
        self._shown = False
        self.view = None
        try:
            from PySide6.QtWidgets import QApplication
            from .scene3d import Battlefield3DView
        except ImportError as exc:
            raise RuntimeError('Native 3D rendering requires had-env[viewer3d] (PySide6 and VisPy).') from exc
        existing = QApplication.instance()
        if existing is not None and not isinstance(existing, QApplication):
            raise RuntimeError('Native 3D rendering requires a QApplication with QWidget/OpenGL support.')
        self._app = existing or QApplication([])
        if existing is None:
            self._app.setQuitOnLastWindowClosed(False)
        try:
            self.view = Battlefield3DView()
            self.view.resize(1000, 700)
            self.view.setWindowTitle('SkyHAD · 3D environment')
            self.view.set_layers(dict(labels=True, trails=False, groups=False,
                                      assignments=False, ranges=False, velocities=True))
            frame = simulation_frame(simulation)
            self.view.set_world_bounds((), metadata=frame['metadata'], bounds=simulation.world_bounds)
            self.view.set_frame(frame)
            self.metadata = frame['metadata']
        except Exception:
            if self.view is not None:
                self.view.close_resources()
                self.view.close()
            raise

    def _update(self):
        if self._closed or self.view._closed:
            self._closed = True
            return False
        self._app.processEvents()
        if self.view._closed:
            self._closed = True
            return False
        if self._shown and not self.view.isVisible():
            self.close()
            return False
        frame = simulation_frame(self.simulation)
        self.metadata = frame['metadata']
        self.view.set_frame(frame)
        if not self._shown:
            self.view.show()
            self._shown = True
        self._app.processEvents()
        return not self.view._closed

    def render(self):
        """Paint current truth, process window input and return False after close."""
        return bool(self._update())

    def render_rgb_array(self):
        """Capture the real scene as independently owned HWC uint8 RGB pixels."""
        if not self._update():
            raise RuntimeError('The native 3D window is closed; create a new renderer for RGB capture.')
        image = self.view.snapshot()
        stride, height, width = image.bytesPerLine(), image.height(), image.width()
        pixels = np.frombuffer(image.constBits(), dtype=np.uint8, count=height * stride)
        rgba = pixels.reshape(height, stride)[:, :width * 4].reshape(height, width, 4)
        return rgba[:, :, :3].copy()

    def close(self):
        """Release this scene without closing other windows or ending Python."""
        self._closed = True
        if self.view is not None:
            self.view.close_resources()
            self.view.close()
        self._app.processEvents()
