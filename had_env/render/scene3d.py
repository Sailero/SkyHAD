"""Metric, read-only VisPy scene. Geometry remains usable without GUI packages.

The widget is resolved lazily so headless simulation/evaluation users never load
Qt or OpenGL. Aircraft geometry is symbolic; only its anchor is physical size.
Body axes are FLU, recorded quaternions are scalar-first FLU -> ENU.
"""
from __future__ import annotations

from functools import lru_cache
import math
import numpy as np

from had_env.dynamics.control import quaternion_matrix


DEFAULT_LAYERS = dict(labels=False, trails=True, groups=True, assignments=False,
                      ranges=False, velocities=True)
SIDE_COLORS = {'red': '#ef777b', 'blue': '#70b7ff', 'targets': '#f3bd72'}


def entity_key(entity):
    return str(entity.get('side', '')), str(entity.get('id', ''))


def valid_position(entity):
    try:
        value = np.asarray(entity.get('position'), dtype=float)
        return value.shape == (3,) and bool(np.isfinite(value).all())
    except (TypeError, ValueError):
        return False


def attitude_rotation(entity):
    """Return physical FLU -> ENU rotation, or None when attitude is unknown."""
    rigid = entity.get('rigid_state')
    fallback = rigid[6:10] if isinstance(rigid, (list, tuple, np.ndarray)) else None
    for value in (entity.get('attitude'), fallback):
        try:
            q = np.asarray(value, dtype=float)
            if q.shape == (4,) and np.isfinite(q).all() and np.linalg.norm(q) > 1e-12:
                return quaternion_matrix(q / np.linalg.norm(q))
        except (TypeError, ValueError):
            pass
    return None


def _box(center, size):
    vertices = np.array([[x, y, z] for z in (-1, 1) for y in (-1, 1) for x in (-1, 1)], dtype=float)
    vertices = vertices * np.asarray(size) / 2 + center
    faces = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
                      [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                      [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], dtype=np.uint32)
    return vertices, faces[:, ::-1].copy()


def _disc(center, radius=.28, height=.06):
    angles = np.arange(16) * math.tau / 16
    ring = np.c_[radius * np.cos(angles), radius * np.sin(angles), np.full(16, -height / 2)]
    vertices = np.vstack([ring, ring + [0, 0, height], [[0, 0, -height / 2], [0, 0, height / 2]]]) + center
    faces = []
    for i in range(16):
        j = (i + 1) % 16
        faces.extend([[32, j, i], [33, i + 16, j + 16], [i, j, j + 16], [i, j + 16, i + 16]])
    return vertices, np.asarray(faces, dtype=np.uint32)


@lru_cache(maxsize=3)
def body_mesh(model):
    """Genuine closed triangular bodies in FLU; + layout is front/right/rear/left."""
    rotors = np.empty((0, 3))
    if model == 'UAV_fixedwing':
        pieces = [_box([0, 0, 0], [2.5, .25, .25]),
                  _box([.05, 0, .04], [.5, 2.1, .07]),
                  _box([-.95, 0, .10], [.35, .90, .06]),
                  _box([-1.03, 0, .28], [.32, .06, .5])]
        # Tapered nose distinguishes physical front from the tail.
        pieces.append((np.array([[1.25, -.125, -.125], [1.25, .125, -.125],
                                 [1.25, .125, .125], [1.25, -.125, .125], [1.6, 0, 0]]),
                       np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]], dtype=np.uint32)))
        nose = np.array([1.6, 0, 0])
    elif model == 'UAV_quadrotor':
        rotors = np.array([[.85, 0, .12], [0, -.85, .12], [-.85, 0, .12], [0, .85, .12]])
        pieces = [_box([0, 0, 0], [.65, .42, .22]), _box([0, 0, 0], [1.7, .09, .09]),
                  _box([0, 0, 0], [.09, 1.7, .09]), _box([.34, 0, .14], [.18, .16, .16])]
        pieces.extend(_disc(center) for center in rotors)
        nose = np.array([.65, 0, .12])
    else:
        pieces = [_box([0, 0, 0], [.65, .65, .65])]
        nose = np.array([0, 0, .4])
    vertices, faces, offset = [], [], 0
    for v, f in pieces:
        vertices.append(v)
        faces.append(f + offset)
        offset += len(v)
    result = dict(vertices=np.vstack(vertices), faces=np.vstack(faces), rotor_centers=rotors, nose=nose)
    for value in result.values():
        value.flags.writeable = False
    return result


def entity_geometry(entity, size):
    """Derive geometry and explicit status without writing recorded data."""
    position = np.asarray(entity['position'], dtype=float)
    rotation = attitude_rotation(entity)
    target = entity.get('side') == 'targets' or str(entity.get('role', '')).lower() in {'target', 'entity'}
    model = entity.get('env_agent_type')
    kind = 'target' if target else model if rotation is not None and model in {'UAV_fixedwing', 'UAV_quadrotor'} else 'unknown'
    mesh = body_mesh(model if kind not in {'target', 'unknown'} else 'target')
    world_rotation = np.eye(3) if kind in {'target', 'unknown'} else rotation
    vertices = mesh['vertices'] @ world_rotation.T * size + position
    dead = entity.get('alive') is False or entity.get('alive') == 0
    state = 'dead' if dead else 'alive unknown' if entity.get('alive') is None else 'alive'
    health = entity.get('health')
    hp = f'{health:g}' if isinstance(health, (float, int)) and math.isfinite(health) else '?'
    if rotation is None:
        attitude = 'attitude unknown'
    else:
        # Recover physical bank/pitch directly from the body axes (FLU).
        bank = math.degrees(math.atan2(rotation[2, 1], rotation[2, 2]))
        pitch = math.degrees(math.asin(float(np.clip(rotation[2, 0], -1, 1))))
        attitude = f'R {bank:+.0f}°  P {pitch:+.0f}°'
    status = f"{entity.get('side', '')} {entity.get('id', '')} · {entity.get('role', 'unknown')} · HP {hp} · {state} · {attitude}"
    return dict(position=position.copy(), vertices=vertices, faces=mesh['faces'],
                nose=mesh['nose'] @ world_rotation.T * size + position,
                known_attitude=rotation is not None, kind=kind, dead=dead,
                unknown_alive=entity.get('alive') is None, status=status)


def world_bounds(frames, metadata=None, bounds=None):
    metadata = metadata or {}
    configured = bounds if bounds is not None else metadata.get('effective_config', {}).get('world_bounds', metadata.get('world_bounds'))
    try:
        value = np.asarray(configured, dtype=float)
        if value.shape == (3, 2) and np.isfinite(value).all() and np.all(value[:, 1] >= value[:, 0]):
            return value.copy()
    except (TypeError, ValueError):
        pass
    points = [e['position'] for frame in frames for e in frame.get('entities', []) if valid_position(e)]
    if points:
        points = np.asarray(points, dtype=float)
        return np.c_[points.min(axis=0), points.max(axis=0)]
    return np.array([[-2700., 2700.], [-2700., 2700.], [0., 1200.]])


def sampled_trails(history):
    samples = {}
    for frame in history:
        for entity in frame.get('entities', []):
            if valid_position(entity):
                samples.setdefault(entity_key(entity), []).append(entity['position'])
    return {key: np.asarray(points, dtype=float).copy() for key, points in samples.items()}


def nearest_entity(position, candidates):
    """Screen-nearest anchor within its projected glyph radius; no global pick radius."""
    position = np.asarray(position, dtype=float)[:2]
    nearest, best = None, math.inf
    for key, projected, radius in candidates:
        projected = np.asarray(projected, dtype=float)[:2]
        if not np.isfinite(projected).all():
            continue
        distance = float(np.linalg.norm(projected - position))
        if distance <= radius and distance < best:
            nearest, best = key, distance
    return nearest


def wrapped_overlay_text(text, width, measure):
    """Wrap annotation words against measured pixels, including long identifiers."""
    lines = []
    for paragraph in text.splitlines():
        line = ''
        for word in paragraph.split():
            if line and measure(line + ' ' + word) > width:
                lines.append(line)
                line = ''
            while measure(word) > width and len(word) > 1:
                split = len(word) - 1
                while split > 1 and measure(word[:split]) > width:
                    split -= 1
                lines.append(word[:split])
                word = word[split:]
            line = f'{line} {word}'.strip()
        if line:
            lines.append(line)
    return lines or ['']


def _widget_class():
    try:
        from PySide6.QtCore import Signal
        from PySide6.QtGui import QImage, QGuiApplication, QOffscreenSurface, QOpenGLContext, QFont, QFontMetricsF
        from PySide6.QtWidgets import QWidget, QVBoxLayout
        from vispy import app, scene
        from vispy.geometry import create_sphere
        from vispy.color import Color
    except ImportError as exc:
        raise RuntimeError('3D view requires PySide6 and VisPy; install had-env[viewer3d].') from exc

    class ReportingCamera(scene.cameras.TurntableCamera):
        _state_props = scene.cameras.TurntableCamera._state_props + ('distance',)

        def viewbox_mouse_event(self, event):
            # Resize and projection updates also call view_changed; only user
            # camera input is a linking event. Programmatic changes emit below.
            super().viewbox_mouse_event(event)
            callback = getattr(self, '_report_change', None)
            if callback is not None:
                callback()

    class MetricCanvas(scene.SceneCanvas):
        def _draw_scene(self, bgcolor=None):
            # glClear honours depth_mask; translucent visuals may have disabled
            # it on the preceding draw. Restore it before clearing every frame.
            self.context.set_depth_mask(True)
            super()._draw_scene(bgcolor)

    class Battlefield3DView(QWidget):
        entity_selected = Signal(str, str)
        camera_changed = Signal(object)

        def __init__(self, parent=None):
            super().__init__(parent)
            self.entities = {}
            self.frame, self.history, self.metadata = {}, (), {}
            self.layers = DEFAULT_LAYERS.copy()
            self.selected_key = None
            self.light = False
            self.symbol_scale = .02
            self._bounds = world_bounds(())
            self._camera_suppressed = True
            self._last_camera = None
            self._closed = False
            self.canvas = None
            try:
                if QGuiApplication.platformName() in {'offscreen', 'minimal'}:
                    raise RuntimeError('This Qt platform has no supported QOpenGLWidget context')
                # Test availability without rendering the hidden QOpenGLWidget.
                # Rendering it before first show causes Qt to replace its context,
                # invalidating the shader/FBO handles cached by VisPy on Windows.
                surface = QOffscreenSurface()
                surface.create()
                probe = QOpenGLContext()
                if not probe.create() or not probe.makeCurrent(surface):
                    raise RuntimeError('A usable OpenGL context could not be created')
                probe.doneCurrent()
                application = app.use_app('pyside6')
                self.canvas = MetricCanvas(keys='interactive', bgcolor='#17202f',
                                                size=(800, 600), app=application, show=False)
                layout = QVBoxLayout(self)
                layout.setContentsMargins(0, 0, 0, 0)
                layout.addWidget(self.canvas.native)
                self.view = self.canvas.central_widget.add_view()
                self.camera = ReportingCamera(up='+z', fov=45, elevation=25, azimuth=-45)
                self.view.camera = self.camera
                self.camera._report_change = self._report_camera
                self._line_visuals = {name: scene.visuals.Line(parent=self.view.scene, method='gl',
                                                              connect='segments')
                                      for name in ('bounds', 'axes', 'trails', 'velocity', 'groups', 'assignments', 'range_rings')}
                self._world_labels = []
                self._range_mesh = scene.visuals.Mesh(meshdata=create_sphere(rows=12, cols=24),
                                                      color=(.6, .7, .9, .07), parent=self.view.scene)
                self._range_mesh.set_gl_state('translucent', depth_test=True, depth_mask=False, cull_face=False)
                self._range_mesh.visible = False
                self._hud = scene.visuals.Text(['', '', ''], parent=self.canvas.scene, font_size=10,
                                               anchor_x='left', anchor_y='top',
                                               pos=[(14, 30), (14, 50), (14, 70)])
                self._selection_info = scene.visuals.Text('', parent=self.canvas.scene, font_size=9,
                                                         anchor_x='left', anchor_y='top', pos=(14, 580))
                self.canvas.events.mouse_press.connect(self._mouse_press)
                self.canvas.events.mouse_release.connect(self._mouse_release)
                self.canvas.events.key_press.connect(self._key_press)
                self.canvas.events.resize.connect(self._resize)
                self._press = None
                self._draw_world()
                self.fit_world()
            except Exception as exc:
                if self.canvas is not None:
                    self.canvas.close()
                raise RuntimeError(f'3D view unavailable: VisPy/PySide6 OpenGL initialization failed ({exc}). Use XY/XZ or a graphics-enabled session.') from exc
            self._camera_suppressed = False
            self._last_camera = self.camera_state()

        @property
        def _symbol_size(self):
            return max(float(np.max(np.ptp(self._bounds, axis=1))), 1.) * self.symbol_scale

        def _report_camera(self):
            if self._camera_suppressed or self._closed:
                return
            state = self.camera_state()
            if state != self._last_camera:
                self._last_camera = state
                self.camera_changed.emit(state)

        def camera_state(self):
            state = self.camera.get_state()
            return {name: list(value) if name == 'center' else float(value) if value is not None else None
                    for name, value in state.items()}

        def set_camera_state(self, state, emit=False):
            allowed = {'center', 'scale_factor', 'elevation', 'azimuth', 'roll', 'distance', 'fov'}
            clean = {}
            for name, value in state.items():
                if name not in allowed:
                    continue
                if name == 'center':
                    array = np.asarray(value, dtype=float)
                    if array.shape != (3,) or not np.isfinite(array).all():
                        raise ValueError('Camera center must contain three finite coordinates')
                    clean[name] = tuple(array)
                elif value is None and name == 'distance':
                    clean[name] = None
                else:
                    value = float(value)
                    if not math.isfinite(value) or (name in {'scale_factor', 'distance'} and value <= 0):
                        raise ValueError('Camera state must be finite with positive scale/distance')
                    clean[name] = value
            self._camera_suppressed = True
            try:
                self.camera.set_state(clean)
            finally:
                self._camera_suppressed = False
            if emit:
                self._report_camera()
            else:
                self._last_camera = self.camera_state()

        def set_world_bounds(self, frames, metadata=None, bounds=None):
            self.metadata = metadata or {}
            self._bounds = world_bounds(frames, self.metadata, bounds)
            self._draw_world()
            self.set_frame(self.frame, self.history)
            self.fit_world()

        def fit_world(self):
            # A rotated world spans its diagonal, not just its longest axis.
            # Use that envelope so reset also fits tall, narrow native windows.
            extent = max(float(np.linalg.norm(np.ptp(self._bounds, axis=1))), 1.)
            self.set_camera_state(dict(center=np.mean(self._bounds, axis=1), scale_factor=extent * 1.35,
                                       distance=None, elevation=25, azimuth=-45, roll=0), emit=True)

        def focus_selected(self):
            item = self.entities.get(self.selected_key)
            if item:
                self.set_camera_state(dict(center=item['geometry']['position'],
                                           scale_factor=max(self._symbol_size * 16, 1.)), emit=True)

        def set_symbol_scale(self, fraction):
            value = float(fraction)
            if not math.isfinite(value) or not 0 < value <= .2:
                raise ValueError('Symbol scale must be between 0 and 0.2 of world extent')
            self.symbol_scale = value
            self.set_frame(self.frame, self.history)

        def _line(self, name, segments, color, width=1):
            visual = self._line_visuals[name]
            visual.visible = bool(len(segments))
            if len(segments):
                visual.set_data(pos=np.asarray(segments, dtype=np.float32).reshape(-1, 3),
                                color=color, width=width, connect='segments')

        def _draw_world(self):
            corners = np.array([[x, y, z] for x in self._bounds[0] for y in self._bounds[1] for z in self._bounds[2]])
            segments = [(corners[i], corners[j]) for i in range(8) for j in range(i + 1, 8)
                        if bin(i ^ j).count('1') == 1]
            self._line('bounds', segments, '#6b7f99' if self.light else '#526079')
            origin = self._bounds[:, 0]
            ends = [origin.copy() for _ in range(3)]
            for axis, end in enumerate(ends):
                end[axis] = self._bounds[axis, 1]
            colors = np.repeat(np.array([[.9, .35, .35, 1], [.3, .75, .5, 1], [.4, .65, 1, 1]]), 2, axis=0)
            self._line('axes', [(origin, end) for end in ends], colors, 2)
            if not self._world_labels:
                self._world_labels = [scene.visuals.Text('', font_size=9, parent=self.view.scene) for _ in range(4)]
            for axis, name in enumerate(('X', 'Y', 'Z up')):
                end = ends[axis]
                text = f'{name}  {self._bounds[axis, 1]:g} m'
                label = self._world_labels[axis]
                label.text, label.pos = text, end
                label.color = '#33465e' if self.light else '#c8d7ec'
            label = self._world_labels[3]
            label.text = ' / '.join(f'{v:g}' for v in origin) + ' m'
            label.pos, label.color = origin, '#33465e' if self.light else '#c8d7ec'

        def _new_item(self):
            node = scene.Node(parent=self.view.scene)
            mesh = scene.visuals.Mesh(parent=node, shading='flat')
            mesh.set_gl_state('opaque', depth_test=True, depth_mask=True, cull_face=False)
            return dict(node=node, mesh=mesh,
                        marker=scene.visuals.Markers(parent=node),
                        nose=scene.visuals.Markers(parent=node),
                        label=scene.visuals.Text('', parent=node, font_size=9,
                                                anchor_x='left', anchor_y='bottom'),
                        selected=scene.visuals.Markers(parent=node))

        def set_frame(self, frame, history=()):
            self.frame = frame or {}
            self.history = tuple(history or ())
            current = {entity_key(e): e for e in self.frame.get('entities', []) if valid_position(e)}
            for key in self.entities.keys() - current.keys():
                self.entities.pop(key)['node'].parent = None
            for key, entity in current.items():
                item = self.entities.setdefault(key, self._new_item()) if key not in self.entities else self.entities[key]
                geometry = entity_geometry(entity, self._symbol_size)
                item['entity'], item['geometry'] = entity, geometry
                color = np.asarray(Color('#77818e' if geometry['dead'] else SIDE_COLORS.get(key[0], '#c1c8d1')).rgba)
                if geometry['unknown_alive']:
                    color[:3] = color[:3] * .65 + .35
                mesh_visible = geometry['kind'] != 'unknown'
                item['mesh'].visible = mesh_visible
                item['marker'].visible = not mesh_visible
                if mesh_visible:
                    body = body_mesh(entity.get('env_agent_type') if geometry['kind'] != 'target' else 'target')
                    colors = np.tile(color, (len(body['vertices']), 1))
                    # A bright dorsal surface and darker belly make inversion observable.
                    colors[body['vertices'][:, 2] < 0, :3] *= .45
                    signature = (geometry['kind'], colors.tobytes())
                    if item.get('mesh_signature') != signature:
                        item['mesh'].set_data(vertices=body['vertices'].astype(np.float32),
                                              faces=geometry['faces'], vertex_colors=colors)
                        item['mesh_signature'] = signature
                    rotation = attitude_rotation(entity) if geometry['kind'] != 'target' else np.eye(3)
                    matrix = np.eye(4)
                    matrix[:3, :3] = rotation.T * self._symbol_size
                    matrix[3, :3] = geometry['position']
                    item['mesh'].transform = scene.transforms.MatrixTransform(matrix=matrix)
                else:
                    item['marker'].set_data(np.asarray([geometry['position']], dtype=np.float32),
                                            face_color=(0, 0, 0, 0), edge_color=color, size=15, edge_width=2,
                                            symbol='diamond')
                item['nose'].visible = geometry['kind'] in {'UAV_fixedwing', 'UAV_quadrotor'} and not geometry['dead']
                if item['nose'].visible:
                    item['nose'].set_data(np.asarray([geometry['nose']], dtype=np.float32), size=5,
                                          face_color='#ffffff', edge_width=0)
                selected = key == self.selected_key
                item['selected'].visible = selected
                if selected:
                    item['selected'].set_data(np.asarray([geometry['position']], dtype=np.float32),
                        size=27, face_color=(0, 0, 0, 0), edge_color='#f8d67a', edge_width=2)
                label = f"{ {'red': 'R', 'blue': 'B', 'targets': 'T'}.get(key[0], key[0])}{key[1]}"
                if geometry['kind'] == 'unknown':
                    label += ' ?'
                if geometry['dead']:
                    label += '  dead'
                item['label'].text = label
                item['label'].pos = geometry['position'] + [self._symbol_size * 1.7, 0, self._symbol_size * .4]
                item['label'].color = '#33465e' if self.light else '#dce6f3'
                item['label'].visible = self.layers['labels'] or selected or geometry['kind'] == 'unknown'
            self._draw_overlays()
            self._update_hud()
            self.canvas.update()

        def _draw_overlays(self):
            trails = []
            if self.layers['trails']:
                for points in sampled_trails(self.history[-12:]).values():
                    trails.extend(zip(points[:-1], points[1:]))
            self._line('trails', trails, '#8295af', 1.2)
            velocities = []
            if self.layers.get('velocities', True):
                for item in self.entities.values():
                    try:
                        velocity = np.asarray(item['entity'].get('velocity'), dtype=float)
                        if velocity.shape == (3,) and np.isfinite(velocity).all() and np.linalg.norm(velocity) > 0:
                            start = item['geometry']['position']
                            end = start + velocity  # One second, preserving the recorded metric vector.
                            velocities.append((start, end))
                            direction = velocity / np.linalg.norm(velocity)
                            side = np.cross(direction, [0, 0, 1])
                            if np.linalg.norm(side) < 1e-8:
                                side = np.cross(direction, [0, 1, 0])
                            side /= np.linalg.norm(side)
                            head = min(np.linalg.norm(velocity) * .25, self._symbol_size * .4)
                            velocities.extend([(end, end - direction * head + side * head * .5),
                                               (end, end - direction * head - side * head * .5)])
                    except (TypeError, ValueError):
                        pass
            self._line('velocity', velocities, '#a6bc9f', 1.4)
            links, assignments = [], []
            for side, groups in (self.frame.get('groups') or {}).items():
                if not isinstance(groups, list):
                    continue
                for group in groups:
                    members = [self.entities[(str(side), str(identifier))] for identifier in group.get('members', [])
                               if (str(side), str(identifier)) in self.entities]
                    members = [item for item in members if not item['geometry']['dead']]
                    if not members:
                        continue
                    center = np.mean([item['geometry']['position'] for item in members], axis=0)
                    if self.layers['groups']:
                        links.extend((center, item['geometry']['position']) for item in members)
                    target = self.entities.get(('targets', str(group.get('target_id'))))
                    if self.layers['assignments'] and target:
                        assignments.append((center, target['geometry']['position']))
            self._line('groups', links, '#8294aa', 1.2)
            self._line('assignments', assignments, '#f3bd72', 1.2)
            selected = self.entities.get(self.selected_key)
            radius = selected['entity'].get('attack_range', selected['entity'].get('range')) if selected else None
            visible = self.layers['ranges'] and isinstance(radius, (int, float)) and math.isfinite(radius) and radius > 0
            self._range_mesh.visible = bool(visible)
            rings = []
            if visible:
                center = selected['geometry']['position']
                self._range_mesh.transform = scene.transforms.STTransform(scale=(radius,) * 3, translate=center)
                t = np.linspace(0, math.tau, 65)
                for axes in ((0, 1), (0, 2), (1, 2)):
                    points = np.tile(center, (65, 1))
                    points[:, axes[0]] += radius * np.cos(t)
                    points[:, axes[1]] += radius * np.sin(t)
                    rings.extend(zip(points[:-1], points[1:]))
            self._line('range_rings', rings, (.65, .72, .85, .5), 1)

        def _update_hud(self):
            width, height = self.canvas.size
            if width < 560:
                text = [
                    'ENU / m · equal axes',
                    f'Glyphs: symbolic · {self.symbol_scale:.1%} extent',
                    'Drag orbit · Shift+drag pan',
                    'Wheel zoom · R reset · F focus',
                    'Velocity: 1 s',
                ]
                if self.layers['ranges']:
                    text += ['Range: approx. max sphere', 'Hits follow recorded events']
            else:
                text = [f'ENU / m · equal metric axes · glyph size symbolic ({self.symbol_scale:.1%} of extent)',
                        'Drag: orbit · Shift+drag: pan · wheel: zoom · R: reset · F: focus · v: 1 s']
                if self.layers['ranges']:
                    text.append('Range: approximate maximum-radius sphere; recorded events determine hits')
            if any(item['geometry']['kind'] == 'unknown' for item in self.entities.values()):
                text.append('?: attitude unknown')
            if 'step' in self.frame:
                metadata = self.frame.get('metadata', {})
                text.append(f"Step {self.frame['step']} · t {self.frame.get('sim_time', 0):g} s"
                            f" · target damage {metadata.get('target_damage', 0):g}")
            metrics = QFontMetricsF(QFont('OpenSans', 10))
            available = max(width - 28, 12)
            # Qt may substitute a local font for VisPy's bundled OpenSans; the
            # small margin accommodates that difference and glyph antialiasing.
            measure = lambda value: metrics.horizontalAdvance(value) * 1.1
            lines = [line for paragraph in text for line in wrapped_overlay_text(paragraph, available, measure)]
            self._hud.text = lines
            self._hud.pos = [(14, 30 + index * 20) for index in range(len(lines))]
            color = '#33465e' if self.light else '#dce6f3'
            self._hud.color = color
            item = self.entities.get(self.selected_key)
            metrics = QFontMetricsF(QFont('OpenSans', 9))
            status = item['geometry']['status'] if item else ''
            if width < 560 and status:
                parts = status.split(' · ')
                paragraphs = [' · '.join(parts[:2]), ' · '.join(parts[2:4]), ' · '.join(parts[4:])]
            else:
                paragraphs = [status]
            lines = [line for paragraph in paragraphs for line in wrapped_overlay_text(
                paragraph, available, lambda value: metrics.horizontalAdvance(value) * 1.1)]
            self._selection_info.text = lines
            self._selection_info.pos = [(14, height - 14 - (len(lines) - index - 1) * 18)
                                        for index in range(len(lines))]
            self._selection_info.color = color

        def set_selected(self, key_or_none):
            self.selected_key = None if key_or_none is None else tuple(map(str, key_or_none))
            self.set_frame(self.frame, self.history)

        def select_entity(self, side, identifier):
            self.set_selected((side, identifier))

        def set_layers(self, layers_dict):
            self.layers.update({key: bool(value) for key, value in layers_dict.items()})
            self.set_frame(self.frame, self.history)

        def set_layer(self, name, enabled):
            self.set_layers({name: enabled})

        def set_light_theme(self, enabled):
            self.light = bool(enabled)
            self.canvas.bgcolor = '#f1f4f8' if self.light else '#17202f'
            self._draw_world()
            self.set_frame(self.frame, self.history)

        def _mouse_press(self, event):
            self._press = np.asarray(event.pos, dtype=float) if event.button == 1 else None

        def _mouse_release(self, event):
            if event.button != 1 or self._press is None or np.linalg.norm(np.asarray(event.pos) - self._press) > 4:
                self._press = None
                return
            self._press = None
            candidates = []
            transform = self.view.scene.node_transform(self.canvas.scene)
            width, height = self.canvas.size
            for key, item in self.entities.items():
                geometry = item['geometry']
                projected = np.asarray(transform.map(geometry['position']))
                if len(projected) > 3:
                    if projected[3] <= 0:
                        continue
                    projected = projected / projected[3]
                if not np.isfinite(projected).all() or not (0 <= projected[0] <= width and 0 <= projected[1] <= height):
                    continue
                points = np.asarray(transform.map(geometry['vertices']))
                if points.shape[1] > 3:
                    points = points / points[:, 3:4]
                radius = float(np.nanmax(np.linalg.norm(points[:, :2] - projected[:2], axis=1))) if geometry['kind'] != 'unknown' else 8.
                candidates.append((key, projected[:2], max(5., radius)))
            key = nearest_entity(event.pos, candidates)
            if key is not None:
                self.set_selected(key)
                self.entity_selected.emit(*key)

        def _key_press(self, event):
            key = str(event.key).lower()
            if key == 'r':
                self.fit_world()
            elif key == 'f':
                self.focus_selected()

        def _resize(self, event):
            self._update_hud()

        def clear(self):
            self.selected_key = None
            self.set_frame({}, ())

        def snapshot(self):
            try:
                if not self.canvas.native.isValid():
                    raise RuntimeError('Show the 3D canvas before taking its OpenGL snapshot')
                pixels = np.ascontiguousarray(self.canvas.render(alpha=True), dtype=np.uint8)
                height, width, channels = pixels.shape
                if channels != 4:
                    raise RuntimeError('OpenGL snapshot did not return RGBA pixels')
                return QImage(pixels.data, width, height, pixels.strides[0], QImage.Format.Format_RGBA8888).copy()
            except Exception as exc:
                raise RuntimeError(f'3D OpenGL snapshot failed: {exc}') from exc

        def close_resources(self):
            if not self._closed:
                self._closed = True
                self.camera._report_change = None
                self.canvas.close()

        def closeEvent(self, event):
            self.close_resources()
            super().closeEvent(event)

    Battlefield3DView.__module__ = __name__
    return Battlefield3DView


def __getattr__(name):
    if name == 'Battlefield3DView':
        widget = _widget_class()
        globals()[name] = widget
        return widget
    raise AttributeError(name)
