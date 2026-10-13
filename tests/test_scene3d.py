"""Geometry and real-canvas contracts for the presentation-only 3D view."""
import copy
import importlib
import math
import os
import subprocess
import sys

import numpy as np
import pytest


def scene():
    assert importlib.util.find_spec('had_env.render.scene3d') is not None, '3D presentation module missing'
    return importlib.import_module('had_env.render.scene3d')


def aircraft(**changes):
    row = dict(side='red', id=4, position=[30, 40, 100],
               env_agent_type='UAV_fixedwing', attitude=[1, 0, 0, 0],
               alive=True, health=1, role='Attack')
    row.update(changes)
    return row


def test_geometry_import_does_not_load_qt_or_vispy():
    scene()
    result = subprocess.run([sys.executable, '-c',
        "import sys; import had_env.render.scene3d; assert not any(k.startswith(('vispy', 'PySide6')) for k in sys.modules)"],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_scalar_first_rotation_and_quaternion_sign_are_physical():
    module = scene()
    half = math.sqrt(.5)
    rotation = module.attitude_rotation(aircraft(attitude=[half, 0, 0, half]))
    np.testing.assert_allclose(rotation @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(rotation @ [0, 0, 1], [0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(rotation, module.attitude_rotation(aircraft(attitude=[-half, 0, 0, -half])))


def test_roll_sign_and_inversion_preserve_three_dimensional_body():
    module = scene()
    half = math.sqrt(.5)
    for quaternion, expected in (([half, half, 0, 0], [0, 0, 1]),
                                  ([half, -half, 0, 0], [0, 0, -1]),
                                  ([0, 1, 0, 0], [0, -1, 0])):
        np.testing.assert_allclose(module.attitude_rotation(aircraft(attitude=quaternion)) @ [0, 1, 0], expected, atol=1e-12)
    mesh = module.body_mesh('UAV_fixedwing')
    assert np.ptp(mesh['vertices'][:, 2]) > .2  # A tail and body have genuine depth.
    transformed = module.entity_geometry(aircraft(attitude=[0, 1, 0, 0]), 10)
    np.testing.assert_allclose(transformed['vertices'], mesh['vertices'] * [10, -10, -10] + [30, 40, 100])


@pytest.mark.parametrize('attitude', [None, [0, 0, 0, 0], [float('nan'), 0, 0, 0], ['bad', 0, 0, 0]])
def test_missing_attitude_does_not_invent_orientation_from_velocity(attitude):
    module = scene()
    row = aircraft(attitude=attitude, velocity=[0, 100, 0])
    geometry = module.entity_geometry(row, 10)
    assert not geometry['known_attitude']
    assert geometry['kind'] == 'unknown'
    assert 'unknown' in geometry['status'].lower()
    np.testing.assert_array_equal(geometry['position'], [30, 40, 100])


def test_invalid_primary_attitude_falls_back_to_rigid_state():
    module = scene()
    row = aircraft(attitude=[0, 0, 0, 0], rigid_state=[30, 40, 100, 0, 0, 0, 1, 0, 0, 0])
    np.testing.assert_allclose(module.attitude_rotation(row), np.eye(3))


def test_quadrotor_is_plus_configuration_with_front_right_rear_left_rotors():
    mesh = scene().body_mesh('UAV_quadrotor')
    np.testing.assert_allclose(mesh['rotor_centers'], [[.85, 0, .12], [0, -.85, .12], [-.85, 0, .12], [0, .85, .12]])
    assert len(mesh['faces']) > 32
    assert np.ptp(mesh['vertices'][:, 2]) > 0


def test_body_surface_normals_face_outward_for_correct_lighting():
    mesh = scene().body_mesh('UAV_fixedwing')
    triangles = mesh['vertices'][mesh['faces'][:12]]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    assert np.all(np.sum(normals * triangles.mean(axis=1), axis=1) > 0)


def test_bounds_are_raw_xyz_and_explicit_bounds_take_precedence():
    module = scene()
    frames = [dict(entities=[aircraft(position=[-200, 400, 1200]), aircraft(position=[300, -100, 20])])]
    np.testing.assert_array_equal(module.world_bounds(frames), [[-200, 300], [-100, 400], [20, 1200]])
    np.testing.assert_array_equal(module.world_bounds(frames, bounds=[[-20, 20], [-100, 100], [0, 50]]), [[-20, 20], [-100, 100], [0, 50]])
    assert np.isfinite(module.world_bounds([dict(entities=[aircraft(position=[0, float('nan'), 0])])])).all()


def test_sparse_trails_keep_recorded_samples_and_side_identity():
    module = scene()
    frames = [dict(step=step, entities=[aircraft(position=[step, 4, 5]), aircraft(side='blue', position=[-step, 4, 5])]) for step in (0, 3, 8)]
    before = copy.deepcopy(frames)
    trails = module.sampled_trails(frames)
    np.testing.assert_array_equal(trails[('red', '4')], [[0, 4, 5], [3, 4, 5], [8, 4, 5]])
    np.testing.assert_array_equal(trails[('blue', '4')], [[0, 4, 5], [-3, 4, 5], [-8, 4, 5]])
    assert frames == before


def test_picking_uses_screen_nearest_visible_glyph_radius():
    pick = scene().nearest_entity
    candidates = [(('red', '4'), [100, 100], 8), (('blue', '4'), [110, 100], 30)]
    assert pick([104, 100], candidates) == ('red', '4')
    assert pick([120, 100], candidates) == ('blue', '4')
    assert pick([170, 100], candidates) is None


def test_target_and_dead_status_use_truth_without_mutating_entity():
    row = aircraft(side='targets', role='Entity', alive=False, attitude=None, health=None)
    before = copy.deepcopy(row)
    geometry = scene().entity_geometry(row, 10)
    assert geometry['kind'] == 'target'
    assert geometry['dead']
    assert 'HP ?' in geometry['status']
    assert row == before


@pytest.fixture
def real_view():
    pytest.importorskip('PySide6')
    pytest.importorskip('vispy')
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    try:
        view = scene().Battlefield3DView()
    except RuntimeError as exc:
        cause = str(exc.__cause__).lower()
        if os.environ.get('SKYHAD_REQUIRE_3D') or not any(word in cause for word in ('context', 'opengl', 'framebuffer', 'glir', 'graphics driver')):
            raise
        pytest.skip(str(exc))
    view.resize(800, 600)
    view.show()
    app.processEvents()
    yield view, app
    view.close_resources()
    view.close()
    app.processEvents()


def test_real_canvas_reuses_entities_camera_links_and_renders(real_view):
    view, app = real_view
    original = dict(entities=[aircraft(), aircraft(side='blue', id=4, env_agent_type='UAV_quadrotor')])
    frame = copy.deepcopy(original)
    view.set_world_bounds([frame], bounds=[[-100, 100], [-100, 100], [0, 200]])
    view.set_frame(frame, [frame])
    item = view.entities[('red', '4')]
    view.set_frame(frame, [frame])
    assert view.entities[('red', '4')] is item
    view.set_selected(('red', '4'))
    view.focus_selected()
    np.testing.assert_allclose(view.camera_state()['center'], [30, 40, 100])
    signals = []
    view.camera_changed.connect(signals.append)
    state = view.camera_state()
    state['azimuth'] += 20
    view.set_camera_state(state)
    app.processEvents()
    assert signals == []
    state['azimuth'] += 20
    view.set_camera_state(state, emit=True)
    assert len(signals) == 1
    view.set_layers(dict(labels=True, ranges=True, trails=True, groups=True, assignments=True, velocities=True))
    for theme in (True, False):
        view.set_light_theme(theme)
        image = view.snapshot()
        assert not image.isNull()
        assert image.width() >= 600
    assert frame == original
    view.clear()
    assert not view.entities


def test_real_canvas_pick_and_camera_interaction_follow_visible_position(real_view):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    view, app = real_view
    view.set_world_bounds([], bounds=[[-100, 100], [-100, 100], [0, 200]])
    view.set_frame(dict(entities=[aircraft(position=[0, 0, 100])]))
    view.set_camera_state(dict(center=[0, 0, 100]))
    view.snapshot()
    chosen, camera_changes = [], []
    view.entity_selected.connect(lambda side, identifier: chosen.append((side, identifier)))
    view.camera_changed.connect(camera_changes.append)
    center = QPoint(view.canvas.native.width() // 2, view.canvas.native.height() // 2)
    QTest.mouseClick(view.canvas.native, Qt.MouseButton.LeftButton, pos=center)
    app.processEvents()
    assert chosen == [('red', '4')]
    assert view.selected_key == ('red', '4')
    before = view.camera_state()
    QTest.mousePress(view.canvas.native, Qt.MouseButton.LeftButton, pos=center)
    QTest.mouseMove(view.canvas.native, center + QPoint(60, 30), delay=20)
    QTest.mouseRelease(view.canvas.native, Qt.MouseButton.LeftButton, pos=center + QPoint(60, 30))
    app.processEvents()
    assert view.camera_state()['azimuth'] != before['azimuth']
    assert camera_changes


def test_real_canvas_images_distinguish_inversion_and_unknown_attitude(real_view):
    view, app = real_view
    view.set_world_bounds([], bounds=[[-10, 10], [-10, 10], [-10, 10]])
    view.set_symbol_scale(.1)
    images = []
    for quaternion in ([1, 0, 0, 0], [0, 1, 0, 0], None):
        view.set_frame(dict(entities=[aircraft(position=[0, 0, 0], attitude=quaternion)]))
        app.processEvents()
        image = view.snapshot()
        images.append(np.frombuffer(image.constBits(), dtype=np.uint8).copy())
    assert np.count_nonzero(images[0] != images[1]) > 100
    assert np.count_nonzero(images[0] != images[2]) > 100


def test_real_canvas_keeps_scale_and_interaction_text_inside_image(real_view):
    view, app = real_view
    view.set_layers(dict(ranges=True))
    image = view.snapshot()
    pixels = np.frombuffer(image.constBits(), dtype=np.uint8).reshape(image.height(), image.width(), 4)
    background = pixels[100, 20, :3]
    for start in (15, 35, 55):
        assert np.count_nonzero(np.any(pixels[start:start + 20, 14:700, :3] != background, axis=2)) > 50


def test_real_canvas_dense_4v4_with_unknown_dead_target_and_range(real_view):
    view, app = real_view
    entities = []
    for side, direction in (('red', -1), ('blue', 1)):
        for index in range(4):
            entities.append(aircraft(side=side, id=index,
                position=[direction * (400 + index * 140), -600 + index * 400, 400 + index * 150],
                env_agent_type='UAV_fixedwing' if index < 2 else 'UAV_quadrotor',
                attitude=None if index == 3 else [1, 0, 0, 0],
                alive=None if index == 3 else True, attack_range=450, velocity=[100, 10, 0]))
    entities.extend([aircraft(side='targets', id=0, role='Entity', position=[900, 900, 150]),
                     aircraft(side='targets', id=1, role='Entity', position=[-900, 900, 150], alive=False)])
    frame = dict(entities=entities, groups={'red':[dict(members=[0, 1], target_id=0)]})
    view.set_world_bounds([frame], bounds=[[-1500, 1500], [-1500, 1500], [0, 1400]])
    view.set_frame(frame, [frame])
    view.set_selected(('red', '0'))
    view.set_layers(dict(labels=True, ranges=True, trails=True, groups=True, assignments=True))
    # Live polling may deliver several different samples before a native paint.
    original = copy.deepcopy(frame)
    for step in (3, 8, 14):
        next_frame = copy.deepcopy(frame)
        for row in next_frame['entities'][:8]:
            row['position'][0] += step * 10
            if row['attitude'] is not None:
                row['attitude'] = [math.cos(step / 10), math.sin(step / 10), 0, 0]
        view.set_frame(next_frame, [frame, next_frame])
    for light in (False, True):
        view.set_light_theme(light)
        app.processEvents()
        assert not view.snapshot().isNull()
    assert len(view.entities) == 10
    np.testing.assert_allclose(view.entities[('red', '0')]['geometry']['position'], [-260, -600, 400])
    assert frame == original


def test_range_sphere_preserves_visible_selected_aircraft(real_view):
    view, app = real_view
    view.set_world_bounds([], bounds=[[-10, 10], [-10, 10], [-10, 10]])
    view.set_symbol_scale(.07)
    view.set_frame(dict(entities=[aircraft(position=[0, 0, 0], attack_range=6)]))
    view.set_selected(('red', '4'))
    view.set_layers(dict(ranges=True))
    for light in (False, True, False):
        view.set_light_theme(light)
        app.processEvents()
        image = view.snapshot()
        pixels = np.frombuffer(image.constBits(), dtype=np.uint8).reshape(image.height(), image.width(), 4)
        center = pixels[image.height() // 2 - 60:image.height() // 2 + 60,
                        image.width() // 2 - 60:image.width() // 2 + 60, :3].astype(float)
        red = (center[:, :, 0] > center[:, :, 1] * 1.2) & (center[:, :, 0] > center[:, :, 2] * 1.3)
        assert red.sum() > 40


def test_narrow_canvas_wraps_hud_and_selected_status_within_available_width(real_view):
    from PySide6.QtGui import QFont, QFontMetricsF
    view, app = real_view
    view.resize(250, 650)
    app.processEvents()
    view.set_frame(dict(entities=[aircraft(attack_range=450)]))
    view.set_selected(('red', '4'))
    view.set_layers(dict(ranges=True))
    view.snapshot()
    for visual, point_size in ((view._hud, 10), (view._selection_info, 9)):
        metrics = QFontMetricsF(QFont('OpenSans', point_size))
        lines = [visual.text] if isinstance(visual.text, str) else visual.text
        assert len(lines) > 1
        assert all(metrics.horizontalAdvance(line) <= view.canvas.size[0] - 28 for line in lines)
        positions = np.asarray(visual.pos).reshape(-1, 3)
        assert positions[:, 0].min() >= 12
        assert positions[:, 1].min() >= 20
        assert positions[:, 1].max() <= view.canvas.size[1] - 12
    assert 'HP' in ' '.join(view._selection_info.text)
    assert 'symbolic' in ' '.join(view._hud.text)
    image = view.snapshot()
    assert image.width() == 250


def test_roomy_canvas_retains_full_metric_and_range_semantics_after_resize(real_view):
    view, app = real_view
    view.resize(250, 650)
    app.processEvents()
    view.set_layers(dict(ranges=True))
    view.resize(1100, 650)
    app.processEvents()
    text = ' '.join(view._hud.text)
    assert 'equal metric axes' in text
    assert 'maximum-radius sphere' in text
    assert 'recorded events determine hits' in text


def test_overlay_wrap_keeps_long_ids_and_words_without_overflow():
    wrap = scene().wrapped_overlay_text
    assert wrap('red aircraft-12345678 HP 1', 8, len) == ['red', 'aircraft', '-1234567', '8 HP 1']


def test_fit_world_keeps_rotated_bounds_inside_narrow_canvas(real_view):
    view, app = real_view
    view.resize(250, 650)
    app.processEvents()
    view.set_world_bounds([], bounds=[[-1000, 1000], [-1000, 1000], [0, 1000]])
    view.snapshot()
    transform = view.view.scene.node_transform(view.canvas.scene)
    corners = np.array([[x, y, z] for x in (-1000, 1000) for y in (-1000, 1000) for z in (0, 1000)])
    points = np.asarray(transform.map(corners))
    points = points / points[:, 3:4]
    assert points[:, 0].min() >= 18
    assert points[:, 0].max() <= view.canvas.size[0] - 18
