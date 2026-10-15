"""Orthographic aircraft glyphs for the 2D XY/XZ projections.

Each aircraft is a small set of flat body-frame (FLU) shapes: a top planform
and a side outline. Rotating them by the physical attitude quaternion and
projecting onto the panel plane shows heading, pitch and bank directly, so a
banked wing looks shorter from above and a climbing fuselage tilts in the side
view. Glyph size is symbolic (pixels), not metric.
"""
from __future__ import annotations

import math

import numpy as np

from had_env.dynamics.control import quaternion_matrix


def _planar(points, axis):
    """Lift 2D outline points into the body x-y (axis=2) or x-z (axis=1) plane."""
    result = np.zeros((len(points), 3))
    result[:, 0] = [p[0] for p in points]
    result[:, 2 if axis == 1 else 1] = [p[1] for p in points]
    return result


def _ring(center, radius, count=12):
    angle = np.linspace(0., 2. * np.pi, count, endpoint=False)
    return np.c_[center[0] + radius*np.cos(angle), center[1] + radius*np.sin(angle), np.full(count, center[2])]


FIXEDWING_TOP = _planar(((1.4, 0), (.45, -.16), (.1, -.95), (-.2, -.95), (-.25, -.16), (-.9, -.13),
                         (-1.1, -.42), (-1.25, -.42), (-1.1, 0), (-1.25, .42), (-1.1, .42), (-.9, .13),
                         (-.25, .16), (-.2, .95), (.1, .95), (.45, .16)), axis=2)
FIXEDWING_SIDE = _planar(((1.4, 0), (.9, .17), (-.75, .17), (-1.05, .65), (-1.25, .65), (-1.1, -.13),
                          (.85, -.13)), axis=1)
# Plus configuration matching the dynamics: rotors front, right, rear, left.
QUAD_ROTORS = ((.85, 0., .12), (0., -.85, .12), (-.85, 0., .12), (0., .85, .12))
QUAD_TOP = _planar(((.35, 0), (.18, -.24), (-.3, -.24), (-.38, 0), (-.3, .24), (.18, .24)), axis=2)
QUAD_SIDE = _planar(((.38, -.08), (-.38, -.08), (-.38, .12), (.3, .12)), axis=1)
QUAD_ARMS = (np.array([[.85, 0., 0.], [-.85, 0., 0.]]), np.array([[0., .85, 0.], [0., -.85, 0.]]))

SHAPES = {
    'UAV_fixedwing': dict(polygons=(FIXEDWING_TOP, FIXEDWING_SIDE), lines=(), rings=(), nose=(1.4, 0., 0.)),
    'UAV_quadrotor': dict(polygons=(QUAD_TOP, QUAD_SIDE), lines=QUAD_ARMS,
                          rings=tuple(_ring(center, .32) for center in QUAD_ROTORS), nose=QUAD_ROTORS[0]),
}
ROLE_BADGES = {'disturb': 'D', 'scout': 'S'}


def attitude_matrix(entity):
    """FLU-to-ENU rotation from ``attitude`` (or ``rigid_state``), or None when unknown."""
    rigid = entity.get('rigid_state')
    for value in (entity.get('attitude'), None if rigid is None else rigid[6:10]):
        if value is None:
            continue
        q = np.asarray(value, dtype=float)
        if q.shape == (4,) and np.isfinite(q).all() and np.linalg.norm(q) > 1e-12:
            return quaternion_matrix(q)
    return None


def role_badge(entity):
    """Short role suffix; attackers, the default role, get none."""
    return ROLE_BADGES.get(str(entity.get('role', entity.get('type', ''))).lower(), '')


def aircraft_geometry(entity, projection='xy', size=12., center=(0., 0.)):
    """Screen-space polygons, polylines and nose point of a UAV glyph.

    Screen y grows downward, so world +y (XY panel) and +z (XZ panel) map to -y.
    Returns None when the attitude is unknown.
    """
    rotation = attitude_matrix(entity)
    shape = SHAPES.get(entity.get('env_agent_type'))
    if rotation is None or shape is None:
        return None
    axis = 1 if projection == 'xy' else 2

    def project(points):
        world = np.asarray(points, dtype=float) @ rotation.T
        return [(center[0] + size*x, center[1] - size*y) for x, y in world[:, [0, axis]]]

    return dict(polygons=[project(points) for points in shape['polygons']],
                lines=[project(points) for points in (*shape['lines'], *shape['rings'])],
                closed=[False]*len(shape['lines']) + [True]*len(shape['rings']),
                nose=project([shape['nose']])[0])


def particle_triangle(center, velocity, size, projection='xy', apex_degrees=45.):
    """Isosceles triangle pointing along the projected velocity (scout/disturb particles)."""
    axis = 1 if projection == 'xy' else 2
    angle = -math.atan2(float(velocity[axis]), float(velocity[0]))
    half_base = 2. * size * math.tan(math.radians(apex_degrees / 2))
    points = ((size, 0.), (-size, -half_base), (-size, half_base))
    c, s = math.cos(angle), math.sin(angle)
    return [(center[0] + x*c - y*s, center[1] + x*s + y*c) for x, y in points]
