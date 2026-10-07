"""Small vector glyphs shared by the native and scientific viewers."""
from __future__ import annotations

import math
import numpy as np


AIRPLANE = ((1.4, 0), (.45, -.16), (.1, -.95), (-.2, -.95), (-.25, -.16),
            (-.9, -.13), (-1.1, -.42), (-1.25, -.42), (-1.1, 0),
            (-1.25, .42), (-1.1, .42), (-.9, .13), (-.25, .16),
            (-.2, .95), (.1, .95), (.45, .16))


def heading_angle(entity, projection="xy"):
    """Screen heading in radians, using scalar-first FLU-to-world attitude."""
    quaternion = entity.get("attitude")
    if quaternion is None and entity.get("rigid_state") is not None:
        quaternion = entity["rigid_state"][6:10]
    axis = 1 if projection == "xy" else 2
    if quaternion is not None:
        q = np.asarray(quaternion, dtype=float)
        if q.shape == (4,) and np.isfinite(q).all() and np.linalg.norm(q) > 0:
            w, x, y, z = q / np.linalg.norm(q)
            forward = (1-2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y))
            return -math.atan2(forward[axis], forward[0])
    velocity = entity.get("velocity")
    velocity = (0, 0, 0) if velocity is None else velocity
    return -math.atan2(float(velocity[axis]), float(velocity[0]))


def rotate_points(points, angle, size=1., center=(0., 0.)):
    c, s = math.cos(angle), math.sin(angle)
    return [(center[0]+size*(x*c-y*s), center[1]+size*(x*s+y*c)) for x, y in points]


def airplane_points(angle=0., size=12., center=(0., 0.)):
    return rotate_points(AIRPLANE, angle, size, center)


def rotor_centers(angle=0., size=12., center=(0., 0.)):
    return rotate_points(((-.62, -.62), (.62, -.62), (.62, .62), (-.62, .62)), angle, size, center)
