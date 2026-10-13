"""Small stateless controller and rotation utilities (radians, right handed)."""
import numpy as np


def clip_norm(vector, limit):
    vector = np.asarray(vector, dtype=np.float64)
    length = np.linalg.norm(vector)
    return vector * (min(1., float(limit) / length) if length > 0. else 1.)


def wrap_angle(angle):
    return (angle + np.pi) % (2. * np.pi) - np.pi


def quaternion_matrix(quaternion):
    """Body FLU to world ENU matrix from scalar-first quaternion."""
    q = np.asarray(quaternion, dtype=np.float64)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])


def quaternion_from_euler(roll=0., pitch=0., yaw=0.):
    """Intrinsic XYZ roll/pitch/yaw; positive pitch points the nose down."""
    cr, sr = np.cos(roll/2), np.sin(roll/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cy, sy = np.cos(yaw/2), np.sin(yaw/2)
    return np.array([cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
                     cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy])


def flight_angles(quaternion):
    """Physical bank (right wing down), nose-up pitch and ENU heading."""
    matrix = quaternion_matrix(quaternion)
    return (np.arctan2(matrix[2, 1], matrix[2, 2]),
            np.arcsin(np.clip(matrix[2, 0], -1., 1.)),
            np.arctan2(matrix[1, 0], matrix[0, 0]))


def vee(matrix):
    return np.array([matrix[2, 1], matrix[0, 2], matrix[1, 0]])
