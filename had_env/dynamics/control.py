"""Rotation and saturation helpers (radians, right-handed, scalar-first quaternions).

Every helper accepts one vector/quaternion or a batch with a leading axis.
"""
import numpy as np


def clip_norm(vector, limit):
    """Scale each vector down to at most ``limit`` Euclidean length."""
    vector = np.asarray(vector, dtype=np.float64)
    length = np.linalg.norm(vector, axis=-1, keepdims=True)
    return vector * np.minimum(1., limit / np.where(length > 0., length, 1.))


def cross(a, b):
    """Row-wise cross product; same arithmetic as ``np.cross`` without its axis handling."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    result = np.empty(np.broadcast_shapes(a.shape, b.shape))
    result[..., 0] = a[..., 1]*b[..., 2] - a[..., 2]*b[..., 1]
    result[..., 1] = a[..., 2]*b[..., 0] - a[..., 0]*b[..., 2]
    result[..., 2] = a[..., 0]*b[..., 1] - a[..., 1]*b[..., 0]
    return result


def wrap_angle(angle):
    return (angle + np.pi) % (2. * np.pi) - np.pi


def quaternion_matrix(quaternion):
    """Body FLU to world ENU rotation matrix from a scalar-first quaternion."""
    q = np.asarray(quaternion, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    matrix = np.empty(q.shape[:-1] + (3, 3))
    matrix[..., 0, 0] = 1 - 2*(y*y+z*z)
    matrix[..., 0, 1] = 2*(x*y-w*z)
    matrix[..., 0, 2] = 2*(x*z+w*y)
    matrix[..., 1, 0] = 2*(x*y+w*z)
    matrix[..., 1, 1] = 1-2*(x*x+z*z)
    matrix[..., 1, 2] = 2*(y*z-w*x)
    matrix[..., 2, 0] = 2*(x*z-w*y)
    matrix[..., 2, 1] = 2*(y*z+w*x)
    matrix[..., 2, 2] = 1-2*(x*x+y*y)
    return matrix


def quaternion_from_euler(roll=0., pitch=0., yaw=0.):
    """Quaternion of R = Rz(yaw) Ry(pitch) Rx(roll); positive pitch points the FLU nose down."""
    cr, sr = np.cos(roll/2), np.sin(roll/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cy, sy = np.cos(yaw/2), np.sin(yaw/2)
    return np.array([cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
                     cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy])


def matrix_angles(matrix):
    """Physical bank (right wing down), nose-up pitch and ENU heading of FLU-to-ENU matrices."""
    return (np.arctan2(matrix[..., 2, 1], matrix[..., 2, 2]),
            np.arcsin(np.clip(matrix[..., 2, 0], -1., 1.)),
            np.arctan2(matrix[..., 1, 0], matrix[..., 0, 0]))


def flight_angles(quaternion):
    """Physical bank (right wing down), nose-up pitch and ENU heading."""
    return matrix_angles(quaternion_matrix(quaternion))


def vee(matrix):
    matrix = np.asarray(matrix)
    return np.stack([matrix[..., 2, 1], matrix[..., 0, 2], matrix[..., 1, 0]], axis=-1)
