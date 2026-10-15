from scipy.spatial.transform import Rotation
from had_env.config import (
    AeroPoint,
    AttackIntensity,
    Boundary,
    DisturbAngleMax,
    DisturbDistanceMax,
    DisturbIntensity,
    EnvDim,
    GaussSigma,
    attack_distance_for,
    reward_boundary,
    wMax,
)
import numpy as np


def get_area_point(area_vector, ratio):
    """Interpolate a position within an ordered interval at ratio in [0, 1]."""
    assert 0 <= ratio <= 1
    return area_vector[0] + (area_vector[1] - area_vector[0]) * ratio


def distance(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    return np.sqrt(np.sum((p1 - p2) ** 2))


def distances_from(position, positions):
    """Batch the same subtraction/square/sum/sqrt used by ``distance``.

    Homogeneous world coordinates share one small NumPy reduction. Mixed
    coordinate dtypes or shapes retain the scalar path so batching cannot
    silently promote a previously lower-precision pair or alter broadcasting.
    """
    points = [np.asarray(point) for point in positions]
    if not points:
        return np.empty(0, dtype=np.float64)
    first = points[0]
    observer = np.asarray(position)
    if observer.ndim != 1 or first.ndim != 1 or any(point.dtype != first.dtype or point.shape != first.shape
                              for point in points[1:]):
        return np.asarray([distance(position, point) for point in points])
    offsets = observer - np.asarray(points)
    return np.sqrt(np.sum(offsets ** 2, axis=1))


def vectors_to_angle(vector1, vector2):

    if distance(vector1, np.zeros(vector1.shape)) < 1e-3 or distance(vector2, np.zeros(vector2.shape)) < 1e-3:
        return 0.0

    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)

    dot_product = np.dot(vector1, vector2)

    cos_theta = dot_product / (norm1 * norm2)
    cos_theta = np.clip(cos_theta, -1, 1)

    theta = np.arccos(cos_theta)
    return np.abs(theta)


def attack_intensity_ratio(d, attack_distance=None):
    inner, outer = attack_distance_for(2) if attack_distance is None else attack_distance
    if isinstance(d, float):
        if d < inner:
            return 1
        elif d >= outer:
            return 0
        else:
            return (outer - d) / (outer - inner)
    elif isinstance(d, (list, np.ndarray)):
        result = []
        for element in d:
            if element < inner:
                result.append(1)
            elif element >= outer:
                result.append(0)
            else:
                result.append((outer - element) / (outer - inner))
        return np.array(result)
    else:
        raise ValueError("Input should be a float, list or numpy array.")


def within_sector_area(position3d1, position3d2, velocity3d, AngleMax, DistMax):

    position3d1 = np.array(position3d1)
    position3d2 = np.array(position3d2)
    velocity3d = np.array(velocity3d)
    position3d = position3d2 - position3d1

    theta = vectors_to_angle(position3d, velocity3d)
    dist = distance(position3d1, position3d2)
    return (theta < AngleMax / 2) * (dist < DistMax)


def disturb_intensity_ratio(position3d1, position3d2, velocity3d, scene_scale=1.):
    """Gaussian jamming strength inside the disturber's forward sector.

    Distance is measured in units of the sector radius, so the strength falls
    from 0.399 at point blank to 0.242 at the sector edge for GaussSigma = 1.
    """
    dist = distance(position3d1, position3d2)
    radius = DisturbDistanceMax * scene_scale
    IsDisturbed = within_sector_area(position3d1, position3d2, velocity3d, DisturbAngleMax, radius) * (dist > 1e-3)
    return IsDisturbed * np.exp(-.5 * (dist / radius / GaussSigma) ** 2) / (GaussSigma * np.sqrt(2. * np.pi))


def rotate_vector(p, axis, angle):

    axis = axis / np.linalg.norm(axis)
    r = Rotation.from_rotvec(angle * axis)
    pnew = r.apply(p)
    return pnew


def _orthogonal_axis(vector):
    """Deterministic unit axis orthogonal to ``vector`` via Gram-Schmidt."""
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm < 1e-9:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    unit = value / norm
    helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(unit[0]) > 0.9:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    axis = np.cross(unit, helper)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-9:
        axis = np.cross(unit, np.array([0.0, 0.0, 1.0], dtype=np.float64))
        axis_norm = float(np.linalg.norm(axis))
    return axis / max(axis_norm, 1e-12)


def rotate_restrict_velocity(v1, v2):

    if np.linalg.norm(v1) * np.linalg.norm(v2) <= 1e-3:
        angle = 0
    else:
        angle = np.abs(np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1)))

    if angle > wMax:
        axis = np.cross(v1, v2).astype(float)
        if np.linalg.norm(axis) < 1e-3:
            axis = _orthogonal_axis(v1)
        angle = wMax

        v2 = rotate_vector(v1, axis, angle) / np.linalg.norm(v1) * np.linalg.norm(v2)
    return v2.tolist()


def swept_separations(tracks):
    """Pairwise minimum simultaneous distance between piecewise-linear tracks.

    ``tracks`` has shape (agents, samples, 3) on a shared clock. Each segment
    pair uses one common time parameter in [0, 1]; independent parameters would
    falsely collide agents that pass the same point at different times.
    Returns a symmetric (agents, agents) matrix with an infinite diagonal.
    """
    tracks = np.asarray(tracks, dtype=np.float64)
    count = len(tracks)
    first, second = np.tril_indices(count, -1)
    start = tracks[:, :-1]
    motion = tracks[:, 1:] - start
    offset = start[first] - start[second]
    relative = motion[first] - motion[second]
    speed2 = np.einsum("pkd,pkd->pk", relative, relative)
    moving = speed2 > 1e-12
    time = np.where(moving, np.clip(-np.einsum("pkd,pkd->pk", offset, relative)
                                    / np.where(moving, speed2, 1.), 0., 1.), 0.)
    closest = offset + time[..., None] * relative
    result = np.full((count, count), np.inf)
    result[first, second] = result[second, first] = np.sqrt(np.einsum("pkd,pkd->pk", closest, closest)).min(axis=1)
    return result


def world_diagonal(bounds=None):
    bounds = AeroPoint if bounds is None else bounds
    return float(np.sqrt(sum((bounds[dim][1] - bounds[dim][0]) ** 2 for dim in range(EnvDim))))


class WorldKinematics:
    """Lightweight pre-update snapshot used instead of deepcopy(world)."""

    __slots__ = ("positions", "velocities", "health", "is_fire", "is_disturb",
                 "color_code", "type_code", "attack_distance", "scene_scale")

    def __init__(self, world, color_code=None, type_code=None, *, attack_distance=None):
        self.attack_distance = tuple(attack_distance if attack_distance is not None else
                                     world[0].attack_distance if world else attack_distance_for(2))
        self.scene_scale = getattr(world[0], "scene_scale", 1.) if world else 1.
        self.positions = np.asarray([entity.position for entity in world], dtype=np.float64)
        self.velocities = np.asarray([entity.velocity for entity in world], dtype=np.float64)
        self.health = np.asarray([float(entity.Health) for entity in world], dtype=np.float64)
        self.is_fire = np.asarray([bool(getattr(entity, "IsFire", False)) for entity in world], dtype=bool)
        self.is_disturb = np.asarray([bool(getattr(entity, "IsDisturb", False)) for entity in world], dtype=bool)
        if color_code is None:
            mapping = {"Red": 0, "Blue": 1, "Entity": 2}
            color_code = np.asarray([mapping[entity.Color] for entity in world], dtype=np.int8)
        if type_code is None:
            mapping = {"Entity": 0, "Attack": 1, "Disturb": 2, "Scout": 3}
            type_code = np.asarray([mapping[entity.Type] for entity in world], dtype=np.int8)
        self.color_code = color_code
        self.type_code = type_code

    def attack_loss(self, victim_pos, victim_color, victim_type):
        if victim_type == "Entity":
            mask = (self.type_code == 1) & (self.color_code == 1) & (self.health > 0)
        else:
            victim_code = 0 if victim_color == "Red" else 1
            mask = (self.type_code == 1) & (self.color_code != victim_code) & (self.health > 0)
        mask &= self.is_fire
        if not np.any(mask):
            return 0.0
        distances = np.linalg.norm(self.positions[mask] - np.asarray(victim_pos, dtype=np.float64), axis=1)
        return float(AttackIntensity * np.sum(attack_intensity_ratio(distances, self.attack_distance)))

    def disturb_loss(self, victim_pos):
        mask = (self.type_code == 2) & (self.health > 0) & self.is_disturb
        if not np.any(mask):
            return 0.0
        total = 0.0
        victim = np.asarray(victim_pos, dtype=np.float64)
        for position, velocity in zip(self.positions[mask], self.velocities[mask]):
            total += disturb_intensity_ratio(position, victim, velocity, self.scene_scale)
        return float(DisturbIntensity * total)


def boundary_loss(position, boundary_range=100, bounds=None, scene_scale=1.):
    """Scout/Disturb shaping penalty inside a ``boundary_range`` band along the x/y walls."""
    bounds = AeroPoint if bounds is None else bounds
    if not Boundary:
        return 0
    band = boundary_range * scene_scale
    min_distance = min(min(abs(position[0] - bounds[0][0]), abs(position[0] - bounds[0][1])),
                       min(abs(position[1] - bounds[1][0]), abs(position[1] - bounds[1][1])))
    return reward_boundary * np.exp(-min_distance / band) if min_distance < band else 0
