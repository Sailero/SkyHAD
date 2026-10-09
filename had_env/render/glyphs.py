"""Small vector glyphs shared by the native and scientific viewers."""
from __future__ import annotations

import math
import numpy as np
from had_env.dynamics.control import flight_angles, quaternion_matrix


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


AIRPLANE_SIDE = ((1.4, 0), (.9, -.17), (-.75, -.17), (-1.05, -.65),
                 (-1.25, -.65), (-1.1, .13), (.85, .13))
QUAD_BODY = ((.5, 0), (.15, -.24), (-.3, -.24), (-.4, 0), (-.3, .24), (.15, .24))
QUAD_SIDE = ((.55, -.12), (-.5, -.12), (-.5, .2), (.35, .2), (.55, .08))
ROLE_LEGEND = "A attack  D disturb  S scout | R bank / P nose-up | dot/cross: nose out/in"


def attitude_cues(entity):
    """Read physical bank/nose-up angles; velocity cannot supply attitude."""
    quaternion = entity.get("attitude")
    rigid = entity.get("rigid_state")
    candidates = (quaternion, rigid[6:10] if rigid is not None else None)
    for value in candidates:
        if value is None:
            continue
        q = np.asarray(value, dtype=float)
        if q.shape == (4,) and np.isfinite(q).all() and np.linalg.norm(q) > 1e-12:
            q = q / np.linalg.norm(q)
            bank, pitch, _ = flight_angles(q)
            return dict(forward=tuple(quaternion_matrix(q)[:, 0]), bank=float(bank),
                        nose_up_pitch=float(pitch), valid_attitude=True)
    velocity = entity.get("velocity")
    velocity = np.asarray((0., 0., 0.) if velocity is None else velocity, dtype=float)
    length = np.linalg.norm(velocity)
    forward = velocity / length if length > 0 else np.array([1., 0., 0.])
    return dict(forward=tuple(forward), bank=None, nose_up_pitch=None, valid_attitude=False)


def role_badge(entity):
    """Compact role label for aircraft; particle role shapes stay unchanged."""
    role = str(entity.get("role", entity.get("type", "unknown"))).lower()
    return {"attack": "A", "disturb": "D", "scout": "S"}.get(role, "?")


def attitude_label(entity):
    cues = attitude_cues(entity)
    if not cues["valid_attitude"]:
        return "R ?  P ?"
    return f"R {math.degrees(cues['bank']):+.0f}°  P {math.degrees(cues['nose_up_pitch']):+.0f}°"


def aircraft_geometry(entity, projection="xy", size=12., center=(0., 0.)):
    """Small vector primitives in screen coordinates, independent of backend.

    Silhouette rotation is projected body-forward direction, not full attitude.
    Separate screen-fixed R/P ticks carry physical bank and nose-up pitch.
    """
    cues = attitude_cues(entity)
    forward = cues["forward"]
    axis = 1 if projection == "xy" else 2
    head_on = cues["valid_attitude"] and math.hypot(forward[0], forward[axis]) < 1e-8
    angle = -math.atan2(forward[axis], forward[0]) if not head_on else 0.
    polygons, lines, circles = [], [], []
    fixedwing = entity.get("env_agent_type") == "UAV_fixedwing"
    if head_on:
        circles = [((0., 0.), .35)]
        lines = [((-.95, 0.), (.95, 0.)), ((0., -.55), (0., .55))]
    elif fixedwing:
        polygons = [AIRPLANE if projection == "xy" else AIRPLANE_SIDE]
        if projection == "xz":
            lines = [((-.45, .02), (.4, .02))]
    elif projection == "xy":
        polygons = [QUAD_BODY]
        rotors = ((-.62, -.62), (.62, -.62), (.62, .62), (-.62, .62))
        lines = [(rotors[0], rotors[2]), (rotors[1], rotors[3]), ((.35, 0), (1.15, 0))]
        circles = [(rotor, .3) for rotor in rotors]
    else:
        polygons = [QUAD_SIDE]
        lines = [((-.85, -.12), (.85, -.12)), ((-.85, -.32), (-.85, -.05)),
                 ((.85, -.32), (.85, -.05)), ((-1.15, -.32), (-.55, -.32)),
                 ((.55, -.32), (1.15, -.32)), ((-.3, .2), (-.4, .52)),
                 ((.25, .2), (.4, .52)), ((-.55, .52), (.55, .52))]
    transform = lambda points: rotate_points(points, angle, size, center)
    nose = None if head_on else transform(((1.4 if fixedwing else 1.15, 0.),))[0]
    indicators = {}
    for key, value, offset in (("bank", cues["bank"], (-6., 21.)),
                               ("pitch", cues["nose_up_pitch"], (18., 21.))):
        if value is not None:
            indicators[key] = rotate_points(((-5., 0.), (5., 0.)),
                                            value if key == "bank" else -value,
                                            center=(center[0]+offset[0], center[1]+offset[1]))
    hidden = forward[2] if projection == "xy" else -forward[1]
    return dict(polygons=[transform(p) for p in polygons], lines=[transform(p) for p in lines],
                circles=[(transform((p,))[0], radius*size) for p, radius in circles],
                nose=nose, head_on=head_on, toward=hidden > 0, attitude=cues, indicators=indicators)
