"""Shared defaults and immutable configuration for each simulation.

Distances use metres, speeds m/s, accelerations m/s², angles radians and time
seconds. Scene scale changes task geometry, never aircraft mass or inertia.
"""
from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
from collections.abc import Mapping

import numpy as np

from had_env import __version__

# Time and world. Geometry below is multiplied by each model's scene scale
# (particle 1, fixed wing 0.4, quadrotor 0.16) unless given explicitly.
Interval = 1                     # environment step, s (UAVs integrate 100 RK4 substeps)
AeroPoint = [[-2500., 2500.], [-2500., 2500.], [0., 2500.]]  # world box x/y/z, m
EnvDim = len(AeroPoint)          # coordinate slots stay 3 even for planar motion
Boundary = True                  # clip positions to the world box
PlanarAltitude = 100.            # fixed altitude of 2D particle worlds, m
PlanarSpawnSeparation = 30.      # minimum spawn spacing, m

# Particle motion (UAV limits live in PRESETS and the dynamics models).
vDomain = [35., 120.]            # speed range, m/s
aMax = 40.                       # acceleration limit, m/s^2
BlueAmaxCoef = 1.0               # Blue acceleration multiplier
BlueVmaxCoef = 1.0               # Blue speed multiplier
wMax = np.pi                     # maximum heading change per step, rad

# Uniform initialization: x placement fraction measured from each team's x boundary.
AttackRatio = 0.3
DisturbRatio = 0.2
ScoutRatio = 0.1

# Attack: a firing attacker self-destructs and damages every enemy within the
# zero-damage radius, linearly from 1 inside the full-damage radius.
initial_health = 2.0             # protected asset HP (aircraft have HP 1)
AttackDistance = {               # [full-damage, zero-damage] radius, m
    2: [200., 400.],
    3: [300., 600.],
}
AttackIntensity = 1.0            # damage at full strength


def attack_distance_for(spatial_dim=2):
    """Return validated full-damage and zero-damage radii for a dimension."""
    if isinstance(spatial_dim, bool) or spatial_dim not in (2, 3):
        raise ValueError("spatial_dim must be 2 or 3")
    inner, outer = map(float, AttackDistance[spatial_dim])
    if not np.isfinite([inner, outer]).all() or not 0 < inner < outer:
        raise ValueError("AttackDistance must satisfy 0 < full-damage radius < zero-damage radius")
    return inner, outer


# Disturbers jam UAVs of both teams inside a forward sector (never assets).
DisturbAngleMax = np.pi / 3.     # full sector angle, rad
DisturbDistanceMax = 600.        # sector radius, m
DisturbIntensity = 0.1           # HP per step at unit Gaussian strength
GaussSigma = 1.                  # Gaussian width in units of the sector radius
# Scouts have a functional sector only; observations are never filtered by it.
ScoutAngleMax = np.pi * 2 / 3.   # full sector angle, rad
ScoutDistanceMax = 2000.         # sector radius, m

# Survival rewards; damage mode returns raw new target damage instead.
HorizonPolicy = "red_win"        # grouping API outcome at its horizon
reward_disturb_single = 0.
reward_scout_single = 0.
reward_attack_single = 0.
reward_boundary = 0.05           # Scout/Disturb shaping penalty near x/y walls
reward_episode = 10              # terminal win/loss reward
OBS_ENTITY_DIM = 11              # position 3, velocity 3, health, alive, red/blue/target flags


@dataclass(frozen=True)
class ModelPreset:
    """Geometry scale and motion limits; speed fields use m/s."""
    scene_scale: float
    reference_speed: float
    acceleration_limit: float
    min_speed: float
    max_speed: float


PRESETS = {
    "particle": ModelPreset(1., 75., aMax, vDomain[0], vDomain[1]),
    "UAV_fixedwing": ModelPreset(.4, 30., 5., 18., 30.),
    "UAV_quadrotor": ModelPreset(.16, 12., 6., 0., 20.),
}


def _bounds(value, default, name, positive_extent=False):
    a = np.asarray(default if value is None else value, dtype=float)
    if a.shape != (3, 2) or not np.isfinite(a).all() or np.any(a[:, 0] > a[:, 1]):
        raise ValueError(f"{name} must contain finite ordered x/y/z bounds")
    if positive_extent and np.any(a[:, 0] >= a[:, 1]):
        raise ValueError(f"{name} must have a positive extent on every axis (observations are normalized by it)")
    return tuple(tuple(float(v) for v in row) for row in a)


@dataclass(frozen=True)
class EnvConfig:
    env_agent_type: str = "particle"
    env_agent_action_type: str = "acceleration"
    spatial_dim: int = 3
    task_mode: str = "survival"
    world_bounds: object = None
    target_region: object = None
    red_spawn_annulus: object = None
    blue_spawn_x: object = None
    spawn_altitude: object = None
    plane_altitude: float | None = None
    fire_range: float | None = None
    scene_scale: float | None = None

    def __post_init__(self):
        if self.env_agent_type not in PRESETS:
            raise ValueError("env_agent_type must be particle, UAV_fixedwing or UAV_quadrotor")
        if self.env_agent_action_type not in ("acceleration", "actuator", "position"):
            raise ValueError("env_agent_action_type must be acceleration, actuator or position")
        if type(self.spatial_dim) is not int or self.spatial_dim not in (2, 3):
            raise ValueError("spatial_dim must be 2 or 3")
        if self.env_agent_type != "particle" and self.spatial_dim != 3:
            raise ValueError("Six degree of freedom UAV models require spatial_dim=3")
        if self.env_agent_type == "particle" and self.env_agent_action_type == "actuator":
            raise ValueError("particle has no actuators")
        if self.task_mode not in ("survival", "damage"):
            raise ValueError("task_mode must be survival or damage")
        preset = PRESETS[self.env_agent_type]
        scale = preset.scene_scale if self.scene_scale is None else float(self.scene_scale)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("scene_scale must be positive and finite")
        object.__setattr__(self, "scene_scale", scale)
        object.__setattr__(self, "world_bounds", _bounds(self.world_bounds, np.asarray(AeroPoint)*scale,
                                                         "world_bounds", positive_extent=True))
        object.__setattr__(self, "target_region", _bounds(
            self.target_region, np.asarray([[-2300, -1900], [-1200, 1200], [500, 1500]])*scale, "target_region"))
        for name, default in (("red_spawn_annulus", [800, 2200]),
                              ("blue_spawn_x", [0, 2500]), ("spawn_altitude", [200, 1500])):
            value = np.asarray(np.asarray(default)*scale if getattr(self, name) is None else getattr(self, name), float)
            if value.shape != (2,) or not np.isfinite(value).all() or value[0] > value[1]:
                raise ValueError(f"{name} must contain two ordered finite bounds")
            object.__setattr__(self, name, tuple(map(float, value)))
        altitude = PlanarAltitude*scale if self.plane_altitude is None else float(self.plane_altitude)
        if not np.isfinite(altitude) or not self.world_bounds[2][0] <= altitude <= self.world_bounds[2][1]:
            raise ValueError("plane_altitude must be inside world height bounds")
        object.__setattr__(self, "plane_altitude", altitude)
        radius = self.attack_distance[0] if self.fire_range is None else float(self.fire_range)
        if not np.isfinite(radius) or not 0 < radius <= self.attack_distance[0]:
            raise ValueError("fire_range must be positive and within full damage radius")
        object.__setattr__(self, "fire_range", radius)

    @property
    def preset(self):
        return PRESETS[self.env_agent_type]

    @property
    def attack_distance(self):
        return tuple(v*self.scene_scale for v in attack_distance_for(self.spatial_dim))

    @property
    def collision_distance(self):
        return 20.*self.scene_scale

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_values(cls, values):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in values.items() if k in names})


def load_config(config):
    if config is None:
        return {}
    if isinstance(config, EnvConfig):
        return config.to_dict()
    if isinstance(config, (str, Path)):
        config = json.loads(Path(config).read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("config must be EnvConfig, mapping or JSON file path")
    return dict(config)


CORE_VERSION = "skyhad-" + __version__
PHYSICS_PROTOCOL = "skyhad-4.3"  # changes whenever simulated trajectories change
