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

# Motion: particle limits, timestep and world/plane geometry.
vDomain = [35., 120.]
aMax = 40.
BlueAmaxCoef = 1.0
BlueVmaxCoef = 1.0
wMax = np.pi  # Maximum heading change per physical step.
Boundary = True
Interval = 1
AeroPoint = [[-2500., 2500.], [-2500., 2500.], [0., 2500.]]
EnvDim = len(AeroPoint)  # Coordinate slots stay 3 even for planar motion.
PlanarAltitude = 100.
PlanarSpawnSeparation = 30.

# Uniform placement fractions measured from each team's x boundary.
AttackRatio = 0.3
DisturbRatio = 0.2
ScoutRatio = 0.1

# Combat: target HP, full/zero-damage radii and scout/disturber sectors.
initial_health = 2.0
AttackDistance = {
    2: [200., 400.],
    3: [300., 600.],
}
AttackIntensity = 1.0


def attack_distance_for(spatial_dim=2):
    """Return validated full-damage and zero-damage radii for a dimension."""
    if isinstance(spatial_dim, bool) or spatial_dim not in (2, 3):
        raise ValueError("spatial_dim must be 2 or 3")
    inner, outer = map(float, AttackDistance[spatial_dim])
    if not np.isfinite([inner, outer]).all() or not 0 < inner < outer:
        raise ValueError("AttackDistance must satisfy 0 < full-damage radius < zero-damage radius")
    return inner, outer


DisturbAngleMax = np.pi / 3.
DisturbDistanceMax = 600.
DisturbIntensity = 0.1
GaussSigma = 1.
ScoutAngleMax = np.pi * 2 / 3.
ScoutDistanceMax = 2000.

# Survival rewards; damage mode returns raw new target damage instead.
HorizonPolicy = "red_win"
reward_disturb_single = 0.
reward_scout_single = 0.
reward_attack_single = 0.
reward_boundary = 0.05
reward_episode = 10
OBS_ENTITY_DIM = 11  # Position 3, velocity 3, health, alive and side flags 3.

# Optional native renderer: pixel dimensions and RGBA colours.
ScreenLength = 800
ScreenWidth = ScreenLength * (AeroPoint[1][1] - AeroPoint[1][0]) / (AeroPoint[0][1] - AeroPoint[0][0])
ScreenHeight = int(ScreenLength * (AeroPoint[2][1] - AeroPoint[2][0]) / (AeroPoint[0][1] - AeroPoint[0][0]) * 0.5)
SurfaceColor = (238, 243, 248, 255)
BorderColor = (130, 149, 170, 255)
RedColor = (158, 41, 50, 255)
BlueColor = (24, 126, 183, 255)
DeadAgentColor = (104, 115, 131, 180)


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


def _bounds(value, default, name):
    a = np.asarray(default if value is None else value, dtype=float)
    if a.shape != (3, 2) or not np.isfinite(a).all() or np.any(a[:, 0] > a[:, 1]):
        raise ValueError(f"{name} must contain finite ordered x/y/z bounds")
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
        for name, default in (
            ("world_bounds", AeroPoint),
            ("target_region", [[-2300, -1900], [-1200, 1200], [500, 1500]])):
            object.__setattr__(self, name, _bounds(getattr(self, name), np.asarray(default)*scale, name))
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
PHYSICS_PROTOCOL = "skyhad-v3-rigid-body-task-start-combat"
