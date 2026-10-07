"""Immutable instance configuration, shared by public APIs and recordings."""
from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
from collections.abc import Mapping

import numpy as np

from had_env.scenarios.presets import PRESETS


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
            ("world_bounds", [[-2500, 2500], [-2500, 2500], [0, 2500]]),
            ("target_region", [[-2300, -1900], [-1200, 1200], [500, 1500]])):
            object.__setattr__(self, name, _bounds(getattr(self, name), np.asarray(default)*scale, name))
        for name, default in (("red_spawn_annulus", [800, 2200]),
                              ("blue_spawn_x", [0, 2500]), ("spawn_altitude", [200, 1500])):
            value = np.asarray(np.asarray(default)*scale if getattr(self, name) is None else getattr(self, name), float)
            if value.shape != (2,) or not np.isfinite(value).all() or value[0] > value[1]:
                raise ValueError(f"{name} must contain two ordered finite bounds")
            object.__setattr__(self, name, tuple(map(float, value)))
        altitude = 100.*scale if self.plane_altitude is None else float(self.plane_altitude)
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
        return tuple(v*self.scene_scale for v in ((200., 400.) if self.spatial_dim == 2 else (300., 600.)))

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
