"""Task geometry presets; aircraft mass, inertia and dimensions never scale."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPreset:
    scene_scale: float
    reference_speed: float
    acceleration_limit: float
    min_speed: float
    max_speed: float


PRESETS = {
    "particle": ModelPreset(1., 75., 40., 35., 120.),
    "UAV_fixedwing": ModelPreset(.4, 30., 5., 18., 30.),
    "UAV_quadrotor": ModelPreset(.16, 12., 6., 0., 20.),
}
