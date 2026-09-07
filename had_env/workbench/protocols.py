"""Versioned scenario descriptions; observation and action semantics stay explicit."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math

from had_env.grouping.protocol import EpisodeSpec, VERSION, stable_seed

CUSTOM_PROTOCOL = "had-grouping-scenario-v1"
DEFAULT_TARGETS = ((-2100., -650., 100.), (-2100., 650., 100.))


@dataclass(frozen=True)
class ProtocolSpec:
    protocol_id: str = VERSION
    action_space: str = "red_grouping_with_reserve"
    lower_action_space: str = "27_acceleration_primitives"
    firing_control: str = "native_rule"
    observation: str = "public_physical_state_without_pending_blue_action_or_future_rng"
    executor: str = "rule_group_v1"
    reward: str = "native_terminal_success_0_1"
    gamma: float = 1.
    horizon_semantics: str = "task_terminal_defender_success"
    decision_clock: str = "absolute_periodic_or_casualty"

    def __post_init__(self):
        # These are descriptors of existing tasks, not editable simulation knobs.
        if self.protocol_id not in (VERSION, CUSTOM_PROTOCOL):
            raise ValueError("Unsupported workbench protocol")
        defaults = type(self).__dataclass_fields__
        if any(getattr(self, name) != field.default for name, field in defaults.items()
               if name != "protocol_id"):
            raise ValueError("Changing control, observation or reward requires a new registered protocol")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ScenarioSpec:
    red_count: int = 8
    blue_count: int = 8
    opening_seed: int = 20260907
    opponent_seed: int = 20260908
    split: str = "test"
    distribution: str = "id"
    scenario_id: str = ""
    protocol_id: str = VERSION
    opponent: str = "reactive"
    max_steps: int = 50
    command_interval: int = 5
    target_positions: tuple[tuple[float, float, float], ...] = DEFAULT_TARGETS

    def __post_init__(self):
        for name in ("red_count", "blue_count", "opening_seed", "opponent_seed", "max_steps", "command_interval"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value:
                raise ValueError(f"{name} must be an integer")
            object.__setattr__(self, name, int(value))
        if self.red_count < 1 or self.blue_count < 1:
            raise ValueError("Both rosters must be positive")
        if min(self.opening_seed, self.opponent_seed) < 0:
            raise ValueError("Seeds must be nonnegative")
        if self.split not in ("train", "validation", "test", "diagnostic"):
            raise ValueError("split must be train, validation, test or diagnostic")
        if self.distribution not in ("id", "ood"):
            raise ValueError("distribution must be id or ood")
        if self.split == "train" and self.distribution == "ood":
            raise ValueError("OOD is an evaluation designation, not a training split")
        ProtocolSpec(self.protocol_id)
        if self.opponent not in ("reactive", "balanced", "concentrated"):
            raise ValueError("Unsupported frozen opponent")
        if not 1 <= self.max_steps <= 50 or not 1 <= self.command_interval <= 50:
            raise ValueError("Physical horizon and command interval must be in 1..50")
        points = tuple(tuple(float(x) for x in point) for point in self.target_positions)
        from had_env.core.config import AeroPoint
        if not points or any(len(p) != 3 or any(not math.isfinite(x) or not low <= x <= high
                             for x, (low, high) in zip(p, AeroPoint)) for p in points):
            raise ValueError("Targets must have finite xyz positions inside the native world")
        object.__setattr__(self, "target_positions", points)
        if self.protocol_id == VERSION and (self.opponent != "reactive" or self.max_steps != 50
                or self.command_interval != 5 or points != DEFAULT_TARGETS):
            raise ValueError(f"Changed scenario settings require protocol_id={CUSTOM_PROTOCOL!r}")
        if not self.scenario_id:
            values = asdict(self)
            values.pop("scenario_id")
            key = hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()[:20]
            object.__setattr__(self, "scenario_id", f"{self.split}:{self.distribution}:{key}")

    @classmethod
    def from_episode_spec(cls, spec: EpisodeSpec | dict, *, distribution="id"):
        spec = EpisodeSpec.from_dict(spec) if isinstance(spec, dict) else spec
        return cls(red_count=spec.red_count, blue_count=spec.blue_count, opening_seed=spec.opening_seed,
                   opponent_seed=spec.opponent_seed, split=spec.split, distribution=distribution,
                   scenario_id=spec.family_id, protocol_id=spec.protocol_version, opponent=spec.opponent)

    @classmethod
    def from_dict(cls, value):
        return cls(**value)

    def to_dict(self):
        value = asdict(self)
        value["target_positions"] = [list(p) for p in self.target_positions]
        return value

    def episode_spec(self):
        if self.protocol_id != VERSION:
            raise ValueError("Custom scenarios cannot be relabelled as the frozen v5.1 benchmark")
        return EpisodeSpec(self.scenario_id, self.red_count, self.blue_count, self.opening_seed,
                           self.opponent_seed, self.split, self.opponent, self.protocol_id)


def scenario_family(seed, index, *, split="test", distribution="id", cells=((8, 8),)):
    """Scheduling-independent openings, with explicitly separate split domains."""
    if not cells or index < 0:
        raise ValueError("Provide cells and a nonnegative episode index")
    red, blue = cells[index % len(cells)]
    identity = [int(seed), int(index), split, distribution, int(red), int(blue)]
    return ScenarioSpec(red_count=red, blue_count=blue, split=split, distribution=distribution,
                        opening_seed=stable_seed("workbench-opening", identity),
                        opponent_seed=stable_seed("workbench-opponent", identity))
