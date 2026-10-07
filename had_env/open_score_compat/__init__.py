"""Standalone Open-SCORE HAD entity interface and nv1 rule policies.

Adapted from the local Open-SCORE ``open_score/envs`` modules,
``open_score/utils/seeding.py`` and ``open_score/rules/coverage_rule.py``.
All physical simulation is delegated to the existing native HAD core.
"""
from .entity_env import HADEntityEnv
from .features import ENTITY_DIM, FEATURE_NAMES, N_ACTIONS
from .scales import Scale, ScaleSampler, SCALE_POOLS, TRAIN_POOL, TEST_POOL, VALIDATION_POOL
from .wrapper import (
    HADWrapper, DecisionState, Entity, Group, Grouping, PLANAR_NATIVE_IDS,
    NATIVE_TO_PLANAR, build_decision_state, native_actions_to_planar,
    EpisodeDiagnostics, trajectory_frame,
)
from .rules import CoveragePolicy, CoverageExecutor, DirectActionPolicy, RED_RULE_VERSION

PROVENANCE = "Open-SCORE open_score/envs, utils/seeding.py and rules/coverage_rule.py; native HAD physics"


def make_open_score_env(*, profile="evaluation", **kwargs):
    """Construct an ALMA-compatible HAD environment with explicit defaults.

    ``evaluation`` retains upstream entity defaults (50/50/12 slots and
    no shaping); ``training`` uses 10/10/3 slots and shaping coefficient
    1.0. Original constructor keywords override profile defaults.
    """
    if profile in ("evaluation", "eval"):
        defaults = {"pad": "eval", "shaping_coef": 0.0}
    elif profile in ("training", "train"):
        defaults = {"pad": "train", "shaping_coef": 1.0}
    else:
        raise ValueError("Open-SCORE profile must be 'evaluation' or 'training'")
    defaults.update(kwargs)
    return HADEntityEnv(**defaults)


make_entity_env = make_open_score_env

__all__ = [
    "make_open_score_env", "make_entity_env", "HADEntityEnv", "HADWrapper",
    "Scale", "ScaleSampler", "SCALE_POOLS", "TRAIN_POOL", "TEST_POOL", "VALIDATION_POOL",
    "ENTITY_DIM", "FEATURE_NAMES", "N_ACTIONS", "PLANAR_NATIVE_IDS", "NATIVE_TO_PLANAR",
    "DecisionState", "Entity", "Group", "Grouping", "build_decision_state",
    "native_actions_to_planar", "EpisodeDiagnostics", "trajectory_frame",
    "CoveragePolicy", "CoverageExecutor", "DirectActionPolicy", "RED_RULE_VERSION", "PROVENANCE",
]
