"""Portable, strictly typed HAD branch points; never deserialize Python executables."""
from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass
import hashlib
import json
import math

import numpy as np

from had_env.grouping.adapter import HADStage3Event, HADStage3Snapshot
from had_env.grouping.domain import Group, Grouping
from .identity import assert_behavior_compatible, source_identity

SCHEMA = "had-workbench-snapshot-v1"
_CLASSES = {cls.__name__: cls for cls in (HADStage3Snapshot, HADStage3Event, Grouping, Group)}
_SNAPSHOT_KEYS = {"physical", "executor", "previous", "blue_grouping", "opponent_rng", "numpy_state",
                  "done", "opponent", "max_steps", "command_interval", "seed", "group_max_size", "executor_type"}
_ENTITY_KEYS = {"position", "initial_position", "velocity", "Health", "Id", "render_id", "Color", "Type",
                "acceleration", "vMax", "vMin", "wMax", "aMax", "Boundary", "IsFire", "is_active_objective"}


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _encode(value):
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biuf":
            raise ValueError("Only real numeric arrays are supported in snapshots")
        return {"kind": "array", "dtype": str(value.dtype), "shape": list(value.shape), "data": value.tolist()}
    if isinstance(value, np.generic):
        if value.dtype.kind not in "biuf":
            raise ValueError("Unsupported NumPy scalar")
        return {"kind": "scalar", "dtype": str(value.dtype), "data": value.item()}
    if is_dataclass(value):
        if type(value).__name__ not in _CLASSES or type(value) is not _CLASSES[type(value).__name__]:
            raise ValueError("Unregistered snapshot dataclass")
        return {"kind": type(value).__name__, "fields": {f.name: _encode(getattr(value, f.name)) for f in fields(value)}}
    if isinstance(value, dict):
        if any(type(key) not in (str, int) for key in value):
            raise ValueError("Snapshot mapping keys must be strings or integers")
        return {"kind": "mapping", "items": [[key, _encode(item)] for key, item in value.items()]}
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": [_encode(item) for item in value]}
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError(f"Unsupported snapshot value type: {type(value).__name__}")


def _decode(value, depth=0):
    if depth > 40:
        raise ValueError("Snapshot nesting exceeds the registered structure")
    if isinstance(value, list):
        return [_decode(item, depth+1) for item in value]
    if not isinstance(value, dict):
        if value is None or type(value) in (str, bool, int) or (type(value) is float and math.isfinite(value)):
            return value
        raise ValueError("Invalid scalar in snapshot")
    kind = value.get("kind")
    if kind in ("array", "scalar"):
        expected = {"kind", "dtype", "data"} | ({"shape"} if kind == "array" else set())
        if set(value) != expected:
            raise ValueError("Invalid numeric snapshot fields")
        dtype = np.dtype(value["dtype"])
        if dtype.kind not in "biuf" or dtype.itemsize > 8:
            raise ValueError("Unregistered snapshot array dtype")
        array = np.asarray(value["data"], dtype=dtype)
        if array.size > 1_000_000 or not np.isfinite(array).all():
            raise ValueError("Snapshot array is too large or non-finite")
        if kind == "scalar":
            if array.shape != ():
                raise ValueError("Snapshot scalar must have scalar shape")
            return array[()]
        if not isinstance(value["shape"], list) or any(type(n) is not int or n < 0 for n in value["shape"]):
            raise ValueError("Invalid array shape")
        if tuple(value["shape"]) != array.shape:
            raise ValueError("Snapshot array shape differs from its data")
        return array
    if kind == "tuple":
        if set(value) != {"kind", "items"} or not isinstance(value["items"], list):
            raise ValueError("Invalid tuple encoding")
        return tuple(_decode(item, depth+1) for item in value["items"])
    if kind == "mapping":
        if set(value) != {"kind", "items"} or not isinstance(value["items"], list):
            raise ValueError("Invalid mapping encoding")
        result = {}
        for row in value["items"]:
            if not isinstance(row, list) or len(row) != 2 or type(row[0]) not in (str, int) or row[0] in result:
                raise ValueError("Invalid or duplicate snapshot mapping key")
            result[row[0]] = _decode(row[1], depth+1)
        return result
    if kind in _CLASSES:
        cls = _CLASSES[kind]
        if set(value) != {"kind", "fields"} or not isinstance(value["fields"], dict):
            raise ValueError("Invalid registered object encoding")
        if set(value["fields"]) != {field.name for field in fields(cls)}:
            raise ValueError("Registered snapshot object fields differ")
        return cls(**{key: _decode(item, depth+1) for key, item in value["fields"].items()})
    raise ValueError(f"Unregistered snapshot object kind: {kind!r}")


def _validate(snapshot):
    if not isinstance(snapshot, dict) or set(snapshot) != _SNAPSHOT_KEYS:
        raise ValueError("Snapshot environment fields differ from the registered rule task")
    physical = snapshot["physical"]
    if not isinstance(physical, HADStage3Snapshot) or not isinstance(snapshot["previous"], Grouping) or not isinstance(snapshot["blue_grouping"], Grouping):
        raise ValueError("Snapshot object types do not match the registered task")
    if snapshot["executor_type"] != "RuleExecutor" or snapshot["group_max_size"] is not None:
        raise ValueError("Only the unbounded registered rule executor is supported")
    if type(snapshot["done"]) is not bool or type(physical.step_count) is not int or not 0 <= physical.step_count <= snapshot["max_steps"] <= 50:
        raise ValueError("Invalid snapshot time or terminal flag")
    if not 1 <= snapshot["command_interval"] <= 50 or snapshot["opponent"] not in ("reactive", "balanced", "concentrated"):
        raise ValueError("Invalid snapshot protocol")
    red = [i for i, _ in physical.red_assignment]
    blue = [i for i, _ in physical.blue_assignment]
    targets = tuple(physical.active_target_ids)
    if not red or not blue or not targets or targets != tuple(range(len(targets))):
        raise ValueError("Invalid snapshot roster or target identifiers")
    if len(set(red + blue)) != len(red + blue) or len(physical.entity_states) != len(red)+len(blue)+len(targets):
        raise ValueError("Snapshot entity count differs from assignments")
    for state in physical.entity_states:
        if not isinstance(state, dict) or not set(state) <= _ENTITY_KEYS:
            raise ValueError("Snapshot contains unregistered entity attributes")
        if not {"Id", "position", "velocity", "Health", "Color", "Type"} <= set(state):
            raise ValueError("Snapshot omits required physical entity fields")
        for key in ("position", "velocity"):
            value = np.asarray(state[key], float)
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError("Entity position/velocity must be a finite xyz vector")
        if not np.isfinite(float(state["Health"])):
            raise ValueError("Entity health must be finite")
    ordered_ids = [state["Id"] for state in physical.entity_states]
    if len(set(ordered_ids)) != len(ordered_ids) or ordered_ids[:len(red)+len(blue)] != red + blue:
        raise ValueError("Snapshot roster ordering is inconsistent")
    live_red = [state["Id"] for state in physical.entity_states[:len(red)] if state["Health"] > 0]
    live_blue = [state["Id"] for state in physical.entity_states[len(red):len(red)+len(blue)] if state["Health"] > 0]
    snapshot["previous"].validate(live_red, targets, max_members=None)
    if physical.step_count != 0 or snapshot["blue_grouping"].groups or snapshot["blue_grouping"].reserve:
        snapshot["blue_grouping"].validate(live_blue, targets, max_members=None)
    for state in (physical.environment_rng_state, physical.adapter_rng_state, snapshot["opponent_rng"]):
        generator = np.random.default_rng()
        generator.bit_generator.state = copy.deepcopy(state)
    for state in (physical.numpy_random_state, snapshot["numpy_state"]):
        generator = np.random.RandomState()
        generator.set_state(state)
    return snapshot


def encode_snapshot(snapshot, *, source=None):
    payload = _encode(snapshot)
    return dict(schema=SCHEMA, source_identity=copy.deepcopy(source or source_identity()),
                payload_hash=_hash(payload), payload=payload)


def decode_snapshot(document, *, current_identity=None):
    """Reject incompatible sources or malformed state before constructing an environment."""
    if not isinstance(document, dict) or set(document) != {"schema", "source_identity", "payload_hash", "payload"} or document["schema"] != SCHEMA:
        raise ValueError("Unsupported portable snapshot schema")
    identity = document["source_identity"]
    if not isinstance(identity, dict) or not isinstance(identity.get("behavior_hash"), str):
        raise ValueError("Snapshot is missing its behavior identity")
    assert_behavior_compatible(identity, current=current_identity)
    if _hash(document["payload"]) != document["payload_hash"]:
        raise ValueError("Snapshot content hash mismatch")
    try:
        return _validate(_decode(document["payload"]))
    except (KeyError, TypeError, OverflowError) as error:
        raise ValueError("Malformed registered snapshot") from error
