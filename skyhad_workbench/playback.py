"""Display-only interpolation; exact inspection always uses recorded frames."""
from __future__ import annotations

from bisect import bisect_right
import copy
import math
import numpy as np


def _slerp(first, second, alpha):
    a, b = np.asarray(first, float), np.asarray(second, float)
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    dot = float(np.dot(a, b))
    if dot < 0:
        b, dot = -b, -dot
    if dot > .9995:
        value = a + alpha*(b-a)
        return (value/np.linalg.norm(value)).tolist()
    angle = math.acos(min(1., dot))
    return ((math.sin((1-alpha)*angle)*a + math.sin(alpha*angle)*b)/math.sin(angle)).tolist()


def display_frame(episode, position: float, *, interpolate=True):
    frame = episode.frame_at(position)
    if not interpolate or episode.sparse or len(episode.frames) < 2:
        return frame
    steps = [f["step"] for f in episode.frames]
    index = bisect_right(steps, position)
    if index == 0 or index == len(steps) or position <= frame["step"]:
        return frame
    following = episode.frames[index]
    span = following["step"] - frame["step"]
    alpha = (position - frame["step"]) / span if span else 0.
    result = copy.deepcopy(frame)
    others = {(e["side"], e["id"]): e for e in following["entities"]}
    for entity in result["entities"]:
        other = others.get((entity["side"], entity["id"]))
        if other and entity.get("alive", True) and other.get("alive", True):
            entity["position"] = [a + alpha * (b - a)
                                  for a, b in zip(entity["position"], other["position"])]
            first = entity.get("attitude")
            second = other.get("attitude")
            if first is None and entity.get("rigid_state") is not None:
                first = entity["rigid_state"][6:10]
            if second is None and other.get("rigid_state") is not None:
                second = other["rigid_state"][6:10]
            if first is not None and second is not None:
                entity["attitude"] = _slerp(first, second, alpha)
                if entity.get("rigid_state") is not None:
                    entity["rigid_state"][:3] = entity["position"]
                    entity["rigid_state"][6:10] = entity["attitude"]
    result["display_step"] = position
    result["interpolated"] = True
    # Actions, HP, events, assignment, and the physical step stay discrete.
    return result


def next_event_step(episode, current, *, decisions=True, casualties=True):
    steps = set(d["step"] for d in episode.decisions) if decisions else set()
    if casualties:
        kinds = {"death", "collision", "agents_destroyed", "target_destroyed", "self_destruct", "terminal"}
        for frame in episode.frames:
            if any(e.get("kind") in kinds for e in frame.get("events", [])):
                steps.add(frame["step"])
    return min((s for s in steps if s > current), default=episode.max_step)
