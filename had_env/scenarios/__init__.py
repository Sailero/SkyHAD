"""Scenario callbacks and instance geometry presets."""
__all__ = ["Scenario"]

def __getattr__(name):
    if name == "Scenario":
        from .defense import Scenario
        return Scenario
    raise AttributeError(name)
