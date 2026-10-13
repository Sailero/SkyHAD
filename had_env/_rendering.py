"""Optional rendering capability and delegation; no graphics dependencies."""
from importlib import import_module
from pathlib import Path


def has_native_renderer():
    return Path(__file__).with_name('render').joinpath('backend.py').is_file()


def native_backend():
    if not has_native_renderer():
        raise RuntimeError('Rendering is unavailable in the core edition; use the full main branch.')
    return import_module('had_env.render.backend')
