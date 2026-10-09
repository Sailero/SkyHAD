"""Viewer provenance is independent of the scientific compatibility identity.

Legacy recordings stay readable. A saved simulation can continue only when its
behavior identity matches the installed simulator and workbench sources exactly.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess

import had_env

ROOT = Path(had_env.__file__).resolve().parent.parent
VIEWER_FILES = {"gui.py", "views.py", "export.py", "playback.py"}
# These methods only construct/render display data. Lifecycle and all physical
# methods remain strict. Mixed core files retain their full provenance digest.
RENDER_METHODS = {"_render_information", "_render_metadata", "render_rgb_array", "render"}
RENDER_LITERALS = {"ScreenLength", "ScreenWidth", "ScreenHeight", "SurfaceColor",
                   "BorderColor", "RedColor", "BlueColor", "DeadAgentColor"}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _behavior_source(name, content):
    """Project only the explicit presentation sections of mixed core files."""
    if name not in {"had_env/simulation.py", "had_env/config.py"}:
        return content
    tree = ast.parse(content)
    spans = []
    if name == "had_env/simulation.py":
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == "Simulation":
                spans.extend((method.lineno, method.end_lineno, method.name)
                             for method in node.body if isinstance(method, ast.FunctionDef)
                             and method.name in RENDER_METHODS)
    else:
        for node in tree.body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name) and node.targets[0].id in RENDER_LITERALS):
                try:
                    ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    continue  # Calls and expressions can have physical side effects.
                spans.append((node.lineno, node.end_lineno, node.targets[0].id))
    lines = content.splitlines(keepends=True)
    for start, end, section in sorted(spans, reverse=True):
        if name == "had_env/simulation.py":
            # Adding an explicitly registered render helper is presentation too.
            # Remove its adjacent blank separator, preserving all other text.
            while start > 1 and not lines[start - 2].strip():
                start -= 1
            lines[start - 1:end] = []
        else:
            lines[start - 1:end] = [f"# presentation section: {section}\n"]
    return "".join(lines)


def source_identity(root=None):
    explicit_root = root is not None
    root = Path(root or ROOT)
    source, behavior, viewer = {}, {}, {}
    packages = {"had_env": root / "had_env",
                "skyhad_workbench": root / "skyhad_workbench" if explicit_root else Path(__file__).resolve().parent}
    paths = [(f"{package}/{path.relative_to(directory).as_posix()}", path)
             for package, directory in packages.items() for path in directory.rglob("*.py")]
    if (root / "make_env.py").is_file():
        paths.append(("make_env.py", root / "make_env.py"))
    for name, path in sorted(paths):
        content = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
        source[name] = hashlib.sha256(content.encode()).hexdigest()
        if name.startswith("had_env/render/") or (
            name.startswith("skyhad_workbench/") and path.name in VIEWER_FILES
        ):
            viewer[name] = source[name]
        else:
            physical_content = _behavior_source(name, content)
            behavior[name] = hashlib.sha256(physical_content.encode()).hexdigest()
            if physical_content != content:
                viewer[name] = source[name]
    try:
        git_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        # A wheel in another project's venv must not claim that project's HEAD.
        commit = (subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip() if Path(git_root).resolve() == root.resolve() else "unavailable")
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    return {"git_commit": commit, "source_hash": _digest(source),
            "behavior_hash": _digest(behavior), "viewer_hash": _digest(viewer),
            "source_files": source, "identity_schema": "had-workbench-identity-v1"}


def assert_behavior_compatible(recorded, current=None):
    current = current or source_identity()
    if recorded.get("behavior_hash") != current.get("behavior_hash"):
        raise ValueError("Simulation sources differ; replay is readable, but continuation requires its recorded revision")
