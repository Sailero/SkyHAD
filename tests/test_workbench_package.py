"""One distribution, optional desktop dependencies, and strict legacy replay."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_merged_imports_are_headless_from_outside_checkout(tmp_path):
    code = '''import builtins, json
original = builtins.__import__
def headless(name, *args, **kwargs):
    if name.split('.')[0] in {'PySide6', 'matplotlib', 'pygame', 'torch'}:
        raise ImportError('optional dependency blocked: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = headless
import had_env, skyhad_workbench
from skyhad_workbench.session import SimulationSession
from skyhad_workbench.agent_session import FlightSession
from skyhad_workbench.protocols import ScenarioSpec, CUSTOM_PROTOCOL
with SimulationSession(ScenarioSpec(red_count=2, blue_count=2, max_steps=2,
        command_interval=1, protocol_id=CUSTOM_PROTOCOL)) as session:
    assert session.run().metadata['complete']
flight = FlightSession(dict(red_attackers=2, blue_attackers=2, max_steps=2))
try:
    assert flight.run().metadata['complete']
finally:
    flight.close()
print(json.dumps([had_env.__file__, skyhad_workbench.__file__,
                  had_env.__version__, skyhad_workbench.__version__]))
'''
    result = subprocess.run([sys.executable, '-B', '-c', code], cwd=tmp_path,
                            env=dict(os.environ, PYTHONPATH=str(ROOT)),
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    engine, tool, core_version, tool_version = json.loads(result.stdout)
    assert Path(engine).resolve().parent == ROOT / 'had_env'
    assert Path(tool).resolve().parent == ROOT / 'skyhad_workbench'
    import had_env
    assert core_version == tool_version == had_env.__version__


def test_module_cli_prints_all_commands():
    result = subprocess.run([sys.executable, '-B', '-m', 'skyhad_workbench', '--help'],
                            cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert all(command in result.stdout for command in ('record', 'view', 'branch', 'export', 'evaluate'))


def test_identity_follows_installed_packages_outside_working_directory(tmp_path, monkeypatch):
    from skyhad_workbench.identity import source_identity
    before = source_identity()
    fake = tmp_path / 'had_env'
    fake.mkdir()
    (fake / '__init__.py').write_text('unrelated = True\n')
    monkeypatch.chdir(tmp_path)
    after = source_identity()
    assert before == after
    assert 'skyhad_workbench/session.py' in after['source_files']
    assert 'had_env/simulation.py' in after['source_files']
    assert not any('/core/' in path or '/workbench/' in path for path in after['source_files'])


def test_actual_v3_recording_loads_but_cannot_continue():
    from skyhad_workbench.recording import load_episode
    from skyhad_workbench.cli import reconstruct_branch
    from skyhad_workbench.snapshot import decode_snapshot
    episode = load_episode(ROOT / 'tests/data/legacy_v3.json.gz')
    assert episode.metadata['complete']
    assert episode.frames and episode.decisions
    assert episode.frame_at(0)['entities']
    with pytest.raises(ValueError, match='Simulation sources differ'):
        reconstruct_branch(episode, episode.decisions[0]['step'], policy='rule')
    with pytest.raises(ValueError, match='Simulation sources differ'):
        decode_snapshot(episode.metadata['initial_snapshot'])


def test_wheel_contains_both_packages_and_installed_identity(tmp_path):
    build = tmp_path / 'build'
    build.mkdir()
    for name in ('had_env', 'skyhad_workbench'):
        shutil.copytree(ROOT / name, build / name, ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('pyproject.toml', 'README.md', 'make_env.py'):
        shutil.copy2(ROOT / name, build / name)
    wheels, installed = tmp_path / 'wheels', tmp_path / 'installed'
    result = subprocess.run([sys.executable, '-B', '-m', 'pip', 'wheel', str(build),
                             '--no-deps', '--no-build-isolation', '--wheel-dir', str(wheels)],
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    wheel, = wheels.glob('*.whl')
    result = subprocess.run([sys.executable, '-B', '-m', 'pip', 'install', str(wheel),
                             '--no-deps', '--target', str(installed)],
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    # A wheel inside another project's directory must never adopt that HEAD.
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    code = '''import json, had_env, skyhad_workbench
from importlib.metadata import distribution
from skyhad_workbench.identity import source_identity
identity = source_identity()
assert identity['git_commit'] == 'unavailable'
assert 'skyhad_workbench/agent_session.py' in identity['source_files']
assert 'had_env/simulation.py' in identity['source_files']
assert had_env.__version__ == skyhad_workbench.__version__ == distribution('had-env').version
entry = {e.name: e.value for e in distribution('had-env').entry_points}
assert entry['skyhad-workbench'] == entry['had-workbench'] == 'skyhad_workbench.cli:main'
print(json.dumps([had_env.__file__, skyhad_workbench.__file__]))
'''
    result = subprocess.run([sys.executable, '-B', '-c', code], cwd=tmp_path,
                            env=dict(os.environ, PYTHONPATH=str(installed)),
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert all(Path(path).resolve().is_relative_to(installed) for path in json.loads(result.stdout))


def test_qt_target_images_use_merged_resources(monkeypatch):
    monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    pytest.importorskip('PySide6')
    from PySide6.QtWidgets import QApplication
    from skyhad_workbench.views import EntityItem
    app = QApplication.instance() or QApplication([])
    item = EntityItem(dict(id=0, side='targets', role='Entity', position=[0., 0., 0.]))
    assert not item.target_image.isNull()
    assert not item.dead_target_image.isNull()
    app.processEvents()
