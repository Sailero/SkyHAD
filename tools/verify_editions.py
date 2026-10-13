"""Build both editions, test installed core, and compare isolated trajectories."""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import zipfile

from generate_core import generate, git


def run(command, cwd, *, capture=False):
    environment = dict(os.environ)
    environment.pop('PYTHONPATH', None)
    environment['PYTHONNOUSERSITE'] = '1'
    result = subprocess.run([str(x) for x in command], cwd=cwd, env=environment,
                            capture_output=capture, text=True, check=True)
    return result.stdout if capture else None


def check_archive(path):
    if path.suffix == '.whl':
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            metadata = archive.read(next(n for n in names if n.endswith('.dist-info/METADATA'))).decode()
            assert 'Provides-Extra: test' in metadata
            assert 'Provides-Extra: render' in metadata
            assert not any(f'Provides-Extra: {extra}' in metadata for extra in ('viewer', 'viewer3d', 'record'))
            requirements = [line.lower() for line in metadata.splitlines() if line.startswith('Requires-Dist:')]
            assert not any(any(x in line for x in ('pyside', 'matplotlib', 'vispy', 'torch', 'imageio')) for line in requirements)
            assert all('extra == "render"' in line for line in requirements if 'pygame' in line)
            assert not any(n.endswith('.dist-info/entry_points.txt') for n in names)
    else:
        with tarfile.open(path) as archive:
            names = [member.name for member in archive.getmembers()]
    assert not any('skyhad_workbench' in name.split('/') for name in names), names
    assert not any(name.endswith(('/scene3d.py', '/native3d.py', '/video.py')) for name in names), names
    assert not any(name.lower().endswith(('.jpg', '.mp4')) or 'legacy_v3' in name for name in names), names
    assert any(name.endswith('had_env/render/backend.py') for name in names)
    assert any(name.endswith('had_env/resources/target.png') for name in names)


def build(source, artifacts, cwd):
    # Setuptools writes build/ and egg-info; the publishable export must stay
    # byte-for-byte unchanged after packaging.
    build_source = artifacts.parent / (artifacts.name + '-build-source')
    shutil.copytree(source, build_source, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    artifacts.mkdir()
    run([sys.executable, '-m', 'pip', 'wheel', build_source, '--no-deps', '--no-build-isolation', '--wheel-dir', artifacts], cwd)
    run([sys.executable, '-c', 'import setuptools.build_meta as b; b.build_sdist(' + repr(str(artifacts)) + ')'], build_source)
    wheel, = artifacts.glob('*.whl')
    sdist, = artifacts.glob('*.tar.gz')
    return wheel, sdist


def runtime(root, wheel):
    run([sys.executable, '-m', 'venv', '--without-pip', root], root.parent)
    python = root / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    run([sys.executable, '-m', 'pip', '--python', python, 'install', wheel, 'pytest>=8.4,<9'], root.parent)
    return python


def verify(source, output, *, ref=None, working_tree=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('validation output must be outside the source checkout')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('validation output must be new or empty')
    output.mkdir(parents=True, exist_ok=True)
    full, core = output / 'full-source', output / 'core-source'
    manifest = generate(source, core, ref=ref, working_tree=working_tree)
    full.mkdir()
    if working_tree:
        for name in ('had_env',):
            shutil.copytree(source / name, full / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        for name in ('pyproject.toml', 'README.md', 'LICENSE', 'make_env.py'):
            shutil.copyfile(source / name, full / name)
    else:
        with zipfile.ZipFile(io.BytesIO(git(source, 'archive', '--format=zip', ref))) as archive:
            for name in archive.namelist():
                destination = full / name
                if not destination.resolve().is_relative_to(full):
                    raise ValueError('unsafe Git archive member')
            archive.extractall(full)
    full_wheel, _ = build(full, output / 'full-artifacts', output)
    core_wheel, core_sdist = build(core, output / 'core-artifacts', output)
    check_archive(core_wheel)
    check_archive(core_sdist)
    full_python = runtime(output / 'full-runtime', full_wheel)
    core_python = runtime(output / 'core-runtime', core_wheel)
    smoke = '''
import importlib.util, sys
from pathlib import Path
from importlib.metadata import distribution
import had_env
import json
assert Path(had_env.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
assert had_env.__version__ == distribution('had-env').version == '4.2.0'
for name in ('pygame', 'PySide6', 'matplotlib', 'vispy', 'torch', 'imageio'):
    assert importlib.util.find_spec(name) is None, name
assert importlib.util.find_spec('had_env.render') is not None
for name in ('had_env.render.scene3d', 'had_env.render.native3d', 'had_env.render.video'):
    assert importlib.util.find_spec(name) is None
assert importlib.util.find_spec('skyhad_workbench') is None
assert not distribution('had-env').entry_points
manifest = json.loads(Path(had_env.__file__).with_name('edition_manifest.json').read_text())
assert manifest['edition'] == 'core' and manifest['version'] == '4.2.0'
'''
    run([core_python, '-I', '-B', '-c', smoke], output)
    # Tests live separately from the generated checkout: imports must resolve
    # to the installed wheel, not to the generated source package.
    checks = output / 'installed-core-checks'
    checks.mkdir()
    for name in ('tests', 'examples'):
        shutil.copytree(core / name, checks / name)
    run([core_python, '-I', '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '-o', 'pythonpath='], checks)
    script = output / 'trajectory_signature.py'
    signature_source = source / 'tools/trajectory_signature.py' if working_tree else full / 'tools/trajectory_signature.py'
    shutil.copyfile(signature_source, script)
    signatures = []
    for python in (full_python, core_python):
        signatures.append(json.loads(run([python, '-I', '-B', script], output, capture=True)))
    assert signatures[0] == signatures[1], 'full/core scientific trajectory mismatch'
    assert len(signatures[0]) == 96
    # Keep the base-only proof above separate, then exercise core's one
    # optional visual dependency without installing the full edition stack.
    run([sys.executable, '-m', 'pip', '--python', core_python, 'install', 'pygame>=2.5,<3'], output)
    render_smoke = '''
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'
import copy
import importlib.util
import numpy as np
from had_env import make_env
for name in ('PySide6', 'vispy', 'imageio', 'torch'):
    assert importlib.util.find_spec(name) is None, name
env = make_env(env_agent_type='UAV_fixedwing', red_count=2, blue_count=2, render_mode='rgb_array')
try:
    env.set_view_mode('projections')
    env.reset(seed=42)
    state = env.state().copy()
    rng = copy.deepcopy(env.np_random.bit_generator.state)
    frame = env.render()
    assert frame.ndim == 3 and frame.shape[2] == 3 and frame.dtype == np.uint8
    assert np.ptp(frame) > 0
    np.testing.assert_array_equal(env.state(), state)
    assert env.np_random.bit_generator.state == rng
    try:
        env.set_view_mode('3d')
    except ValueError as error:
        assert 'full' in str(error).lower()
    else:
        raise AssertionError('core unexpectedly accepts full-only 3D')
finally:
    env.close()
'''
    run([core_python, '-I', '-B', '-c', render_smoke], output)
    run([core_python, '-I', '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '-o', 'pythonpath=',
         'tests/test_render.py', 'tests/test_render_regressions.py'], checks)
    report = {'source_full_sha': manifest['source_full_sha'], 'source_mode': manifest['source_mode'],
              'shared_core_sha256': manifest['shared_core_sha256'],
              'trajectory_count': len(signatures[0]), 'core_tests_passed': True,
              'core_2d_render_passed': True,
              'wheel_and_sdist_checked': True, 'signatures': signatures[0]}
    (output / 'validation-report.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(f'Installed full/core parity: {len(signatures[0])} trajectories; core tests and archives passed')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--ref')
    mode.add_argument('--working-tree', action='store_true')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    verify(args.source, args.output, ref=args.ref, working_tree=args.working_tree)


if __name__ == '__main__':
    main()
