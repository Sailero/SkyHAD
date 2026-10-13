"""Edition export safety, reproducibility, and installed package contracts."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import shutil
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def generate(output, *extra):
    return subprocess.run([PYTHON, str(ROOT / 'tools/generate_core.py'), '--source', str(ROOT),
                           '--working-tree', '--output', str(output), *extra],
                          capture_output=True, text=True)


def test_core_export_is_reproducible_and_preserves_shared_engine(tmp_path):
    outputs = [tmp_path / 'first', tmp_path / 'second']
    for output in outputs:
        result = generate(output)
        assert result.returncode == 0, result.stderr
    first, second = outputs
    a = {p.relative_to(first).as_posix(): p.read_bytes() for p in first.rglob('*') if p.is_file()}
    b = {p.relative_to(second).as_posix(): p.read_bytes() for p in second.rglob('*') if p.is_file()}
    assert a == b
    assert 'had_env/environment.py' in a
    for name, data in a.items():
        if name.startswith('had_env/') and name.endswith('.py'):
            assert data == (ROOT / name).read_bytes(), name
    manifest = json.loads(a['edition-manifest.json'])
    assert a['had_env/edition_manifest.json'] == a['edition-manifest.json']
    assert len(manifest['source_full_sha']) == 40
    assert manifest['source_mode'] == 'working-tree'
    assert manifest['shared_core_sha256']['had_env/environment.py'] == hashlib.sha256(a['had_env/environment.py']).hexdigest()
    assert not any(name.startswith('skyhad_workbench/') for name in a)
    assert 'had_env/render/backend.py' in a
    assert 'had_env/render/render.py' in a
    assert 'had_env/resources/target.png' in a
    assert not any(name in a for name in ('had_env/render/scene3d.py', 'had_env/render/native3d.py', 'had_env/render/video.py'))
    assert 'examples/rendering.py' in a
    assert 'tests/data/legacy_v3.json.gz' not in a
    assert 'docs/showcase.png' not in a
    assert a['.github/workflows/tests.yml'] == (ROOT / '.github/workflows/tests.yml').read_bytes()


def test_generator_rejects_nonempty_and_source_targets_without_mutation(tmp_path):
    sentinel = tmp_path / 'occupied'
    sentinel.mkdir()
    (sentinel / 'keep.txt').write_text('keep')
    result = generate(sentinel)
    assert result.returncode != 0
    assert (sentinel / 'keep.txt').read_text() == 'keep'
    result = generate(ROOT)
    assert result.returncode != 0
    assert 'source' in result.stderr.lower()


def test_generator_requires_exact_full_commit_reference(tmp_path):
    result = subprocess.run([PYTHON, str(ROOT / 'tools/generate_core.py'), '--source', str(ROOT),
                             '--ref', 'HEAD', '--output', str(tmp_path / 'out')], capture_output=True, text=True)
    assert result.returncode != 0
    assert '40' in result.stderr


def test_commit_export_ignores_working_tree_edits(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    shutil.copytree(ROOT / 'had_env', source / 'had_env', ignore=shutil.ignore_patterns('__pycache__'))
    policy = json.loads((ROOT / 'tools/core_files.json').read_text())
    files = set(policy['copy']) | set(policy['templates'].values()) | {'tools/core_files.json'}
    for name in files:
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    subprocess.run(['git', 'init', '-q', str(source)], check=True)
    subprocess.run(['git', '-C', str(source), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(source), '-c', 'user.name=Edition test', '-c',
                    'user.email=edition-test@example.invalid', 'commit', '-qm', 'fixture'], check=True)
    sha = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    committed = subprocess.check_output(['git', '-C', str(source), 'show', f'{sha}:had_env/environment.py'])
    (source / 'had_env/environment.py').write_text('BROKEN WORKING TREE\n')
    output = tmp_path / 'export'
    result = subprocess.run([PYTHON, str(ROOT / 'tools/generate_core.py'), '--source', str(source),
                             '--ref', sha, '--output', str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (output / 'had_env/environment.py').read_bytes() == committed
    manifest = json.loads((output / 'edition-manifest.json').read_text())
    assert manifest['source_full_sha'] == sha
    assert manifest['source_mode'] == 'commit'
    committed_workflow = subprocess.check_output(['git', '-C', str(source), 'show', f'{sha}:.github/workflows/tests.yml'])
    assert (output / '.github/workflows/tests.yml').read_bytes() == committed_workflow


def test_publisher_rejects_workflow_blob_divergence_before_git_mutation(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    workflow = source / '.github/workflows/tests.yml'
    workflow.parent.mkdir(parents=True)
    workflow.write_text('name: canonical\non: push\n')
    subprocess.run(['git', 'init', '-q', str(source)], check=True)
    subprocess.run(['git', '-C', str(source), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(source), '-c', 'user.name=Edition test', '-c',
                    'user.email=edition-test@example.invalid', 'commit', '-qm', 'workflow fixture'], check=True)
    sha = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    generated = tmp_path / 'generated'
    changed = generated / '.github/workflows/tests.yml'
    changed.parent.mkdir(parents=True)
    changed.write_text('name: unauthorized alternate workflow\non: push\n')
    manifest = {'source_mode': 'commit', 'source_full_sha': sha,
                'files_sha256': {'.github/workflows/tests.yml': hashlib.sha256(changed.read_bytes()).hexdigest()}}
    (generated / 'had_env').mkdir()
    for name in ('edition-manifest.json', 'had_env/edition_manifest.json'):
        (generated / name).write_text(json.dumps(manifest))
    environment = dict(__import__('os').environ, GITHUB_REPOSITORY='Sailero/SkyHAD',
                       GITHUB_EVENT_NAME='push', GITHUB_REF='refs/heads/main')
    result = subprocess.run([PYTHON, str(ROOT / 'tools/publish_core.py'), '--source', str(source),
                             '--generated', str(generated), '--source-sha', sha],
                            capture_output=True, text=True, env=environment)
    assert result.returncode != 0
    assert 'canonical workflow' in result.stderr.lower()
    assert subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() == sha
    assert not subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True)


def test_built_core_installs_cleanly_and_matches_full_physics(tmp_path):
    result = subprocess.run([PYTHON, str(ROOT / 'tools/verify_editions.py'), '--source', str(ROOT),
                             '--working-tree', '--output', str(tmp_path / 'validation')],
                            capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((tmp_path / 'validation/validation-report.json').read_text())
    assert report['trajectory_count'] == 96
    assert report['core_tests_passed']
    assert report['core_2d_render_passed']
    assert report['wheel_and_sdist_checked']
    export = tmp_path / 'validation/core-source'
    manifest = json.loads((export / 'edition-manifest.json').read_text())
    names = {p.relative_to(export).as_posix() for p in export.rglob('*') if p.is_file()}
    assert names == set(manifest['files_sha256']) | {'edition-manifest.json', 'had_env/edition_manifest.json'}


def test_building_core_keeps_export_clean_for_publication(tmp_path):
    export = tmp_path / 'core-source'
    result = generate(export)
    assert result.returncode == 0, result.stderr
    before = {p.relative_to(export).as_posix(): p.read_bytes() for p in export.rglob('*') if p.is_file()}
    code = '''import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from verify_editions import build, check_archive
wheel, sdist = build(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
check_archive(wheel)
check_archive(sdist)
'''
    result = subprocess.run([PYTHON, '-c', code, str(ROOT / 'tools'), str(export),
                             str(tmp_path / 'artifacts'), str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    after = {p.relative_to(export).as_posix(): p.read_bytes() for p in export.rglob('*') if p.is_file()}
    assert after == before


def test_full_package_has_environment_extras_without_desktop_entry_points(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    shutil.copytree(ROOT / 'had_env', source / 'had_env', ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('pyproject.toml', 'README.md', 'LICENSE', 'make_env.py'):
        shutil.copyfile(ROOT / name, source / name)
    wheels = tmp_path / 'wheels'
    result = subprocess.run([PYTHON, '-m', 'pip', 'wheel', str(source), '--no-deps',
                             '--no-build-isolation', '--wheel-dir', str(wheels)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    wheel, = wheels.glob('*.whl')
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert not any(name.startswith('skyhad_workbench/') for name in names)
        assert not any(name.endswith('.dist-info/entry_points.txt') for name in names)
        metadata = archive.read(next(name for name in names if name.endswith('.dist-info/METADATA'))).decode()
        assert 'Provides-Extra: render' in metadata
        assert 'Provides-Extra: viewer3d' in metadata
        assert 'Provides-Extra: record' in metadata
        assert 'Provides-Extra: viewer\n' not in metadata


def test_publisher_rejects_development_exports_before_git_mutation(tmp_path):
    generated = tmp_path / 'core'
    result = generate(generated)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([PYTHON, str(ROOT / 'tools/publish_core.py'), '--source', str(ROOT),
                             '--generated', str(generated), '--source-sha', '0' * 40],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert 'commit export' in result.stderr.lower()
