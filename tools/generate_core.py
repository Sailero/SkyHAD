"""Export the reviewed environment + minimal 2D edition without engine edits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess

GENERATOR_VERSION = '2'


def git(source, *args):
    return subprocess.check_output(['git', '-C', str(source), *args])


def generate(source, output, *, ref=None, working_tree=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError('output must be outside the source checkout')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('output must be a new or empty directory; existing files are never removed')
    if working_tree:
        sha = git(source, 'rev-parse', 'HEAD').decode().strip()
        def read(name):
            path = source / name
            if path.is_symlink() or not path.resolve().is_relative_to(source):
                raise ValueError(f'unsafe source path: {name}')
            return path.read_bytes()
        names = {p.relative_to(source).as_posix() for p in (source / 'had_env').rglob('*')
                 if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc', '.pyo')}
    else:
        if ref is None or re.fullmatch(r'[0-9a-f]{40}', ref) is None:
            raise ValueError('--ref requires an exact lowercase 40-character full Git commit SHA')
        sha = git(source, 'rev-parse', ref + '^{commit}').decode().strip()
        if sha != ref:
            raise ValueError('reference must name the exact commit')
        def read(name):
            mode = git(source, 'ls-tree', sha, '--', name).decode().split()[0]
            if mode != '100644':
                raise ValueError(f'only regular source files are allowed: {name}')
            return git(source, 'show', f'{sha}:{name}')
        names = set(git(source, 'ls-tree', '-r', '--name-only', sha, 'had_env').decode().splitlines())
    policy = json.loads(read('tools/core_files.json').decode('utf-8-sig'))
    if any(name.startswith('.github/workflows/') for name in policy['templates']):
        raise ValueError('canonical workflow files must be copied byte-identically, never templated')
    shared = set(policy['shared_core'])
    if shared.intersection(policy['templates']):
        raise ValueError('shared engine files cannot be replaced by edition templates')
    actual_shared = {name for name in names if not name.startswith(('had_env/render/', 'had_env/resources/'))}
    if actual_shared != shared:
        raise ValueError(f'unreviewed shared engine file selection: added={sorted(actual_shared-shared)}, missing={sorted(shared-actual_shared)}')
    files = {}
    for name in policy['copy'] + policy['shared_core']:
        files[name] = read(name)
    for target, template in policy['templates'].items():
        files[target] = read(template).replace(b'\r\n', b'\n')
    workflow_paths = {name for name in files if name.startswith('.github/workflows/')}
    if workflow_paths != {'.github/workflows/tests.yml'}:
        raise ValueError('export must contain only the shared canonical workflow')
    for name in files:
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or '\\' in name:
            raise ValueError(f'unsafe output path: {name}')
    manifest = {'edition': 'core', 'version': '4.2.0', 'generator_version': GENERATOR_VERSION,
                'source_full_sha': sha, 'source_mode': 'working-tree' if working_tree else 'commit',
                'shared_core_sha256': {name: hashlib.sha256(files[name]).hexdigest() for name in sorted(shared)},
                'files_sha256': {name: hashlib.sha256(data).hexdigest() for name, data in sorted(files.items())}}
    files['edition-manifest.json'] = (json.dumps(manifest, sort_keys=True, indent=2) + '\n').encode()
    files['had_env/edition_manifest.json'] = files['edition-manifest.json']
    output.mkdir(parents=True, exist_ok=True)
    for name, data in sorted(files.items()):
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--ref')
    mode.add_argument('--working-tree', action='store_true', help='explicit non-release development export')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        manifest = generate(args.source, args.output, ref=args.ref, working_tree=args.working_tree)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(2, f'core generation failed: {error}\n')
    print(f"core {manifest['version']} from {manifest['source_full_sha']} ({manifest['source_mode']})")


if __name__ == '__main__':
    main()
