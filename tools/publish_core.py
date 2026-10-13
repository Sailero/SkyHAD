"""Publish a previously validated commit export by fast-forward only (CI only)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def git(source, *args):
    return subprocess.check_output(['git', '-C', str(source), *args], text=True).strip()


def publish(source, generated, sha):
    source, generated = Path(source).resolve(), Path(generated).resolve()
    manifest = json.loads((generated / 'edition-manifest.json').read_text())
    if manifest['source_mode'] != 'commit' or manifest['source_full_sha'] != sha:
        raise ValueError('publication requires the validated exact commit export')
    if os.environ.get('GITHUB_REPOSITORY') != 'Sailero/SkyHAD' or os.environ.get('GITHUB_EVENT_NAME') != 'push' or os.environ.get('GITHUB_REF') != 'refs/heads/main':
        raise ValueError('publication is restricted to canonical main push CI')
    files = {p.relative_to(generated).as_posix(): p for p in generated.rglob('*') if p.is_file()}
    if set(files) != set(manifest['files_sha256']) | {'edition-manifest.json', 'had_env/edition_manifest.json'}:
        raise ValueError('generated artifact contains unexpected or missing files')
    if files['had_env/edition_manifest.json'].read_bytes() != files['edition-manifest.json'].read_bytes():
        raise ValueError('source and installed package provenance must match')
    for name, expected in manifest['files_sha256'].items():
        path = files[name]
        if path.is_symlink() or not path.resolve().is_relative_to(generated) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'generated artifact integrity failure: {name}')
    workflow = '.github/workflows/tests.yml'
    if {name for name in files if name.startswith('.github/workflows/')} != {workflow}:
        raise ValueError('publication requires only the canonical workflow path')
    source_workflow = subprocess.check_output(['git', '-C', str(source), 'show', f'{sha}:{workflow}'])
    if files[workflow].read_bytes() != source_workflow:
        raise ValueError('generated canonical workflow differs from the validated source commit')
    def current_main():
        actual = git(source, 'ls-remote', 'origin', 'refs/heads/main').split()[0]
        if actual != sha:
            raise ValueError('main advanced since validation; refusing stale publication')
    current_main()
    core_ref = git(source, 'ls-remote', 'origin', 'refs/heads/core')
    with tempfile.TemporaryDirectory(prefix='skyhad-core-publish-') as temporary:
        checkout = Path(temporary) / 'checkout'
        if core_ref:
            subprocess.run(['git', '-C', str(source), 'fetch', 'origin', 'refs/heads/core'], check=True)
            base = core_ref.split()[0]
        else:
            base = sha
        subprocess.run(['git', '-C', str(source), 'worktree', 'add', '--detach', str(checkout), base], check=True)
        try:
            if not core_ref:
                subprocess.run(['git', '-C', str(checkout), 'switch', '--orphan', 'core-publish-initial'], check=True)
            subprocess.run(['git', '-C', str(checkout), 'rm', '-r', '--ignore-unmatch', '.'], check=True, stdout=subprocess.DEVNULL)
            for name, path in files.items():
                target = checkout / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
            subprocess.run(['git', '-C', str(checkout), 'add', '.'], check=True)
            if git(checkout, 'status', '--porcelain'):
                subprocess.run(['git', '-C', str(checkout), '-c', 'user.name=github-actions[bot]', '-c',
                                'user.email=41898282+github-actions[bot]@users.noreply.github.com',
                                'commit', '-m', f'Generate core 4.2.0 from {sha}'], check=True)
            current_main()
            subprocess.run(['git', '-C', str(checkout), 'push', 'origin', 'HEAD:refs/heads/core'], check=True)
        finally:
            subprocess.run(['git', '-C', str(source), 'worktree', 'remove', '--force', str(checkout)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--generated', required=True, type=Path)
    parser.add_argument('--source-sha', required=True)
    args = parser.parse_args()
    try:
        publish(args.source, args.generated, args.source_sha)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(2, f'core publication refused: {error}\n')


if __name__ == '__main__':
    main()
