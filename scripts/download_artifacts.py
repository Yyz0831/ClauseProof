"""Restore pinned public artifacts; refuse changed existing files and unsafe paths."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def restore(archive, index, root):
    index = json.loads(Path(index).read_text())
    payload = Path(archive).read_bytes()
    if sha(payload) != index['archive_sha256']:
        raise ValueError('Archive checksum mismatch')
    pending = []
    with tarfile.open(fileobj=io.BytesIO(payload), mode='r:gz') as tf:
        members = tf.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)) or set(names) != set(index['files']):
            raise ValueError('Unexpected or duplicate archive members')
        for member in members:
            rel = PurePosixPath(member.name)
            if not member.isfile() or rel.is_absolute() or '..' in rel.parts:
                raise ValueError('Unsafe archive member: ' + member.name)
            dest = root.joinpath(*rel.parts)
            if dest.is_symlink() or not dest.resolve().is_relative_to(root.resolve()):
                raise ValueError('Unsafe destination: ' + member.name)
            data = tf.extractfile(member).read()
            if sha(data) != index['files'][member.name]:
                raise ValueError('Member checksum mismatch: ' + member.name)
            if dest.exists() and (not dest.is_file() or dest.read_bytes() != data):
                raise ValueError('Refusing to overwrite changed file: ' + member.name)
            pending.append((dest, data))
    for dest, data in pending:
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open('xb') as f:
                f.write(data)
    return len(pending)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--artifacts-only', action='store_true', help='Skip the v2 adapter')
    ap.add_argument('--include-base', action='store_true', help='Download gated upstream base after accepting its terms')
    args = ap.parse_args()
    from huggingface_hub import hf_hub_download, snapshot_download
    lock = json.loads((ROOT / 'release-lock.json').read_text())
    for item in lock['datasets']:
        kw = dict(repo_id=item['repo_id'], repo_type='dataset', revision=item['revision'])
        index = hf_hub_download(filename='files.json', **kw)
        if sha(Path(index).read_bytes()) != item['index_sha256']:
            raise ValueError('Index checksum mismatch')
        archive = hf_hub_download(filename='reproduction.tar.gz', **kw)
        count = restore(archive, index, ROOT)
        print(f"Verified/restored {count} files from {item['repo_id']}")
    if not args.artifacts_only:
        item = lock['adapter']
        dest = ROOT / item['path']
        for filename, expected in item['files'].items():
            src = Path(hf_hub_download(repo_id=item['repo_id'], revision=item['revision'], filename=filename))
            data = src.read_bytes()
            if sha(data) != expected:
                raise ValueError('Adapter checksum mismatch: ' + filename)
            target = dest / filename
            if target.exists() and target.read_bytes() != data:
                raise ValueError('Refusing to overwrite changed adapter: ' + filename)
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as f:
                    f.write(data)
        print('Verified/restored v2 adapter')
    if args.include_base:
        snapshot_download(repo_id=lock['base']['repo_id'], revision=lock['base']['revision'])
        print('Upstream base cached; access remains subject to its license')


if __name__ == '__main__':
    main()
