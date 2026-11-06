"""校验旧文件哈希后安装五个聚焦文件；备份和原实验目录均保留。"""
from pathlib import Path
import hashlib
import json
import os
import sys

ROOT = Path('/home2/zhaoyx/scenario-dreamer')
HERE = Path(__file__).resolve().parent


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    expected = json.loads((HERE/'before_sha256.json').read_text())
    sources = sorted(p for p in (HERE/'source_after').rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    if len(sources) != 5:
        raise RuntimeError('unexpected_install_scope')
    active = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            command = (proc/'cmdline').read_bytes().split(b'\0')
            if command and b'python' in command[0] and os.readlink(proc/'cwd') == str(ROOT):
                active.append(int(proc.name))
        except (OSError, PermissionError):
            pass
    if active:
        raise RuntimeError('project_processes_active:'+str(active))
    records = []
    for source in sources:
        name = source.relative_to(HERE/'source_after').as_posix()
        target = ROOT/name
        old = target.read_bytes() if target.exists() else None
        if (digest(old) if old is not None else None) != expected.get(name):
            raise RuntimeError('existing_source_changed:'+name)
        records.append(dict(path=name, before_sha256=digest(old) if old is not None else None,
                            after_sha256=digest(source.read_bytes())))
    if '--apply' not in sys.argv:
        print(json.dumps(dict(dry_run=True, files=records))); return
    backup = HERE/'remote_backup'; backup.mkdir(exist_ok=False)
    for row in records:
        target = ROOT/row['path']
        if target.exists():
            copy = backup/row['path']; copy.parent.mkdir(parents=True, exist_ok=True)
            copy.write_bytes(target.read_bytes())
    for row in records:
        target = ROOT/row['path']; target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name+'.source-facts-tmp')
        with temporary.open('xb') as file:
            file.write((HERE/'source_after'/row['path']).read_bytes())
        temporary.replace(target)
        if digest(target.read_bytes()) != row['after_sha256']:
            raise RuntimeError('installed_hash_mismatch:'+row['path'])
    receipt = dict(status='installed', files=records, changed_environment=False,
                   default_enabled=False, changed_solver_or_safety=False, github_submission=False)
    (HERE/'installation_receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
