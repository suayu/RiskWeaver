"""仅安装本任务审核后的有限差异；预检用户源码哈希并备份，不修改环境或历史产物。"""
from pathlib import Path
import ast,hashlib,json,os,shutil,tempfile

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    after=HERE/'source_after';before=HERE/'source_before';receipt=HERE/'installation_receipt.json'
    if receipt.exists():raise FileExistsError('installation_already_recorded_do_not_overwrite')
    # 真实项目进程存活时不修改它们可能继续导入的公共模块。
    running=[]
    for proc in Path('/proc').glob('[0-9]*/cmdline'):
        if proc.parent.name==str(os.getpid()):continue
        try:
            args=proc.read_bytes().split(b'\0');cmd=b' '.join(args).decode(errors='replace')
            if args and 'python' in args[0].decode(errors='ignore') and str(ROOT) in cmd:running.append(proc.parent.name)
        except OSError:pass
    if running:raise RuntimeError('project_Python_processes_still_live:'+','.join(running))
    files=sorted(p.relative_to(after) for p in after.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    rows=[]
    for rel in files:
        new=after/rel;old=before/rel;target=ROOT/rel
        if new.suffix=='.py':ast.parse(new.read_text(encoding='utf-8'))
        if old.exists():
            if not target.is_file() or sha(target)!=sha(old):raise RuntimeError('remote_user_changes_detected:'+str(rel))
        elif target.exists():raise FileExistsError('new_module_already_exists:'+str(rel))
        rows.append(dict(path=rel.as_posix(),before_sha256=sha(old) if old.exists() else None,after_sha256=sha(new)))
    manifest=json.loads((after/'skills/riskweaver/manifest.json').read_text())
    for rel,expected in manifest['files'].items():
        path=after/'skills/riskweaver'/rel
        if not path.exists():path=ROOT/'skills/riskweaver'/rel
        if sha(path)!=expected:raise RuntimeError('skill_hash_mismatch:'+rel)
    import yaml
    yaml.safe_load((after/'cfgs/sim/base.yaml').read_text())
    backup=HERE/'remote_original_backup';backup.mkdir(exist_ok=False)
    installed=[]
    try:
        for rel in files:
            target=ROOT/rel;new=after/rel
            if target.exists():
                saved=backup/rel;saved.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(target,saved)
            target.parent.mkdir(parents=True,exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=target.parent,delete=False) as f:
                f.write(new.read_bytes());f.flush();os.fsync(f.fileno());temporary=Path(f.name)
            if target.exists():shutil.copymode(target,temporary)
            else:temporary.chmod(0o644)
            os.replace(temporary,target);installed.append(rel)
    except Exception:
        # 仅恢复本次确实替换过的文件；不删除任何历史实验或用户更改。
        for rel in reversed(installed):
            saved=backup/rel
            if saved.exists():shutil.copy2(saved,ROOT/rel)
            else:(ROOT/rel).unlink()
        raise
    result=dict(files=rows,count=len(rows),backup=str(backup),environment_changed=False,github_submitted=False)
    receipt.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
