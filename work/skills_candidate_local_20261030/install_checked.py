"""仅在目标文件仍等于下载前版本且无项目Python运行时安装；原件完整备份。"""
from pathlib import Path
import hashlib,json,os,shutil,stat,time

ROOT=Path('/home2/zhaoyx/scenario-dreamer');HERE=ROOT/'work/skills_candidate_local_20261030'


def main():
    before=json.loads((HERE/'source_before_sha256.json').read_text())
    active=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name)==os.getpid():continue
        try:
            args=[s.decode(errors='replace') for s in (p/'cmdline').read_bytes().split(b'\0') if s]
            cwd=(p/'cwd').resolve()
            scripts=[s for s in args if s.endswith('.py')]
            if scripts and (cwd==ROOT or any(s.startswith(str(ROOT)) for s in scripts)):active.append(int(p.name))
        except OSError:pass
    if active:raise RuntimeError('workspace_python_processes_still_running:'+str(active))
    for name,digest in before.items():
        p=ROOT/name
        if digest is None:
            if p.exists():raise FileExistsError('new_target_already_exists:'+name)
        elif hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise ValueError('concurrent_change_detected:'+name)
        if not (HERE/'source_after'/name).is_file():raise FileNotFoundError(name)
    backup=HERE/'remote_original_backup'
    if backup.exists():raise FileExistsError('preserve_previous_install')
    backup.mkdir()
    changed={}
    for name,digest in before.items():
        p=ROOT/name;data=(HERE/'source_after'/name).read_bytes()
        if digest is not None:
            dest=backup/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        temp=p.with_name(p.name+'.candidate_local_pending')
        if temp.exists():raise FileExistsError('preserve_pending_file')
        temp.write_bytes(data);os.chmod(temp,p.stat().st_mode if p.exists() else stat.S_IRUSR|stat.S_IWUSR|stat.S_IRGRP|stat.S_IROTH);os.replace(temp,p)
        changed[name]=hashlib.sha256(data).hexdigest()
    receipt=dict(installed_at=time.time(),original_sha256=before,installed_sha256=changed,
        default_mode='global',no_live_project_python=True,backup=str(backup),no_environment_changes=True,no_GitHub=True)
    (HERE/'install_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))


if __name__=='__main__':main()
