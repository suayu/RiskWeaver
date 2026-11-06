"""先逐项检查，再备份当前文件并应用；不涉及 Git 提交或历史产物。"""
import hashlib
import json
from pathlib import Path
import shutil

stage=Path(__file__).resolve().parent
repo=Path('/home2/zhaoyx/scenario-dreamer')
manifest=json.loads((stage/'manifest.json').read_text())
for entry in manifest:
    target=repo/entry['target']
    current=hashlib.sha256(target.read_bytes()).hexdigest() if target.exists() else None
    if current!=entry['before']:
        raise RuntimeError('baseline changed: '+entry['target'])
    if hashlib.sha256((stage/'edit'/entry['name']).read_bytes()).hexdigest()!=entry['after']:
        raise RuntimeError('upload hash mismatch: '+entry['target'])
for entry in manifest:
    target=repo/entry['target']
    if target.exists():
        backup=stage/'backup'/entry['target']
        backup.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(target,backup)
    shutil.copy2(stage/'edit'/entry['name'],target)
    print('applied '+entry['target'])
