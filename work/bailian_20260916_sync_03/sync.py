"""校验整个上传批次后备份并同步，拒绝覆盖远端同期更改。"""
import hashlib
import json
from pathlib import Path
import shutil
import sys

root=Path(__file__).resolve().parent
mapping={
 'scenario_generator.py':'scenario_generator.py',
}
if sys.argv[1]=='manifest':
 records=[]
 for name,target in mapping.items():
  edit=root/'edit'/name
  data=edit.read_bytes().replace(b'\r\n',b'\n')
  edit.write_bytes(data)
  before=root/'baseline'/name
  records.append(dict(name=name,target=target,before=hashlib.sha256(before.read_bytes()).hexdigest() if before.exists() else None,
                      after=hashlib.sha256(data).hexdigest()))
 (root/'manifest.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
else:
 repo=Path('/home2/zhaoyx/scenario-dreamer')
 records=json.loads((root/'manifest.json').read_text())
 for entry in records:
  file=repo/entry['target']
  current=hashlib.sha256(file.read_bytes()).hexdigest() if file.exists() else None
  if current!=entry['before']:raise RuntimeError('remote changed: '+entry['target'])
  if hashlib.sha256((root/'edit'/entry['name']).read_bytes()).hexdigest()!=entry['after']:raise RuntimeError('upload corrupted: '+entry['target'])
 for entry in records:
  file=repo/entry['target']
  if file.exists():
   backup=root/'backup'/entry['target']
   backup.parent.mkdir(parents=True,exist_ok=True)
   shutil.copy2(file,backup)
  shutil.copy2(root/'edit'/entry['name'],file)
  print('applied '+entry['target'])
