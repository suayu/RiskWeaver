"""有锁、哈希检查和逐文件备份的小范围同步；拒绝覆盖期间产生的新改动。"""
from pathlib import Path
import hashlib
import json
import os
import sys

ROOT=Path('/home2/zhaoyx/scenario-dreamer');sys.path.insert(0,str(ROOT))
from policies.llm_adversarial_planner import _locked_model_pool


def main():
    folder=ROOT/'work/bailian_pool_update_20261022'
    entries=json.loads((folder/'apply_manifest.json').read_text())
    backup=folder/'base';backup.mkdir(exist_ok=False)
    with _locked_model_pool(ROOT/'policies/llm_adversarial_planner.py'):
        for entry in entries:
            target=ROOT/entry['target'];current=hashlib.sha256(target.read_bytes()).hexdigest() if target.exists() else None
            if current!=entry['before_sha256']:raise RuntimeError('source changed after inspection: '+entry['target'])
            if hashlib.sha256((folder/'staging'/entry['file']).read_bytes()).hexdigest()!=entry['after_sha256']:
                raise RuntimeError('staged file hash mismatch: '+entry['target'])
        for entry in entries:
            target=ROOT/entry['target']
            if target.exists():(backup/entry['file']).write_bytes(target.read_bytes())
            temp=target.with_name('.'+target.name+'.bailian_20261022.tmp')
            with temp.open('xb') as stream:stream.write((folder/'staging'/entry['file']).read_bytes())
            if target.exists():os.chmod(temp,target.stat().st_mode)
            os.replace(temp,target)
    (folder/'applied_manifest.json').write_text(json.dumps(entries,indent=2))
    print(json.dumps(dict(updated=[e['target'] for e in entries],backup=str(backup))))


if __name__=='__main__':main()
