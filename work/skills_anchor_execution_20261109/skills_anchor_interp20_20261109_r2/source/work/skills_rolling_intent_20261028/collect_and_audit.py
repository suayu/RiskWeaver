"""终态只读审计和归档；不得改写实验原件或复制凭据。"""
from pathlib import Path
import importlib.util
import json
import subprocess

ROOT=Path(__file__).resolve().parents[2]
OUT=Path('D:/RiskWeaver/outputs/skills_shared_resource_20261028')
NAME='skills_four_arm_provider_shared_development10_20261028_r2'


def main():
    # 通过stdin传递Python，避免跨Windows与Linux shell的引号歧义。
    script=f'''import sys,json
from pathlib import Path
sys.path.insert(0,'work/skills_shared_formal_20261028')
from audit_shared import audit_shared
root=Path('experiments/{NAME}')
assert (root/'launcher_result.json').exists()
result=audit_shared(root)
print(json.dumps(result,ensure_ascii=False,allow_nan=False))
'''
    r=subprocess.run(['ssh','-o','BatchMode=yes','zhaoyx@172.18.36.182',
        'cd /home2/zhaoyx/scenario-dreamer && /home2/zhaoyx/miniconda3/envs/scenario-dreamer/bin/python -'],
        input=script.encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=180,check=True)
    audit=json.loads(r.stdout)
    output=OUT/'terminal_independent_acceptance_20261029.json'
    with output.open('x',encoding='utf-8') as stream:json.dump(audit,stream,ensure_ascii=False,indent=2)
    spec=importlib.util.spec_from_file_location('collector',ROOT/'work/skills_integration_20261020/collect_verified.py')
    collector=importlib.util.module_from_spec(spec);spec.loader.exec_module(collector)
    start=collector.REMOTE.index('scopes=[');end=collector.REMOTE.index('allowed=',start)
    scopes=[f'experiments/{NAME}',f'work/{NAME}_acceptance.json',f'work/{NAME}_launcher.log',
        f'work/{NAME}_dispatch.json','work/skills_shared_formal_20261028',
        'work/skills_provider_shared_protocol_20261028_r2']
    collector.REMOTE=collector.REMOTE[:start]+'scopes='+repr(scopes)+'\nresource_terminals=[]\n'+collector.REMOTE[end:]
    collector.OUT=OUT;collector.main()
    print(json.dumps(dict(development_gate=audit['development_gate'],groups={k:{x:v for x,v in g.items()
        if x not in ['scenes','attempts']} for k,g in audit['groups'].items()}),ensure_ascii=False))


if __name__=='__main__':main()
