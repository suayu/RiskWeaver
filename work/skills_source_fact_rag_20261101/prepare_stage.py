"""仅准备原十场复核；两组继承同一跨场景已执行事件库，不自动启动。"""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys

ROOT = Path('/home2/zhaoyx/scenario-dreamer'); HERE = Path(__file__).resolve().parent
SCENES = [6,28,45,102,114,143,146,185,229,232]
PROTOCOL = ROOT/'work/skills_source_fact_rag_review10_protocol_20261101_r2'
EXPERIMENT = ROOT/'experiments/skills_source_fact_rag_review10_paired_20261101_r1'


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--knowledge-source',type=Path,required=True)
    args = parser.parse_args()
    # 新准备器保留旧实现；只使用已冻结的原十场清单，禁止按本轮结果删选。
    source = (ROOT/'work/rag_integration_20261030/prepare_pair.py').read_text()
    old = "ROOT/'experiments/skills_temporal_development10_paired_20261029_r1/manifest.json'"
    new = "ROOT/'work/skills_candidate_local_holdout_protocol_20261030_r1/validation10.json'"
    if source.count(old) != 1:
        raise ValueError('preserved_preparer_source_contract_changed')
    prepared = HERE/'prepare_pair.py'
    expected=source.replace(old,new)
    if prepared.exists():
        if prepared.read_text()!=expected:raise ValueError('generated_preparer_changed')
    else:prepared.write_text(expected,encoding='utf-8')
    names = {p.relative_to(ROOT/'work/rag_integration_20261030/source_after')
        for p in (ROOT/'work/rag_integration_20261030/source_after').rglob('*')
        if p.is_file() and '__pycache__' not in p.parts}
    names.update(p.relative_to(ROOT/'work/skills_source_facts_20261101/source_after')
        for p in (ROOT/'work/skills_source_facts_20261101/source_after').rglob('*')
        if p.is_file() and '__pycache__' not in p.parts)
    for name in names:
        target=HERE/'source_after'/name; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
    command=[sys.executable,str(prepared),'--output',str(PROTOCOL),'--experiment-root',str(EXPERIMENT),
        '--scene-indices',*[str(i) for i in SCENES],'--profile','off',
        '--inherit','--knowledge-source',str(args.knowledge_source)]
    result=subprocess.run(command,cwd=ROOT,text=True,capture_output=True,
        env=dict(os.environ,PYTHONPATH='.:safe-sim:safe-sim/trajdata/src',CUDA_VISIBLE_DEVICES=''))
    (HERE/'prepare_r2.log').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
    if result.returncode:raise RuntimeError('new_stage_prepare_failed_see_preserved_log')
    protocol=json.loads((PROTOCOL/'protocol.json').read_text())
    protocol['version']='skills_source_fact_rag_review10_v1'
    protocol['common_source_fact_contract_enabled']=True
    protocol['review_scene_scope']='original_ten_previously_examined; not_project_unseen'
    protocol['inheritance_scope']='previous_completed_74_and_182_executed_events; distinct_from_review_ten'
    protocol['known_counterexample_smoke_required_before_dispatch']=True
    (PROTOCOL/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(status='prepared_not_started',planned_attempts=20,scenes=SCENES,
        models=protocol['models'],source_fact_contract_enabled=True,API_requests=0)))


if __name__=='__main__':
    main()
