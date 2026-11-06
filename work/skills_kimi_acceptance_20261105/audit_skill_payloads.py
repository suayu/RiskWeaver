"""独立技能叶文档审计，不修改正在运行的冻结实验或原始快照。"""
import argparse
import hashlib
import json
from pathlib import Path


RULES=['rw-current-fact-reasoning/SKILL.md','rw-case-evidence/SKILL.md',
       'rw-source-facts/SKILL.md','rw-static-fallback/SKILL.md']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload_contains(text,call):
    """文档全文可以直接或以JSON字符串转义出现，摘要/仅SHA不够。"""
    variants=[text,json.dumps(text,ensure_ascii=False)[1:-1],json.dumps(text,ensure_ascii=True)[1:-1]]
    return any(v in str(message.get('content','')) for v in variants for message in call.get('messages',[]))


def check_document(path,expected,calls):
    if not path.is_file():return dict(verified=False,issue='document_missing',request_ids=[])
    if sha(path)!=expected:return dict(verified=False,issue='document_sha_mismatch',request_ids=[])
    text=path.read_text(encoding='utf-8')
    matches=[c['request_id'] for c in calls if payload_contains(text,c)]
    return dict(verified=bool(matches),issue=None if matches else 'full_text_missing_from_actual_SDK',request_ids=matches)


def audit(root,supplemental):
    protocol=read(root/'protocol.json');frozen=read(root/'source_sha256.json')
    manifest_path=root/'source_snapshot/skills/riskweaver/manifest.json'
    if sha(manifest_path)!=frozen.get('skills/riskweaver/manifest.json'):
        raise ValueError('parent_skill_manifest_not_frozen')
    manifest=read(manifest_path);groups={}
    for arm in protocol['arms']:
        directory=root/arm;calls=[]
        for path in directory.rglob('realtime_calls.jsonl'):
            calls.extend(json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip())
        if protocol.get('transport')=='chat_completions':
            journal=root/(arm+'_api_calls.jsonl')
            calls=[json.loads(line) for line in journal.read_text(encoding='utf-8').splitlines() if line.strip()] if journal.exists() else []
        if len({c['request_id'] for c in calls})!=len(calls):raise ValueError('duplicate_physical_request_id')
        issues=[];documents={};events=[]
        for name in RULES:
            original=root/'source_snapshot/skills/riskweaver'/name
            path=original if original.is_file() else supplemental/'skills/riskweaver'/name
            result=check_document(path,manifest['files'][name],calls)
            documents[name]=dict(result,path=str(path),expected_sha256=manifest['files'][name])
            # 静态未被调用时无需出现静态规则，但文件内容完整性仍必须通过。
            if result['issue'] in ['document_missing','document_sha_mismatch']:issues.append(name+':'+result['issue'])
        for path in directory.glob('movies/*/execution_trace.jsonl'):
            for line in path.read_text(encoding='utf-8').splitlines():
                row=json.loads(line)
                if row.get('kind')!='llm_output':continue
                trace=row.get('planner_trace') or {};selected=trace.get('skill_manifest') or {}
                if not selected:continue
                request_id=trace.get('request_id') or (row.get('validated_plan') or {}).get('request_id')
                associated=[c for c in calls if request_id and any(request_id in str(m.get('content','')) for m in c['messages'])]
                event_issues=[]
                for document in selected.get('documents',[]):
                    name=document['path'];expected=document['sha256']
                    # 原有技能目录优先；不能仅凭同名文件推断版本。
                    candidates=[root/'source_snapshot/work/skills_incremental_evidence_20261025/skills'/name,
                        root/'source_snapshot/skills/riskweaver'/name,supplemental/'skills/riskweaver'/name]
                    candidates=[p for p in candidates if p.is_file() and sha(p)==expected]
                    if not candidates:event_issues.append('selected_document_missing_or_changed:'+name);continue
                    if not check_document(candidates[0],expected,associated)['verified']:
                        event_issues.append('selected_document_not_bound_to_actual_request:'+name)
                # 源事实规则走独立装配支路，不出现在selected.documents中，同样核验请求归属。
                name='rw-source-facts/SKILL.md';value=documents[name]
                if not check_document(Path(value['path']),value['expected_sha256'],associated)['verified']:
                    event_issues.append('source_fact_rule_not_bound_to_actual_request')
                events.append(dict(scene=path.parent.name,step=row['step'],request_id=request_id,
                                   associated_physical_ids=[c['request_id'] for c in associated],issues=event_issues))
                issues.extend(event_issues)
        if not events:issues.append('no_actual_skill_events')
        groups[arm]=dict(verified=not issues,issues=sorted(set(issues)),documents=documents,events=events,
                         physical_requests=len(calls))
    return dict(verified=all(g['verified'] for g in groups.values()),groups=groups,
                original_experiment_modified=False,scope='skill_artifact_and_payload_integrity_only')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    parser.add_argument('--supplemental',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.resolve().is_relative_to(args.root.resolve()):raise ValueError('output_must_be_outside_experiment')
    result=audit(args.root,args.supplemental)
    with args.output.open('x',encoding='utf-8') as file:json.dump(result,file,ensure_ascii=False,indent=2)
    print(json.dumps(dict(verified=result['verified'],groups={k:dict(events=len(v['events']),issues=v['issues']) for k,v in result['groups'].items()})))
