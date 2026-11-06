"""只读两组终态证据；所有预定场景留分母，高层接收/实际执行/文字正确分开报告。"""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from policies.rag_store import AttackRAGStore


def audit(root):
    root=Path(root);protocol=json.loads((root/'protocol.json').read_text());planned=len(protocol['planned_scenes'])
    if protocol['arms']!=['skills_no_rag','skills_rag']:raise ValueError('only_two_RAG_arms')
    groups={};new_ids={}
    for arm in protocol['arms']:
        directory=root/arm;rows=[];errors=[];calls=[];query_rows=[];events=[]
        for path in directory.glob('movies/**/attempt_result.json'):rows.append(json.loads(path.read_text()))
        journal=root/(arm+'_api_calls.jsonl')
        if journal.exists():calls=[json.loads(line) for line in journal.read_text().splitlines()]
        for p in directory.glob('movies/**/rag_queries.jsonl'):query_rows += [json.loads(line) for line in p.read_text().splitlines()]
        receipt=directory/'rag_experiment.json'
        if receipt.exists():
            receipt=json.loads(receipt.read_text());kb=AttackRAGStore(receipt['knowledge_path']);snapshot=kb.snapshot()
            events=[r for r in snapshot['records'] if r.get('experiment_id')==receipt['experiment_id']]
            if any(r.get('status')!='closed' for r in events):errors.append('unclosed_RAG_event')
            lineage=receipt['lineage']['lineage'];expected=protocol.get('inherited_seed_sha256')
            if (lineage or {}).get('source_sha256')!=expected:errors.append('inherit_source_mismatch')
        else:errors.append('missing_RAG_experiment_receipt');receipt={};kb=None
        new_ids[arm]={r['case_id'] for r in events}
        counts=Counter(c.get('scenario_index') for c in calls)
        if any(v>4 for v in counts.values()):errors.append('physical_budget_exceeded')
        if arm=='skills_no_rag':
            if any('retrieved_attack_evidence' in m.get('content','') for c in calls for m in c.get('messages',[])):errors.append('OFF_wire_contains_RAG_cases')
            if any(q.get('retrieved_documents') for q in query_rows):errors.append('OFF_retrieved_cases')
        for q in query_rows:
            for doc in q.get('retrieved_documents',[]):
                if doc['case_id']==q.get('query_facts',{}).get('current_case_id'):errors.append('self_event_retrieved')
        highlevel_scenes={r['scene_id'] for r in events if r.get('planner_validation') in ('accepted','accepted_obstacle_only')}
        measured=[r['scenario_danger_score'] for r in rows if r.get('danger_valid') and r.get('scenario_danger_score') is not None]
        terminal=[r for r in rows if r.get('status')=='completed']
        groups[arm]=dict(planned=planned,attempted=len(rows),unattempted=max(0,planned-len(rows)),
            executed_scenes=sum(r.get('executed_steps',0)>0 for r in rows),completed_runs=len(terminal),
            highlevel_accepted_scenes=len(highlevel_scenes),highlevel_accepted_plans=sum(r.get('planner_validation') in ('accepted','accepted_obstacle_only') for r in events),
            correct_attack_coverage=None,correctness_review_status='not_reviewed; acceptance_is_not_all_reasoning_correct',
            actual_attack_scenes=sum(r.get('attack_executed_frames',0)>0 for r in rows),actual_attack_frames=sum(r.get('attack_executed_frames',0) for r in rows),
            collision_count=sum(bool(r.get('collision')) for r in rows if r.get('executed_steps',0)>0),collision_denominator_executed=sum(r.get('executed_steps',0)>0 for r in rows),
            mean_valid_D=sum(measured)/len(measured) if measured else None,valid_D_scenes=len(measured),
            valid_D_frames=sum(r.get('danger_valid_frame_count',0) for r in rows),total_frames=sum(r.get('danger_total_frame_count',0) for r in rows),
            generation_failures=sum(bool(r.get('generation_failure')) for r in rows),runtime_failures=sum(bool(r.get('runtime_failure')) for r in rows),
            final_service_failures=sum(r.get('planner_service_failure_count',0) for r in rows),
            RAG_new_events=len(events),retrieval_queries=len(query_rows),retrieved_case_count=sum(len(q.get('retrieved_documents',[])) for q in query_rows),
            actual_wire_case_appearances=sum('retrieved_attack_evidence' in m.get('content','') for c in calls for m in c.get('messages',[])),
            physical_SDK_requests=len(calls),HTTP_successes=sum('content' in c for c in calls),HTTP_errors=dict(Counter(c.get('error_type') for c in calls if 'error_type' in c)),
            successful_models=dict(Counter(c.get('model_returned') for c in calls if c.get('model_returned'))),
            known_total_tokens=sum((c.get('usage') or {}).get('total_tokens',0) for c in calls),failed_unknown_usage_not_imputed=True,
            SDK_seconds=sum(c.get('elapsed_seconds',0) for c in calls),knowledge_path=str(kb.path) if kb else None,
            knowledge_sha256=hashlib.sha256(kb.path.read_bytes()).hexdigest() if kb else None,
            video_count=len(list(directory.rglob('*.mp4'))),trace_count=len(list(directory.rglob('execution_trace.jsonl'))),audit_errors=errors)
    cross_group_ids=new_ids['skills_no_rag']&new_ids['skills_rag']
    return dict(groups=groups,cross_group_new_case_ids=sorted(cross_group_ids),planned_attempts=2*planned,
        RAG_effectiveness_proven=False,goal_complete=False,scope='small_development_pair; all_failure_and_unattempted_denominators_preserved')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);a=parser.parse_args();print(json.dumps(audit(a.root),ensure_ascii=False,indent=2))
