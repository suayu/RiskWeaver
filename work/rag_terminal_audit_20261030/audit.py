"""独立终态审计：真实请求中的案例、闭合事件归属与窗口D，不改实验或运行源码。"""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,math,re


def canonical(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
def finite(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()


def actual_payload(call):
    """只读真正SDK的user消息末行JSON；不把较大的内部trace提示当真实传输。"""
    messages=[m['content'] for m in call.get('messages',[]) if m.get('role')=='user']
    if not messages or not isinstance(messages[-1],str):raise ValueError('missing_text_user_message')
    return json.loads(messages[-1].rsplit('\n',1)[-1])


def wire_cases(call):
    payload=actual_payload(call);evidence=payload.get('context',{}).get('retrieved_attack_evidence')
    documents=(evidence or {}).get('documents',[])
    return dict(payload=payload,field_present=evidence is not None,documents=documents,case_ids=[d['case_id'] for d in documents])


def recompute_window(case,windows):
    """E1中observe每次成功状态转移记一帧；窗口以源帧及实际观测帧数界定。"""
    start=case['source_step'];end=start+case['observed_frames']
    selected=[w for w in windows if start<=w['source_step']<end]
    valid=[w for w in selected if w.get('valid') and finite(w.get('difficulty'))]
    weight=sum(w.get('weight',1) for w in valid)
    value=sum(w.get('weight',1)*w['difficulty'] for w in valid)/weight if weight else None
    return dict(source_start=start,source_end_exclusive=end,observed_frames=len(selected),valid_frames=weight,D=value)


def same_number(a,b):return a is None and b is None or finite(a) and finite(b) and abs(a-b)<=1e-12


def audit(root):
    root=Path(root);protocol=json.loads((root/'protocol.json').read_text());terminal=root/'launcher_result.json'
    if not terminal.exists():raise ValueError('whole_run_not_terminal_do_not_publish_partial_performance')
    source_hashes={};errors=[];details=[];groups={};new_ids={}
    def read(p):
        raw=p.read_bytes();source_hashes[str(p)]=hashlib.sha256(raw).hexdigest();return raw.decode('utf-8')
    for arm in protocol['arms']:
        directory=root/arm;receipt_path=directory/'rag_experiment.json';calls_path=root/(arm+'_api_calls.jsonl')
        calls=[json.loads(s) for s in read(calls_path).splitlines()] if calls_path.exists() else []
        events=[];knowledge={};all_cases={};receipt={};wire_results=[];event_results=[]
        if receipt_path.exists():
            receipt=json.loads(read(receipt_path));knowledge_path=Path(receipt['knowledge_path']);knowledge=json.loads(read(knowledge_path))
            all_cases={r['case_id']:r for r in knowledge['records']}
            events=[r for r in knowledge['records'] if r.get('experiment_id')==receipt['experiment_id']]
            for row in knowledge['records']:
                if '_record_sha256' in row:
                    actual=digest({k:v for k,v in row.items() if k not in ('_record_sha256','_kb_sequence')})
                    if actual!=row['_record_sha256']:errors.append(dict(group=arm,case_id=row['case_id'],reason='case_content_sha_mismatch'))
        new_ids[arm]={r['case_id'] for r in events}
        fixed_docs=set();models=Counter();cited_all=[]
        for index,call in enumerate(calls):
            issues=[]
            try:
                visible=wire_cases(call);payload=visible['payload'];docs=visible['documents'];ids=visible['case_ids']
                fixed={d['path']:d['sha256'] for d in payload['knowledge']['documents'] if d['path'] in ['rw-current-fact-reasoning/SKILL.md','rw-case-evidence/SKILL.md']}
                if len(fixed)!=2:issues.append('missing_fixed_rule_at_actual_SDK_boundary')
                fixed_docs.add(canonical(fixed))
                if arm=='skills_no_rag' and visible['field_present']:issues.append('OFF_contains_RAG_field')
                if arm=='skills_no_rag' and (payload['context'].get('history') or any(c.get('vulnerability') for c in payload['context']['candidates'])):
                    issues.append('legacy_history_bypasses_RAG_off')
                if arm=='skills_rag' and not visible['field_present']:issues.append('ON_lost_RAG_field')
                expected=next((r for r in events if r.get('request_id')==payload['request_id']),None)
                if expected and expected['source_step']!=call.get('source_step'):issues.append('SDK_source_step_disagrees_with_event')
                for document in docs:
                    row=all_cases.get(document['case_id'])
                    if row is None:issues.append('case_not_in_this_experiment_KB');continue
                    if row.get('status')!='closed':issues.append('unclosed_case_at_wire')
                    if expected and row['case_id']==expected['case_id']:issues.append('self_case_at_wire')
                    if row.get('closed_at') is not None and row['closed_at']>call['started_at']:issues.append('case_closed_after_this_request')
                    if expected and (row.get('episode_id')==expected['episode_id'] and row.get('source_step',-1)>=expected['source_step']):issues.append('future_same_episode_case_at_wire')
                    if document.get('benefit_evidence_eligible'):
                        if not row.get('benefit_eligible'):issues.append('unverified_benefit_promoted')
                        if expected and (row.get('policy_key')!=expected['policy_key'] or row.get('protocol_key')!=expected['protocol_key']):issues.append('incompatible_benefit_promoted')
                    observed=document['observed_outcome'];value=row.get('D') if row.get('measurement_valid') else None
                    if not same_number(observed.get('D'),value):issues.append('wire_observed_D_disagrees_with_case')
                content=call.get('content') or '';citations=sorted(set(re.findall(r'\[case:([A-Za-z0-9_-]+)\]',content)))
                unknown=sorted(set(citations)-set(ids));cited_all.extend(citations)
                # 引用错ID是意见问题，另列，不倒改既有接收或伪装成服务失败。
                wire_results.append(dict(physical_index=index,source_step=call.get('source_step'),scenario_index=call.get('scenario_index'),
                    HTTP_response_received='content' in call,RAG_field_present=visible['field_present'],actual_case_count=len(ids),
                    actual_case_ids=ids,cited_case_ids=citations,unknown_cited_case_ids=unknown,
                    matched_event_id=expected['case_id'] if expected else None,issues=issues))
            except (ValueError,KeyError,TypeError) as error:issues.append('unparsed_actual_wire:'+type(error).__name__)
            if call.get('model_returned'):models[call['model_returned']]+=1
            errors.extend(dict(group=arm,physical_index=index,reason=r) for r in issues)
        attempts=[];video_states={}
        for path in sorted(directory.glob('movies/**/attempt_result.json')):
            result=json.loads(read(path));trace_path=path.with_name('execution_trace.jsonl')
            rows=[json.loads(s) for s in read(trace_path).splitlines()] if trace_path.exists() else []
            executions=[r for r in rows if r['kind']=='attack_execution' and r.get('executed')]
            state_count=sum(r['kind']=='state' for r in rows);video_states[str(path.parent)]=state_count
            # episode_id只存在KB，按源文件及相邻事件场景ID匹配；不能跨场拿同一target累加。
            source_name=Path(result['scenario_source']).name
            matched=[r for r in events if Path(str(r.get('scenario_source',''))).name==source_name]
            for event in matched:
                if event.get('unattributed_to_attack'):continue
                count=sum(r.get('request_id')==event.get('request_id') and r.get('target_id')==event.get('target_id') for r in executions)
                issues=[]
                if count!=event['actual_attack_execution_frames']:issues.append('actual_execution_attribution_count_mismatch')
                expected=recompute_window(event,result['danger_windows'])
                if expected['observed_frames']!=event['observed_frames']:issues.append('observation_window_frame_count_mismatch')
                if expected['valid_frames']!=event['valid_observation_frames'] or not same_number(expected['D'],event['D']):issues.append('window_D_or_valid_weight_mismatch')
                event_results.append(dict(case_id=event['case_id'],source_step=event['source_step'],request_id=event.get('request_id'),
                    actual_executed_frames=count,recomputed=expected,stored_D=event['D'],issues=issues))
                errors.extend(dict(group=arm,case_id=event['case_id'],reason=r) for r in issues)
            attempts.append(dict(scene_index=result['scenario_index'],executed_frames=result['executed_steps'],status=result['status'],
                actual_attack_frames=result.get('attack_executed_frames'),D=result.get('scenario_danger_score'),D_valid=result.get('danger_valid'),
                background_pairs_seen=any(r.get('background_pairs') or r.get('background_static_pairs') for r in rows if r['kind']=='state'),
                trace_state_count=state_count,video_path=result.get('video_path')))
        groups[arm]=dict(planned=len(protocol['planned_scenes']),attempt_results=len(attempts),unattempted=max(0,len(protocol['planned_scenes'])-len(attempts)),
            physical_SDK_requests=len(calls),HTTP_responses=sum('content' in c for c in calls),successful_models=dict(models),
            wire_RAG_field_count=sum(r['RAG_field_present'] for r in wire_results),
            physical_requests_with_actual_cases=sum(r['actual_case_count']>0 for r in wire_results),
            HTTP_responses_with_actual_cases=sum(r['actual_case_count']>0 and r['HTTP_response_received'] for r in wire_results),
            actual_wire_case_occurrences=sum(r['actual_case_count'] for r in wire_results),
            cited_case_occurrences=len(cited_all),unknown_citation_occurrences=sum(len(r['unknown_cited_case_ids']) for r in wire_results),
            event_count=len(events),events_rechecked=len(event_results),fixed_rule_variants=sorted(fixed_docs),
            attempts=attempts,video_trace_states=video_states,correct_reasoning_status='not_certified',P2_uplift_certified=False)
        details.append(dict(group=arm,wire_results=wire_results,event_results=event_results))
    if new_ids.get('skills_no_rag',set())&new_ids.get('skills_rag',set()):errors.append(dict(reason='cross_arm_new_event_ID_leak'))
    if groups.get('skills_no_rag',{}).get('fixed_rule_variants')!=groups.get('skills_rag',{}).get('fixed_rule_variants'):
        errors.append(dict(reason='different_fixed_rules_at_actual_wire'))
    for p,expected in source_hashes.items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=expected:raise RuntimeError('terminal_source_changed_during_read_only_audit')
    return dict(summary=dict(groups=groups,audit_issue_count=len(errors),audit_passed=not errors,
        scope='terminal_wire_event_integrity_not_semantic_gold_or_causal_effect',goal_complete=False),issues=errors,details=details,source_sha256=source_hashes)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=audit(args.root);args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as file:json.dump(result,file,ensure_ascii=False,indent=2)
    print(json.dumps(result['summary'],ensure_ascii=False,indent=2))
