"""只读核对Realtime闭环，保留未尝试分母，高层接收不代表连续执行。"""
from collections import Counter
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def lines(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()] if path.exists() else []


def signature(messages):
    return hashlib.sha256(json.dumps(messages,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def audit(root):
    root=Path(root)
    protocol=read(root/'protocol.json')
    terminal=(root/'launcher_result.json').exists()
    groups={}
    all_issues=[]
    for arm in protocol['arms']:
        directory=root/arm
        calls=lines(root/(arm+'_api_calls.jsonl'))
        transport=[]
        for path in directory.rglob('realtime_calls.jsonl'):
            transport.extend(lines(path))
        issues=[]
        # 外层一次物理调用与适配器一次连接逐请求内容对应，而非按控制事件数计费。
        outer=Counter(signature(c.get('messages')) for c in calls)
        inner=Counter(signature(c.get('messages')) for c in transport)
        realtime_expected=protocol.get('transport','realtime')=='realtime'
        if not realtime_expected and transport:
            issues.append('unexpected_realtime_calls_in_chat_protocol')
        if not realtime_expected and (not (directory/'realtime_entry.json').exists() or read(directory/'realtime_entry.json').get('enabled') is not False):
            issues.append('chat_transport_runtime_entry_not_verified')
        if realtime_expected and outer!=inner:
            issues.append('outer_and_realtime_call_journals_disagree')
        if any(v>4 for v in Counter(c.get('scenario_index') for c in calls).values()):
            issues.append('physical_request_budget_exceeded')
        if any(c.get('model_requested') not in protocol['models'] for c in calls):
            issues.append('outer_model_not_frozen')
        for call in transport:
            if call.get('model_requested') not in protocol['models']:
                issues.append('realtime_model_not_frozen')
            events=call.get('events',[])
            creates=sum(e.get('direction')=='client' and e.get('type')=='response.create' for e in events)
            if creates!=call.get('response_create_count') or creates>1:
                issues.append('response_create_accounting_error')
            if call.get('model_returned') and not isinstance(call.get('raw_usage'),dict):
                issues.append('successful_response_missing_usage')
        episodes=[]
        for scene in protocol['planned_scenes']:
            candidates=[]
            for path in directory.glob('movies/*/attempt_result.json'):
                value=read(path)
                if value.get('scenario_index')==scene:
                    candidates.append((path,value))
            if not candidates:
                episodes.append(dict(scenario_index=scene,attempted=False,executed_steps=0,
                                     collision=None,status='missing_attempt_record',D=None))
                continue
            if len(candidates)!=1:
                issues.append('duplicate_attempt_record')
            path,value=candidates[0]
            trace=lines(path.parent/'execution_trace.jsonl')
            states=[r for r in trace if r.get('kind')=='state']
            steps=value.get('executed_steps',0)
            if len(states)!=steps+1:
                issues.append('trace_state_execution_count_mismatch')
            outputs=[r for r in trace if r.get('kind')=='llm_output']
            high=sum(bool((r.get('validated_plan') or {}).get('attack')) for r in outputs)
            metrics={k:value.get(k) for k in ['scenario_index','executed_steps','status','collision',
                'completed','off_route','progress','attack_plan_count','attack_executed_frames',
                'scenario_danger_score','danger_valid_frame_count','danger_total_frame_count',
                'min_ttc_s','mean_evasive_acceleration_mps2','generation_failure',
                'generation_failure_reason','runtime_failure','runtime_failure_reason',
                'planner_service_failure_count','prediction_service_failure_count','feasibility']}
            metrics.update(attempted=True,trace_states=len(states),highlevel_validated_attack_outputs=high,
                highlevel_output_count=len(outputs),plan_accepted_trace_events=sum(r.get('kind')=='plan_accepted' for r in trace),
                attack_execution_trace_frames=sum(r.get('kind')=='attack_execution' and bool(r.get('executed')) for r in trace),
                highlevel_correctness='not_semantically_reviewed',video_paths=[str(p.relative_to(root)) for p in path.parent.glob('*.mp4')])
            # 零执行的碰撞保持未知，不能因失败前未撞而写成安全完成。
            if not steps and value.get('collision') is not None:
                issues.append('zero_execution_collision_must_be_unknown')
            episodes.append(metrics)
        groups[arm]=dict(planned=len(protocol['planned_scenes']),attempt_record_count=sum(r['attempted'] for r in episodes),
            executed_scenes=sum(r.get('executed_steps',0)>0 for r in episodes),
            highlevel_attack_scenes=sum(r.get('highlevel_validated_attack_outputs',0)>0 for r in episodes),
            physical_calls=len(calls),expected_transport=protocol.get('transport','realtime'),realtime_connections=len(transport),
            response_creates=sum(c.get('response_create_count',0) for c in transport),
            observed_tokens=sum((c.get('raw_usage') or {}).get('total_tokens',0) for c in transport) if realtime_expected else sum((c.get('usage') or {}).get('total_tokens',0) for c in calls),
            unknown_usage_calls=sum(not isinstance(c.get('raw_usage'),dict) for c in transport) if realtime_expected else sum(not isinstance(c.get('usage'),dict) for c in calls),
            episodes=episodes,audit_issues=issues,correct_attack_coverage=None)
        all_issues.extend(arm+':'+v for v in issues)
    return dict(terminal=terminal,planned_attempts=protocol['planned_attempts'],groups=groups,
                audit_issues=all_issues,accounting_verified=terminal and not all_issues,
                goal_complete=False,scope='accounting_and_execution_evidence; not_text_correctness_or_efficacy_certificate')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    # 原件保持只读，拒绝将审计结果写回实验目录或覆盖旧审计。
    if args.output.resolve().is_relative_to(args.root.resolve()):
        raise ValueError('audit_output_must_be_outside_experiment')
    result=audit(args.root)
    if not result['terminal']:
        raise RuntimeError('experiment_not_terminal')
    with args.output.open('x',encoding='utf-8') as file:
        json.dump(result,file,ensure_ascii=False,indent=2)
    print(json.dumps({'planned':result['planned_attempts'],'audit_issues':result['audit_issues']}))
