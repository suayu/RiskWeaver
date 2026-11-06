"""独立只读验收：精确清单、传输预算、计划执行账本；不修改冻结v1验收器。"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

ROOT=Path('/home2/zhaoyx/scenario-dreamer')
sys.path.insert(0,str(ROOT))
from experiments.skills.audit import audit_experiment


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def manifest_coverage(manifest, attempts):
    """数量相同仍可能跑错场景；全局编号与文件路径都必须一致。"""
    expected=manifest.get('original_scenario_indices',manifest.get('scenario_indices'))
    if expected is None:
        return dict(verified=False,reason='global_indices_missing')
    actual=[a['scenario_index'] for a in attempts]
    source_map=dict(zip(expected,manifest['scenario_files']))
    mismatched=[a['scenario_index'] for a in attempts if a['scenario_index'] in source_map
                and str(Path(a['scenario_source']).resolve())!=str(Path(source_map[a['scenario_index']]).resolve())]
    missing=sorted(set(expected)-set(actual));unexpected=sorted(set(actual)-set(expected))
    duplicate=sorted(k for k,v in Counter(actual).items() if v>1)
    valid_manifest=len(expected)==manifest['scenario_count']==len(manifest['scenario_files']) and len(set(expected))==len(expected)
    return dict(verified=valid_manifest and not (missing or unexpected or duplicate or mismatched),
                manifest_indices_valid=valid_manifest,expected=expected,actual=actual,
                missing=missing,unexpected=unexpected,duplicates=duplicate,source_mismatches=mismatched)


def plan_execution_ledger(rows):
    """将实际控制标记关联到接收计划；无request_id的旧流程只能按目标与时间关联。"""
    accepted=[r for r in rows if r['kind']=='plan_accepted']
    outputs={r['step']:(r.get('validated_plan') or {}) for r in rows if r['kind']=='llm_output'}
    intents={r['step']:r.get('attack_intent') for r in rows if r['kind']=='prediction'}
    predictions={r['step']:r.get('joint') for r in rows if r['kind']=='execution_prediction'}
    result=[]
    for r in accepted:
        plan=outputs.get(r['step'],{});intent=intents.get(r['step']) or {}
        duration=int(intent.get('duration',plan.get('duration',10)))
        result.append(dict(source_step=r['step'],target_id=r['target_id'],request_id=r.get('request_id'),
                           duration_frames=duration,actual_execution_steps=[],
                           attribution='request_id_and_time' if r.get('request_id') else 'legacy_target_and_time'))
    unmatched=[];unguided=[];all_steps=[]
    for marker in rows:
        if marker['kind']!='attack_execution' or not marker.get('executed'):continue
        all_steps.append(marker['step'])
        source=marker.get('diffusion_source_step',marker['step']-1)
        candidates=[p for p in result if p['source_step']<=source and p['target_id']==marker['target_id']]
        rid=marker.get('request_id')
        if rid is not None:candidates=[p for p in candidates if p['request_id']==rid]
        if not candidates:
            unmatched.append(marker['step']);continue
        plan=max(candidates,key=lambda p:p['source_step']);plan['actual_execution_steps'].append(marker['step'])
        joint=predictions.get(source) or {}
        if not joint.get('metadata',{}).get('anchor_guidance_active'):
            unguided.append(marker['step'])
    for p in result:
        p['executed_frames']=len(set(p['actual_execution_steps']))
        p['executed']=p['executed_frames']>0
        p['exceeds_declared_duration']=p['executed_frames']>p['duration_frames']
    return dict(plans=result,accepted_plans=len(result),executed_plans=sum(p['executed'] for p in result),
                unexecuted_plans=sum(not p['executed'] for p in result),
                executed_plan_fraction=sum(p['executed'] for p in result)/len(result) if result else None,
                unmatched_execution_steps=unmatched,execution_without_anchor_guidance_steps=unguided,
                duplicate_execution_markers=len(all_steps)-len(set(all_steps)))


def catalog_ledger(rows):
    result=Counter()
    for r in rows:
        if r['kind']!='llm_output':continue
        t=r.get('planner_trace') or {};c=t.get('context')
        if not isinstance(c,dict):continue
        result['catalog_queries']+=1
        before=c.get('candidates',[])+c.get('preflight_withheld_candidates',[])
        result['nonempty_before_measurement_queries']+=bool(before or c.get('obstacle_candidates') or c.get('preflight_withheld_obstacles'))
        result['nonempty_after_measurement_queries']+=bool(c.get('candidates') or c.get('obstacle_candidates'))
        result['baseline_measurement_declines']+=(c.get('measurement_preflight') or {}).get('ready') is False
    n=result['catalog_queries']
    return dict(result,coverage_before_measurement=result['nonempty_before_measurement_queries']/n if n else None,
                coverage_after_measurement=result['nonempty_after_measurement_queries']/n if n else None)


def transport_ledger(records, expected, budget, log_present=True, logical_calls=None):
    by_scene=Counter(r.get('scenario_index') for r in records)
    unknown=[key for key in by_scene if key not in expected]
    failures=sum(bool(r.get('error_type')) for r in records)
    successful=[r for r in records if not r.get('error_type') and r.get('model_returned')]
    known_times=[r['elapsed_seconds'] for r in records if isinstance(r.get('elapsed_seconds'),(int,float))]
    known_usage=[r['usage'] for r in records if isinstance(r.get('usage'),dict) and isinstance(r['usage'].get('total_tokens'),int)]
    result=dict(actual_transport_calls=len(records),response_events=len(successful),error_events=failures,
                response_fraction=len(successful)/len(records) if records else None,
                calls_by_scene=[dict(scenario_index=k,calls=v) for k,v in by_scene.items()],
                unknown_scene_metadata=unknown,per_scene_budget_verified=not unknown and all(v<=budget for v in by_scene.values()),
                model_distribution=dict(Counter(r['model_returned'] for r in successful)),
                mean_request_seconds=statistics.mean(known_times) if known_times else None,
                median_request_seconds=statistics.median(known_times) if known_times else None,
                total_tokens=sum(r['total_tokens'] for r in known_usage),token_usage_records=len(known_usage))
    result['transport_log_present']=log_present
    if not log_present and logical_calls:
        # 早期冒烟没有传输记录层；缺记录不能写成“实际请求0次”。
        result.update(actual_transport_calls=None,response_events=None,error_events=None,
                      total_tokens=None,per_scene_budget_verified=False,
                      measurement_status='unavailable_transport_log; logical_requests_only')
    else:
        result['measurement_status']='recorded_transport' if log_present else 'zero_logical_requests_no_transport_log'
    return result


def enhanced_audit(root):
    report=audit_experiment(root);manifest=read(root/'manifest.json');launch=read(root/'launch.json')
    budget=launch.get('max_api_calls',launch.get('max_api_calls_per_scene',4))
    # 只读核对冻结输入内容，不重写清单或把缺失文件替换成新场景。
    expected=manifest.get('original_scenario_indices',manifest.get('scenario_indices',[]))
    failures=[]
    if len(manifest['scenario_files'])!=len(manifest['scenario_sha256']):
        failures.append('manifest_file_hash_count_mismatch')
    for name,digest in zip(manifest['scenario_files'],manifest['scenario_sha256']):
        p=Path(name)
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:failures.append(name)
    report['frozen_input_hash_failures']=failures
    for arm,g in report['groups'].items():
        files=sorted((root/arm).rglob('attempt_result.json'));attempts=[read(p) for p in files]
        g['exact_manifest_coverage']=manifest_coverage(manifest,attempts)
        ledgers=[];catalogs=[]
        for f in files:
            rows=[]
            for line in f.with_name('execution_trace.jsonl').read_text(encoding='utf-8').splitlines():
                try:rows.append(json.loads(line))
                except json.JSONDecodeError:continue
            ledgers.append(dict(scenario_index=read(f)['scenario_index'],**plan_execution_ledger(rows)))
            catalogs.append(dict(scenario_index=read(f)['scenario_index'],**catalog_ledger(rows)))
        g['plan_execution_ledgers']=ledgers;g['catalog_ledgers']=catalogs
        g['accepted_plans_from_trace']=sum(l['accepted_plans'] for l in ledgers)
        g['executed_plans']=sum(l['executed_plans'] for l in ledgers)
        g['unexecuted_plans']=sum(l['unexecuted_plans'] for l in ledgers)
        path=root/f'{arm}_api_calls.jsonl';transport=[];malformed=0
        if path.exists():
            for line in path.read_text(encoding='utf-8').splitlines():
                try:transport.append(json.loads(line))
                except json.JSONDecodeError:malformed+=1
        g['transport']=transport_ledger(transport,expected,budget,log_present=path.exists(),logical_calls=g['api_calls'])
        g['transport']['malformed_records']=malformed
        g['transport']['logical_transport_count_match']=(g['api_calls']==len(transport)) if (path.exists() or g['api_calls']==0) else None
        marker_ok=all(not (l['unmatched_execution_steps'] or l['execution_without_anchor_guidance_steps']
                         or l['duplicate_execution_markers'] or any(p['exceeds_declared_duration'] for p in l['plans'])) for l in ledgers)
        g['qualified_coverage_gate']=bool(g['execution_coverage_gate_pass'] and g['exact_manifest_coverage']['verified']
            and not failures and marker_ok and g['accepted_plans_from_trace']==g['attack_plans']
            and not (g['generation_failures'] or g['runtime_failures'] or g['planner_service_failures'])
            and g['transport']['per_scene_budget_verified'] and g['transport']['logical_transport_count_match'] and malformed==0)
        g['qualified_gate_scope']='Conservative artifact/execution/budget gate; not proof of general superiority, anchor fidelity, D validity or goal completion'
    report['audit_version']='enhanced_readonly_v2_20261020'
    control_file=root/'launcher_result.json';pid_file=root.with_suffix('.pid')
    reported=read(control_file).get('status') if control_file.exists() else None
    pid=None;alive=None
    if pid_file.exists():
        try:
            pid=int(pid_file.read_text().strip());os.kill(pid,0);alive=True
        except ProcessLookupError:alive=False
        except (ValueError,PermissionError):pass
    terminal_statuses={'completed','completed_with_failures','completed_with_group_failures',
        'final_planner_service_failure','prerequisite_gate_failed','launcher_failure'}
    # 未启动的方法没有exit_status，不能因此把已终态失败的控制器误判为仍在运行。
    report['terminal_control']=dict(reported_status=reported,pid=pid,pid_exists_now=alive,
                                    terminal=reported in terminal_statuses and alive is False)
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--experiment',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.experiment.resolve() in args.output.resolve().parents:raise ValueError('output must be outside immutable experiment')
    result=enhanced_audit(args.experiment)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:{n:g[n] for n in ['planned_scenes','actual_attack_scenes','attack_plans','executed_plans','unexecuted_plans','actual_attack_frames','generation_failures','qualified_coverage_gate']} for k,g in result['groups'].items()},ensure_ascii=False))


if __name__=='__main__':main()
