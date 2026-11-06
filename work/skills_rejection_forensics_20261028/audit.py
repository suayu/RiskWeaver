"""原日志拒绝理由旁证与冻结纯函数重放；不改原trace，不运行API或仿真。"""
from pathlib import Path
import argparse
import ast
import copy
import hashlib
import json
import math
import re
import sys

WORK=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORK/'skills_stage_a_verification_20261023'))
from verify_stage_a import pure_function


def segments(text,scene_order):
    """只用明确场景初始化序号及规划源帧划段；重复/缺失关联不能自动猜测。"""
    result=[];current=None;scene=None;ordinal=None;initializations=[];unassociated=[]
    for n,line in enumerate(text.splitlines(),1):
        init=re.search(r'Initial data saved to .*scenario_(\d+)_initial\.json',line)
        if init:
            if current:current['end_line']=n-1;result.append(current);current=None
            ordinal=int(init[1]);scene=scene_order[ordinal] if ordinal<len(scene_order) else None
            initializations.append(ordinal)
        planning=re.search(r'\[Adversarial Generator\] Planning attack at step (\d+)',line)
        if planning:
            if current:current['end_line']=n-1;result.append(current)
            current=dict(scenario_index=scene,source_step=int(planning[1]),initialization_ordinal=ordinal,start_line=n,failure_lines=[])
        if '[大模型规划器] 本次高级攻击规划失败，已跳过且仿真继续：' in line:
            record=dict(line_number=n,text=line)
            if current and scene is not None:current['failure_lines'].append(record)
            else:unassociated.append(record)
    if current:current['end_line']=len(text.splitlines());result.append(current)
    ordered=initializations==list(range(len(initializations))) and all(i<len(scene_order) for i in initializations)
    return dict(segments=result,initialization_ordinals=initializations,initialization_order_verified=ordered,unassociated_failures=unassociated)


def unique_segment(parsed,scene,step):
    matches=[s for s in parsed['segments'] if s['scenario_index']==scene and s['source_step']==step]
    return matches[0] if parsed['initialization_order_verified'] and len(matches)==1 and len(matches[0]['failure_lines'])==1 else None


def interaction_threshold(source_text):
    """阈值必须从归档函数对应的比较表达式读取，不用当前仓库参数替代历史协议。"""
    tree=ast.parse(source_text);found=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Compare) and isinstance(node.left,ast.Name) and node.left.id=='closest_approach':
            if len(node.ops)==1 and isinstance(node.ops[0],ast.Gt) and len(node.comparators)==1 and isinstance(node.comparators[0],ast.Constant):
                found.append(node.comparators[0].value)
    if len(found)!=1 or not isinstance(found[0],(int,float)):raise ValueError('frozen_interaction_threshold_ambiguous')
    return float(found[0])


def aligned_distances(plan,normalized):
    """独立计算t=1/2/3锚点对应时刻与自车CV位置的距离，不混用时刻。"""
    ego=normalized['ego_state'];anchors=plan['anchors'];values=[];future=[]
    if len(anchors)!=4:raise ValueError('four_sparse_anchors_required')
    for t in [1,2,3]:
        p=[ego[0]+t*ego[2],ego[1]+t*ego[3]];future.append(p)
        values.append(math.hypot(anchors[t][0]-p[0],anchors[t][1]-p[1]))
    return values,future


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def audit(pack_path):
    pack=json.loads(pack_path.read_text(encoding='utf-8'));root=Path(pack['experiment'])
    if root.name!='skills_four_arm_incremental_development10_20261026_r2' or not (root/'launcher_result.json').exists():raise ValueError('exact_terminal_historical_run_required')
    source=root/'source_snapshot/policies/llm_adversarial_planner.py';hashes=json.loads((root/'source_sha256.json').read_text(encoding='utf-8'))
    if sha(source)!=hashes['policies/llm_adversarial_planner.py']:raise ValueError('archived_source_digest_mismatch')
    normalize=pure_function(source,'_normalize_env_state');validate=pure_function(source,'_validate_attack_plan');threshold=interaction_threshold(source.read_text(encoding='utf-8'))
    manifest_path=root/'manifest.json';manifest=json.loads(manifest_path.read_text(encoding='utf-8'));order=manifest['original_scenario_indices']
    log=root/'legacy_shared_recovery.log';parsed=segments(log.read_text(encoding='utf-8'),order)
    paths=[pack_path,source,root/'source_sha256.json',manifest_path,log,WORK/'skills_stage_a_verification_20261023/verify_stage_a.py']
    watched={str(p):sha(p) for p in paths};rows_by_scene={};traces={}
    for path in sorted((root/'legacy_shared_recovery').rglob('attempt_result.json')):
        a=json.loads(path.read_text(encoding='utf-8'));scene=a['scenario_index'];trace=path.with_name('execution_trace.jsonl')
        watched[str(path)]=sha(path);watched[str(trace)]=sha(trace)
        rows_by_scene[scene]=[json.loads(x) for x in trace.read_text(encoding='utf-8').splitlines()];traces[scene]=str(trace)
    assert len(rows_by_scene)==len(parsed['initialization_ordinals'])
    missing=[r for r in pack['records'] if 'structured_nonacceptance_reason_missing_or_not_standardized' in r['evidence_issues']]
    if len(missing)!=6:raise ValueError('this_historical_gap_audit_requires_six_gaps')
    records=[]
    for record in missing:
        scene=record['scenario_index'];step=record['source_step'];segment=unique_segment(parsed,scene,step)
        item=dict(review_id=record['review_id'],scenario_index=scene,source_step=step,associated=False,
            original_trace_reason_remains_missing=True,does_not_modify_originals=True)
        if segment is None:records.append(item);continue
        calls=record['all_physical_requests_at_scene_step'];item.update(associated=True,stdout_segment=segment,trace=traces[scene])
        text=segment['failure_lines'][0]['text']
        success=[c for c in calls if c.get('model_returned') and not c.get('error_type')]
        failures=[c for c in calls if c.get('error_type')]
        if len(success)==1 and not failures:
            plan=json.loads(success[0]['content']);inputs=[r['state'] for r in rows_by_scene[scene] if r.get('kind')=='llm_input' and r['step']==step]
            if len(inputs)!=1:raise ValueError('source_input_not_unique')
            state=inputs[0];before=copy.deepcopy(state);normalized=normalize(None,state);error=None
            try:validate(copy.deepcopy(plan),normalized)
            except ValueError as exc:error=str(exc)
            assert state==before and error and error in text
            distances,future=aligned_distances(plan,normalized);closest=min(distances);match=re.search(r'最近距离为 ([0-9.]+) 米',text)
            assert closest>threshold and match and f'{closest:.1f}'==match[1]
            source_prediction=next((r.get('joint') for r in rows_by_scene[scene] if r.get('kind')=='execution_prediction' and r['step']==step),None)
            metadata=(source_prediction or {}).get('metadata') or {}
            item.update(category='semantic_interaction_distance_rejection',http_success=True,model=success[0]['model_returned'],
                request_id=success[0]['request_id'],raw_plan=plan,source_state=state,normalized_source_state=normalized,
                threshold_m=threshold,aligned_anchor_times_seconds=[1,2,3],ego_CV_future_local=future,
                aligned_distances_m=distances,closest_approach_m=closest,frozen_validator_error=error,
                pure_validator_replay_agrees_with_stdout=True,coordinate_normalization='archived_normalize_env_state',
                background_execution_prediction=source_prediction,
                this_rejected_plan_guided_diffusion=bool(metadata.get('anchor_guidance_active')),
                prediction_scope='existing_background_future_if_any; not_a_trajectory_generated_for_the_rejected_anchors')
            assert not item['this_rejected_plan_guided_diffusion']
        elif not success and len(failures)==1 and 'Error code: 400' in text and 'enable_thinking' in text:
            assert record['request_failed'] is True
            item.update(category='service_parameter_failure_HTTP400',http_success=False,
                model_requested=failures[0]['model_requested'],request_id=failures[0]['request_id'],
                error_type=failures[0]['error_type'],parameter='enable_thinking',required_value=True,
                no_successful_plan_returned=True,raw_plan=None,
                verification='physical_API_journal_error_plus_exact_scene_step_stdout_and_trace_request_failed')
        else:item.update(category='unresolved',http_success=None,reason='not_unique_or_not_recognized; preserve_unknown')
        records.append(item)
    if any(sha(Path(p))!=digest for p,digest in watched.items()):raise RuntimeError('input_changed_during_forensic_audit')
    return dict(version='historical_missing_reason_forensics_20261028_v1',experiment=str(root),
        gaps=len(missing),associations=sum(r['associated'] for r in records),
        semantic_rejections=sum(r.get('category')=='semantic_interaction_distance_rejection' for r in records),
        service_parameter_failures=sum(r.get('category')=='service_parameter_failure_HTTP400' for r in records),
        unresolved=sum(r.get('category') in [None,'unresolved'] for r in records),records=records,
        original_traces_unchanged=True,source_sha256=watched,inputs_unchanged=True,
        new_API_requests=0,new_execution_frames=0,gate_changes=False,goal_complete=False,
        interpretation='five_specific_raw_plans_fail_actual_time_aligned_geometry; not_proof_that_scenes_have_no_other_feasible_attack; no_gate_optimality_claim')


def main():
    p=argparse.ArgumentParser();p.add_argument('--pack',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    result=audit(args.pack);root=Path(result['experiment']).resolve()
    if args.output.resolve()==root or root in args.output.resolve().parents:raise ValueError('do_not_write_original_experiment')
    with args.output.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k not in ['records','source_sha256']},ensure_ascii=True))


if __name__=='__main__':main()
