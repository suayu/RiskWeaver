"""只读导出全部规划机会的P1审查证据；不以契约、CV代理或自动填表认证正确性。"""
from collections import Counter
from pathlib import Path
import argparse
import csv
import hashlib
import html
import json
import math
import sys

WORK=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORK/'skills_scene_facts_prototype_20261028'))
from scene_facts import relation_card


class Inputs:
    def __init__(self):self.hashes={}

    def text(self,path):
        data=path.read_bytes();self.hashes[str(path)]=hashlib.sha256(data).hexdigest()
        return data.decode('utf-8')

    def json(self,path):return json.loads(self.text(path))

    def jsonl(self,path):
        if not path.exists():return [],'missing'
        values=[]
        for n,line in enumerate(self.text(path).splitlines(),1):
            if not line.strip():continue
            value=json.loads(line)
            if not isinstance(value,dict):raise ValueError('non_object_jsonl:'+str(path)+':'+str(n))
            values.append(value)
        return values,None

    def verify(self):
        if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h for p,h in self.hashes.items()):
            raise RuntimeError('source_changed_during_review_export')


def finite(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)


def actor_facts(state,step):
    """原始执行状态独立计算位姿事实；不把运动方向当作车道拓扑。"""
    if state is None:return [],['missing_source_state']
    agents=state.get('agents',[]);active=state.get('active',[]);facts=[];errors=[]
    if len(agents)!=len(active):errors.append('active_actor_array_length_mismatch')
    for target,on in enumerate(active):
        if not on:continue
        if target>=len(agents):errors.append('missing_active_actor:'+str(target));continue
        try:facts.append(relation_card(state['ego'],agents[target],target,step))
        except (ValueError,TypeError,IndexError,KeyError) as exc:errors.append('invalid_actor_state:'+str(target)+':'+type(exc).__name__)
    return facts,errors


def supplied_comparison(cards,selected):
    """仅比较所提供的完整有效期中心距CV差值，保留负值，不定义正确/错误战术。"""
    usable={cid:card['min_gap_difference_m'] for cid,card in cards.items()
        if isinstance(card,dict) and card.get('full_active_window_covered') is True and finite(card.get('min_gap_difference_m'))}
    known=usable.get(selected)
    return dict(selected_candidate_id=selected,selected_supplied_increment_m=known,
        best_supplied_increment_m=max(usable.values()) if usable else None,
        best_minus_selected_supplied_increment_m=max(usable.values())-known if known is not None else None,
        candidates_with_complete_supplied_proxy=len(usable),candidates_in_supplied_cards=len(cards),
        all_supplied_proxy_differences_m=usable,proxy_missing_or_incomplete_ids=sorted(set(cards)-set(usable)),
        correct_strategy_certified=False,actual_threat_certified=False,
        scope='descriptive_supplied_center_gap_CV_proxy; not_ground_truth_risk_or_closed_loop_causal_gain')


def review_record(row,states,calls,scene,ordinal,arm):
    step=row['step'];trace=row.get('planner_trace') or {};context=trace.get('context') or {}
    response=trace.get('response_summary') or {};plan=row.get('validated_plan') or {}
    candidates=context.get('candidates') or [];catalog={};issues=[]
    for c in candidates:
        cid=c.get('candidate_id')
        if not isinstance(cid,str) or cid in catalog:issues.append('missing_or_duplicate_catalog_id');continue
        catalog[cid]=c
    ranking=response.get('ranking') or []
    if any(not isinstance(cid,str) for cid in ranking):issues.append('unsupported_ranking_shape')
    else:
        if len(ranking)!=len(set(ranking)):issues.append('duplicate_ranked_ids')
        if catalog and any(cid not in catalog for cid in ranking):issues.append('ranked_id_not_in_context_catalog')
    if context.get('source_step') is not None and context['source_step']!=step:issues.append('context_source_time_mismatch')
    facts,errors=actor_facts(states.get(step),step);issues.extend(errors)
    selected=plan.get('candidate_id');selected_target=plan.get('attack_target_id')
    if plan.get('attack') and catalog:
        if selected not in catalog:issues.append('selected_id_missing_from_catalog')
        elif catalog[selected].get('target_id')!=selected_target:issues.append('selected_target_does_not_match_catalog')
    matched=[c for c in calls if c.get('scenario_index')==scene and c.get('source_step')==step]
    accepted=bool(plan.get('attack'))
    # “未接收”不等于已知no_attack；结构化原因缺失必须暴露，原始HTTP正文仍保留。
    reason=(row.get('planner_skipped_reason') or (plan.get('reason') if not accepted else None)
        or trace.get('error') or trace.get('validation_error') or trace.get('plan_validation')
        or (response.get('no_attack_reason') if not accepted else None))
    if not accepted and not reason:issues.append('structured_nonacceptance_reason_missing_or_not_standardized')
    return dict(review_id=f'{arm}:scene{scene}:step{step}:output{ordinal}',arm=arm,scenario_index=scene,source_step=step,
        plan_accepted=accepted,validated_plan=plan or None,explicit_skipped_reason=row.get('planner_skipped_reason'),
        request_failed=row.get('request_failed'),planner_stage=trace.get('stage'),structured_reason=reason,
        physical_call_matching_scope='same_scene_and_source_step; multiple_logical_outputs_at_one_step_may_share_journal_evidence',
        source_pose_facts=facts,scene_analysis=response.get('scene_analysis'),all_ranked_reasons=response.get('candidate_reasons'),
        raw_response_summary=response or None,ranking=ranking,selected_candidate=catalog.get(selected),
        all_context_candidates=candidates,all_priority_evidence=context.get('priority_evidence'),
        all_incremental_cards=context.get('incremental_cards'),
        compile_attempts=trace.get('compile_attempts'),plan_validation=trace.get('plan_validation'),
        context_candidate_rejections=context.get('candidate_rejections'),context_semantic_precheck=context.get('semantic_precheck'),
        context_measurement_preflight=context.get('measurement_preflight'),context_route_tail_preflight=context.get('route_tail_preflight'),
        supplied_proxy_comparison=supplied_comparison(context.get('incremental_cards') or {},selected),
        all_physical_requests_at_scene_step=matched,physical_request_count=len(matched),
        successful_models=[c.get('model_returned') for c in matched if c.get('model_returned') and not c.get('error_type')],
        execution_markers=[r for r in states.get('_execution_markers',[]) if r.get('target_id')==selected_target
            and r.get('diffusion_source_step',r['step']-1)>=step
            and r.get('diffusion_source_step',r['step']-1)<step+plan.get('duration',10)] if accepted else [],
        execution_marker_scope='raw_target_and_declared_time_markers_only; use_independent_ledger_for_guidance_attribution',
        evidence_issues=issues,
        review_status=dict(scene_facts='not_reviewed',time_and_identity='not_reviewed',
            all_ranked_reasons='not_reviewed',selected_strategy='not_reviewed',actual_threat='not_reviewed'),
        review_findings=[],overall_correctness=None,
        review_requirement='quotes_plus_source_fact_or_metric_required; unreviewed_never_counts_as_correct')


def build(root):
    root=Path(root);inputs=Inputs()
    if not (root/'launcher_result.json').is_file():raise ValueError('terminal_experiment_required')
    terminal=inputs.json(root/'launcher_result.json');manifest=inputs.json(root/'manifest.json');launch=inputs.json(root/'launch.json')
    scenes=manifest['original_scenario_indices'];arms=launch['arms']
    if not scenes or len(scenes)!=len(set(scenes)) or manifest['scenario_count']!=len(scenes):raise ValueError('manifest_denominator_mismatch')
    if not arms or len(arms)!=len(set(arms)):raise ValueError('arm_denominator_mismatch')
    records=[];groups={}
    for arm in arms:
        calls,missing_calls=inputs.jsonl(root/(arm+'_api_calls.jsonl'));seen=set();attempts=[];orphan_calls=[]
        for path in sorted((root/arm).rglob('attempt_result.json')):
            a=inputs.json(path);scene=a['scenario_index']
            if scene not in scenes or scene in seen:raise ValueError('duplicate_or_unknown_attempt')
            seen.add(scene);rows,missing_trace=inputs.jsonl(path.with_name('execution_trace.jsonl'))
            states={};outputs=[]
            for row in rows:
                if row.get('kind')=='state':
                    if row['step'] in states:raise ValueError('duplicate_source_state')
                    states[row['step']]=row
                if row.get('kind')=='llm_output':outputs.append(row)
            states['_execution_markers']=[r for r in rows if r.get('kind')=='attack_execution' and r.get('executed')]
            local_records=[review_record(row,states,calls,scene,n,arm) for n,row in enumerate(outputs)]
            records.extend(local_records)
            output_steps={row['step'] for row in outputs}
            orphan_calls.extend(c for c in calls if c.get('scenario_index')==scene and c.get('source_step') not in output_steps)
            attempts.append(dict(scenario_index=scene,attempted=True,planning_outputs=len(outputs),
                accepted_outputs=sum(r['plan_accepted'] for r in local_records),
                actual_attack_frames=a.get('attack_executed_frames'),executed_steps=a.get('executed_steps'),
                generation_failure=a.get('generation_failure'),runtime_failure=a.get('runtime_failure'),
                planner_service_failure_count=a.get('planner_service_failure_count'),
                missing_trace=missing_trace,reviewed_correct_opportunities=None,correct_attack_scene=None))
        # 保留未尝试与没有输出的场景，禁止用“无攻击”替代缺失记录。
        attempts.extend(dict(scenario_index=i,attempted=False,planning_outputs=0,accepted_outputs=0,
            actual_attack_frames=0,reviewed_correct_opportunities=None,correct_attack_scene=None,
            reason='unattempted_not_normal_no_attack') for i in scenes if i not in seen)
        unknown_calls=[c for c in calls if c.get('scenario_index') not in scenes]
        groups[arm]=dict(planned_scenes=len(scenes),attempted_scenes=len(seen),unattempted_scenes=len(scenes)-len(seen),
            planning_outputs=sum(a['planning_outputs'] for a in attempts),
            known_physical_requests=len(calls),missing_physical_journal=missing_calls,
            calls_without_matching_output=orphan_calls,unknown_scene_calls=unknown_calls,
            opportunities_with_fresh_response=sum(bool(r['successful_models']) for r in records if r['arm']==arm),
            accepted_outputs=sum(a['accepted_outputs'] for a in attempts),
            all_outputs_fully_reviewed=False,correct_attack_scenes=None,correct_attack_coverage=None,
            attempts=sorted(attempts,key=lambda a:scenes.index(a['scenario_index'])))
    inputs.verify()
    return dict(version='all_opportunity_review_export_v2_20261028',experiment=str(root),terminal_status=terminal.get('status'),groups=groups,records=records,
        planned_attempts=len(arms)*len(scenes),planning_outputs=len(records),
        source_sha256=inputs.hashes,inputs_unchanged=True,new_API_requests=0,new_execution_frames=0,
        runtime_changes=False,all_review_annotations_empty=True,goal_complete=False,
        analysis_source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),WORK/'skills_scene_facts_prototype_20261028/scene_facts.py']},
        rubric=dict(required_dimensions=['source_identity_time','current_pose_heading','evidence_numbers_and_comparisons',
            'every_ranked_reason','selected_strategy_applicability','actual_guided_execution','threat_evidence'],
            status_values=['not_reviewed','supported_with_cited_evidence','contradiction_with_cited_evidence','indeterminate'],
            no_automatic_correctness_from=['HTTP_success','schema_pass','grounding_pass','execution_marker','positive_CV_proxy'],
            reject_inference='no_catalog_or_no_plan_is_not_proof_no_attack_possible',
            priority_scope='P1 review preparation; P2 evidence export; NL/profile/static/target_modes_not_certified'))


def write(result,out):
    out=Path(out);root=Path(result['experiment'])
    if out.resolve()==root.resolve() or root.resolve() in out.resolve().parents:raise ValueError('do_not_write_experiment')
    out.mkdir(parents=True,exist_ok=False)
    (out/'review_evidence.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    with (out/'review_queue.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['review_id','组别','场景','源帧','接收计划','场景事实审查','全部排序理由审查','战术适用性审查','实际威胁审查','结论','证据引用'])
        for r in result['records']:writer.writerow([r['review_id'],r['arm'],r['scenario_index'],r['source_step'],r['plan_accepted'],'未审','未审','未审','未审','未知',''])
    doc='<meta charset="utf-8"><style>body{font:16px/1.7 Microsoft YaHei;max-width:1250px;margin:30px auto}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:8px}pre{white-space:pre-wrap;overflow-wrap:anywhere}summary{cursor:pointer}details{margin:12px 0}</style>'
    doc+='<h1>P1场景理解与策略选择：完整审查材料</h1><p>全部规划输出纳入，包含接收、拒绝、跳过和服务失败。每条材料提供源帧真实状态计算的位置/航向、全部排序理由、候选事实、原始SDK请求回复及未经认证的执行标记。所有人工/代理语义审查默认未审，不自动计算正确率；正CV差值不是实测威胁或战术正确证书。没有候选也不证明物理上不可攻击。</p>'
    doc+='<p><a href="review_evidence.json">完整JSON及输入SHA</a> · <a href="review_queue.csv">待审列表CSV</a></p>'
    doc+='<table><tr><th>组别</th><th>预定/尝试/未尝试</th><th>规划输出/接收</th><th>已知物理请求</th><th>正确攻击率</th></tr>'
    for arm,g in result['groups'].items():
        doc+='<tr><td>'+html.escape(arm)+'</td><td>'+f"{g['planned_scenes']}/{g['attempted_scenes']}/{g['unattempted_scenes']}"+'</td><td>'+f"{g['planning_outputs']}/{g['accepted_outputs']}"+'</td><td>'+str(g['known_physical_requests'])+'</td><td>未完成语义审查，未知</td></tr>'
    doc+='</table>'
    for r in result['records']:
        doc+='<details><summary>'+html.escape(r['review_id'])+'：'+('已接收' if r['plan_accepted'] else '未接收/跳过，原因须看原件')+'</summary><pre>'+html.escape(json.dumps(r,ensure_ascii=False,indent=2))+'</pre></details>'
    doc+='<h2>完整预定场景与未尝试</h2><pre>'+html.escape(json.dumps(result['groups'],ensure_ascii=False,indent=2))+'</pre>'
    (out/'index.html').write_text(doc,encoding='utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=build(args.experiment);write(result,args.output)
    print(json.dumps(dict(planned_attempts=result['planned_attempts'],planning_outputs=result['planning_outputs'],
        input_files=len(result['source_sha256']),all_annotations_unreviewed=True,new_API_requests=0,output=str(args.output)),ensure_ascii=True))


if __name__=='__main__':main()
