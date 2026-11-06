"""完整分母、事实几何、拒绝与缺失、负代理和写入边界的必要回归。"""
from pathlib import Path
import copy
import json
import math
import pytest
import build_review as r


def test_negative_proxy_is_retained_and_never_certifies_correctness():
    cards={'a':dict(full_active_window_covered=True,min_gap_difference_m=-3.),
        'b':dict(full_active_window_covered=True,min_gap_difference_m=-1.),
        'c':dict(full_active_window_covered=False,min_gap_difference_m=100.),
        'd':dict(full_active_window_covered=True,min_gap_difference_m=None)}
    out=r.supplied_comparison(cards,'a')
    assert out['selected_supplied_increment_m']==-3. and out['best_supplied_increment_m']==-1.
    assert out['best_minus_selected_supplied_increment_m']==2.
    assert out['proxy_missing_or_incomplete_ids']==['c','d']
    assert not out['correct_strategy_certified'] and not out['actual_threat_certified']
    assert r.supplied_comparison(cards,'unknown')['selected_supplied_increment_m'] is None


def test_true_pose_is_independent_of_false_free_prose_and_time_mismatch_is_visible():
    state=dict(ego=[0,0,0,2,math.pi/2],agents=[[1,-20,0,3,math.pi]],active=[True])
    row=dict(step=3,validated_plan=dict(attack=True,attack_target_id=0,candidate_id='a'),planner_trace=dict(
        context=dict(source_step=4,candidates=[dict(candidate_id='a',target_id=0)]),
        response_summary=dict(scene_analysis='Target0 is ahead-left and oncoming.',ranking=['a'],candidate_reasons={'a':'test'})))
    original=copy.deepcopy(row)
    result=r.review_record(row,{3:state},[],140,0,'full')
    facts=result['source_pose_facts'][0]
    assert facts['longitudinal_relation']=='behind' and facts['lateral_relation']=='right'
    assert facts['heading_relation']=='crossing_headings'
    assert result['scene_analysis']==row['planner_trace']['response_summary']['scene_analysis']
    assert 'context_source_time_mismatch' in result['evidence_issues']
    assert result['overall_correctness'] is None and row==original


def test_rejected_successful_http_and_missing_reason_are_preserved_not_certified():
    row=dict(step=0,validated_plan=None,planner_trace={},request_failed=False)
    calls=[dict(scenario_index=53,source_step=0,model_returned='modelA',content='raw rejected response'),
           dict(scenario_index=53,source_step=1,model_returned='modelB')]
    result=r.review_record(row,{},calls,53,0,'old')
    assert result['physical_request_count']==1 and result['successful_models']==['modelA']
    assert result['structured_reason'] is None and result['overall_correctness'] is None
    assert 'structured_nonacceptance_reason_missing_or_not_standardized' in result['evidence_issues']
    assert 'missing_source_state' in result['evidence_issues']
    assert result['all_physical_requests_at_scene_step'][0]['content']=='raw rejected response'


@pytest.mark.parametrize('extra,expected',[
    (dict(validated_plan=dict(attack=False,reason='no_feasible_candidates')),'no_feasible_candidates'),
    (dict(planner_trace=dict(plan_validation='plan_validation_failed')),'plan_validation_failed'),
    (dict(planner_trace=dict(response_summary=dict(no_attack_reason='no_interaction'))),'no_interaction'),
])
def test_original_structured_decline_fields_are_not_marked_missing(extra,expected):
    row=dict(step=0,validated_plan=None,planner_trace={},request_failed=False);row.update(extra)
    result=r.review_record(row,{},[],53,0,'old')
    assert result['structured_reason']==expected
    assert 'structured_nonacceptance_reason_missing_or_not_standardized' not in result['evidence_issues']


def fixture(tmp_path):
    root=tmp_path/'experiment';root.mkdir()
    (root/'launcher_result.json').write_text(json.dumps(dict(status='completed_with_group_failures')))
    (root/'manifest.json').write_text(json.dumps(dict(scenario_count=2,original_scenario_indices=[53,220])))
    (root/'launch.json').write_text(json.dumps(dict(arms=['old','full'])))
    arm=root/'old'/'movies'/'scenario_053';arm.mkdir(parents=True)
    (arm/'attempt_result.json').write_text(json.dumps(dict(scenario_index=53,attack_executed_frames=0,
        executed_steps=0,generation_failure=True,runtime_failure=False,planner_service_failure_count=0)))
    (arm/'execution_trace.jsonl').write_text(json.dumps(dict(kind='llm_output',step=0,
        planner_skipped_reason='no_feasible_candidates',validated_plan=None))+'\n')
    return root,arm


def test_all_arms_all_scenes_missing_journal_and_unattempted_stay_in_denominator(tmp_path):
    root,_=fixture(tmp_path);result=r.build(root)
    assert result['planned_attempts']==4 and result['planning_outputs']==1
    assert result['groups']['old']['unattempted_scenes']==1 and result['groups']['full']['unattempted_scenes']==2
    assert result['groups']['old']['missing_physical_journal']=='missing'
    assert result['groups']['full']['correct_attack_coverage'] is None
    assert result['records'][0]['explicit_skipped_reason']=='no_feasible_candidates'
    assert result['records'][0]['overall_correctness'] is None and result['all_review_annotations_empty']
    assert result['groups']['old']['attempts'][1]['reason']=='unattempted_not_normal_no_attack'
    assert not result['goal_complete'] and result['inputs_unchanged']


def test_running_trial_is_not_collected_as_terminal_and_originals_not_written(tmp_path):
    root,_=fixture(tmp_path);(root/'launcher_result.json').unlink()
    with pytest.raises(ValueError,match='terminal_experiment_required'):r.build(root)
    result=dict(experiment=str(root))
    with pytest.raises(ValueError,match='do_not_write_experiment'):r.write(result,root/'derived')
    assert not (root/'derived').exists()


def test_duplicate_or_unknown_attempt_is_rejected(tmp_path):
    root,arm=fixture(tmp_path);p=arm.parent/'duplicate';p.mkdir()
    (p/'attempt_result.json').write_bytes((arm/'attempt_result.json').read_bytes())
    with pytest.raises(ValueError,match='duplicate_or_unknown_attempt'):r.build(root)


def test_evidence_input_mutation_detected(tmp_path):
    p=tmp_path/'source.json';p.write_text('{}');inputs=r.Inputs();inputs.json(p);p.write_text('{"changed":true}')
    with pytest.raises(RuntimeError,match='source_changed'):inputs.verify()


def test_invalid_geometry_is_reported_without_inventing_zero_distance():
    facts,issues=r.actor_facts(dict(ego=[0,0,0,0,0],agents=[[float('nan'),0,0,0,0]],active=[True]),0)
    assert facts==[] and issues==['invalid_actor_state:0:ValueError']

