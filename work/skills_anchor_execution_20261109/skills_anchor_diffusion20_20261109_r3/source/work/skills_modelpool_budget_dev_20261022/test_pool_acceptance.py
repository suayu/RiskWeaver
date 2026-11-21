"""多数覆盖必须来自完整的新鲜LLM执行，不能删除失败或用标记替代。"""
import copy
import pytest
from pool_acceptance import qualify_scene,paired_gate


def fixture():
    attempt=dict(scenario_index=53,status='completed',executed_steps=188,D=.2,D_valid_frames=163,
        actual_attack_frames=10,completeness=dict(trace_exists=True,terminal_state_present=True,
        state_sequence_complete=True,video_frames_match=True,malformed_trace_lines=0))
    ledger=dict(executed_plans=1,plans=[dict(exceeds_declared_duration=False)],
                unmatched_execution_steps=[],execution_without_anchor_guidance_steps=[],duplicate_execution_markers=[])
    return attempt,ledger,[dict(model_returned='unit-model')]


def test_multiple_physical_requests_allowed_even_for_one_logical_call():
    attempt,ledger,requests=fixture()
    requests=[dict(error_type='quota')]*3+requests
    result=qualify_scene(attempt,ledger,requests,4)
    assert result['qualified'] and result['physical_requests']==4


@pytest.mark.parametrize('field,value,reason',[
    ('generation_failure',True,'generation_failure'),('runtime_failure',True,'runtime_failure'),
    ('planner_service_failures',1,'planner_service_failure'),('background_collision_frames',1,'background_collision'),
    ('background_static_collision_frames',1,'background_collision'),('D',None,'invalid_D'),
    ('D_valid_frames',0,'invalid_D'),('actual_attack_frames',0,'no_executed_plan'),
    ('status','generation_failure','episode_not_completed')])
def test_failed_scene_never_counts_as_qualified(field,value,reason):
    attempt,ledger,requests=fixture();attempt[field]=value
    result=qualify_scene(attempt,ledger,requests)
    assert not result['qualified'] and reason in result['reasons']


def test_fixed_plan_replay_cannot_count_as_fresh_generation():
    attempt,ledger,_=fixture()
    assert 'no_fresh_model_response' in qualify_scene(attempt,ledger,[])['reasons']


def test_budget_overrun_and_unguided_marker_are_rejected():
    attempt,ledger,requests=fixture();ledger['execution_without_anchor_guidance_steps']=[3]
    result=qualify_scene(attempt,ledger,requests*5)
    assert not result['qualified'] and {'execution_attribution_invalid','physical_request_budget_exceeded'}<=set(result['reasons'])


def test_majority_is_six_of_planned_ten_not_of_remaining_successes():
    groups={'legacy_shared_recovery':dict(group_integrity_pass=True,qualified_scenes=3),
            'skills_full_shared_recovery':dict(group_integrity_pass=True,qualified_scenes=5)}
    assert not paired_gate(groups)['passed']
    groups['skills_full_shared_recovery']['qualified_scenes']=6
    assert paired_gate(groups)['passed']
    groups['legacy_shared_recovery']['qualified_scenes']=7
    assert not paired_gate(groups)['passed']


def test_service_or_source_integrity_failure_blocks_validation():
    groups={'legacy_shared_recovery':dict(group_integrity_pass=True,qualified_scenes=3),
            'skills_full_shared_recovery':dict(group_integrity_pass=False,qualified_scenes=9)}
    assert not paired_gate(groups)['passed']
