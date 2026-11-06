"""合成几何与归属反例，不生成新的仿真或真实攻击结果。"""
import copy
import pytest
from danger_acceptance import box_ttc,audit_episode,aggregate


def fixture(static=True):
    attempt=dict(scenario_index=1,status='completed',executed_steps=10,generation_failure=False,
        runtime_failure=False,planner_service_failure_count=0,prediction_service_failure_count=0,
        feasibility=dict(background_collision_frames=0,background_static_collision_frames=0),collision=False)
    obstacle=dict(id='o',center=[8,0],yaw=0,length=2,width=2)
    s0=dict(kind='state',step=0,initial=True,ego=[0,0,5,0,0,4,2],agents=[[8,0,0,0,0,2,2]],active=[True],static_obstacles=[])
    plan=dict(request_id='r',attack=not static,attack_target_id=-1 if static else 0,obstacle_plan=[{}] if static else [])
    rows=[s0,dict(kind='llm_output',validated_plan=plan)]
    if static:rows.append(dict(kind='static_attack_execution',step=0,request_id='r',obstacle_ids=['o']))
    for step in range(1,11):
        if not static:rows.append(dict(kind='attack_execution',step=step,request_id='r',target_id=0,executed=True))
        rows.append(dict(s0,step=step,initial=False,ego=[.5*step,0,5,0,0,4,2],static_obstacles=[obstacle] if static else []))
    return attempt,rows


def test_static_zero_velocity_first_contact():
    assert box_ttc([0,0,5,0,0,4,2],[8,0,0,0,0,2,2])==pytest.approx(1)
    assert box_ttc([0,0,5,0,0,4,2],[8,10,0,0,0,2,2]) is None


@pytest.mark.parametrize('static',[True,False])
def test_applied_and_attributed_near_conflict_counts(static):
    a,r=fixture(static);out=audit_episode(a,r,True)
    assert out['effective_danger_constructed'] and out['min_attack_related_OBB_TTC_s']<3


def test_existing_unattributed_static_does_not_count():
    a,r=fixture();r=[x for x in r if x['kind']!='static_attack_execution']
    assert not audit_episode(a,r,True)['effective_danger_constructed']


def test_background_failure_invalidates_success():
    a,r=fixture();a['feasibility']['background_static_collision_frames']=1
    assert not audit_episode(a,r,True)['effective_danger_constructed']


def test_unverified_video_does_not_certify():
    a,r=fixture();out=audit_episode(a,r)
    assert out['effective_danger_constructed'] is None and out['internal_evidence_passed']


def test_placement_overlapping_at_creation_is_invalid():
    a,r=fixture();r[-1]['static_obstacles'][0]['center']=[0,0]
    assert not audit_episode(a,r,True)['effective_danger_constructed']


def test_no_fallback_to_point_TTC_for_missing_dimensions():
    with pytest.raises(ValueError):box_ttc([0,0,5,0],[8,0,0,0,0,2,2])


def test_duplicate_or_missing_step_is_rejected():
    a,r=fixture();r[-1]['step']=9
    assert 'trace_state_count_mismatch' in audit_episode(a,r,True)['audit_issues']


def test_unrelated_background_near_conflict_is_not_attack_success():
    a,r=fixture(False)
    for row in r:
        if row.get('kind')=='state':row['agents']=[[80,20,0,0,0,2,2],[8,0,0,0,0,2,2]];row['active']=[True,True]
    assert not audit_episode(a,r,True)['effective_danger_constructed']


def test_missing_scenes_stay_in20_denominator():
    reports=[dict(scenario_index=i,effective_danger_constructed=True,collision_any=False) for i in range(18)]
    out=aggregate(list(range(20)),reports,True)
    assert out['effective_danger_rate']==.9 and out['missing_scenes']==[18,19]
    assert out['rows'][-1]['collision_any'] is None


def test_development3_does_not_meet_formal20_goal():
    reports=[dict(scenario_index=i,effective_danger_constructed=True) for i in range(3)]
    assert not aggregate([0,1,2],reports,True)['target_18_of20_met']


def test_integrity_and18_boundary_are_required():
    reports=[dict(scenario_index=i,effective_danger_constructed=True) for i in range(18)]
    assert not aggregate(list(range(20)),reports)['target_18_of20_met']
    assert aggregate(list(range(20)),reports,True)['target_18_of20_met']
    assert not aggregate(list(range(20)),reports[:-1],True)['target_18_of20_met']


def test_duplicate_scene_cannot_inflate_success():
    with pytest.raises(ValueError):aggregate([1,2],[dict(scenario_index=1),dict(scenario_index=1)],True)


def test_duplicate_execution_does_not_inflate_exposure():
    a,r=fixture(False)
    events=[x for x in r if x['kind']=='attack_execution']
    r=[x for x in r if x['kind']!='attack_execution']+[events[0]]*10
    out=audit_episode(a,r,True)
    assert out['actual_dynamic_frames']==1
    assert not out['effective_danger_constructed']
    assert 'duplicate_dynamic_execution_frame' in out['audit_issues']
