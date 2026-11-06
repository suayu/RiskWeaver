"""帧间交互、原约束保留、速度对齐及真实高级链装配的零API回归。"""
import copy
import json
import sys
from pathlib import Path
import numpy as np
import pytest
from temporal_intents import fractional_minimum,guarded_fractional_validator,velocity_aligned_candidates
from temporal_planner import TemporalCandidatePlanner,ROOT


def original_validator():
    from policies.llm_adversarial_planner import LLMAdversarialPlanner
    return LLMAdversarialPlanner._validate_attack_plan


def crossing_case(strategy='lane_change'):
    return dict(attack=True,attack_target_id=0,strategy=strategy,anchors=[[10,10],[10,-30],[10,-60],[10,-90]]),


def fixture():
    plan=crossing_case()[0]
    state=dict(ego_state=[0,0,0,0,0],agents=[dict(id=0,state=[10,10,0,-40,0])])
    return plan,state


def test_exact_fractional_minimum_finds_between_anchor_event():
    p,s=fixture();r=fractional_minimum(p['anchors'],s['ego_state'])
    assert r['relative_time_seconds']==.25 and r['min_center_gap_m']==10.
    with pytest.raises(ValueError,match='不能形成有效交互'):original_validator()(p,s)
    events=[];guarded_fractional_validator(original_validator(),events)(p,s)
    assert events[0]['accepted'] and events[0]['threshold_m']==15.


def test_fractional_rescue_still_runs_final_cut_in_corridor_check():
    p,s=fixture();p['strategy']='cut_in'
    with pytest.raises(ValueError,match='cut_in 终点'):guarded_fractional_validator(original_validator(),[])(p,s)


@pytest.mark.parametrize('mutation,reason',[('dense','连续轨迹'),('offset','第一个锚点'),('velocity','速度不连续')])
def test_original_preconditions_are_not_bypassed(mutation,reason):
    p,s=fixture()
    if mutation=='dense':p['controls']=[]
    elif mutation=='offset':p['anchors'][0][0]+=5
    else:s['agents'][0]['state'][2:4]=[0,0]
    with pytest.raises(ValueError,match=reason):guarded_fractional_validator(original_validator(),[])(p,s)


def test_fractional_still_rejects_no_near_interaction_and_reports_time():
    p,s=fixture();p['anchors']=[[100,10],[100,-30],[100,-60],[100,-90]];s['agents'][0]['state'][0]=100
    with pytest.raises(ValueError,match='超过15米交互门槛'):guarded_fractional_validator(original_validator(),[])(p,s)


def test_unknown_validator_revision_cannot_bypass_new_checks():
    def changed(plan,state):return None
    with pytest.raises(ValueError,match='requires_review'):guarded_fractional_validator(changed,[])


def test_fractional_result_invariant_to_rotation_and_translation():
    p,s=fixture();a=np.asarray(p['anchors'],float);e=np.asarray(s['ego_state'],float)
    before=fractional_minimum(a,e);q=np.array([[np.cos(.9),-np.sin(.9)],[np.sin(.9),np.cos(.9)]])
    a=a@q.T+[200,300];e[:2]=e[:2]@q.T+[200,300];e[2:4]=e[2:4]@q.T
    after=fractional_minimum(a,e)
    assert after['min_center_gap_m']==pytest.approx(before['min_center_gap_m'])
    assert after['relative_time_seconds']==pytest.approx(before['relative_time_seconds'])


def test_velocity_alignment_uses_current_measured_state_and_existing_acceleration_cap():
    prior=dict(anchors=[[0,0],[0,11],[0,24],[0,39]],source_step=0,target_id=0,strategy='lane_change',episode_id='p')
    values,fit=velocity_aligned_candidates(prior,[3,10,0,6],11)
    assert fit['available'] and len(values)==2
    for c in values:
        a=np.asarray(c['anchors']);assert np.array_equal(a[0],[3,10])
        assert np.linalg.norm((a[1]-a[0])-[0,6])<=1.9+1e-8
        assert np.max(np.linalg.norm(np.diff(a,n=2,axis=0),axis=1))<=3.8+1e-8


def test_nonquadratic_prior_is_not_falsely_extended_as_constant_acceleration():
    p=dict(anchors=[[0,0],[0,1],[0,4],[1,10]],source_step=0,target_id=0,strategy='lane_change',episode_id='p')
    values,fit=velocity_aligned_candidates(p,[0,1,0,1],11)
    assert values==[] and fit['reason']=='prior_not_constant_acceleration_sparse_recipe'


def test_actual_074_source_replay_is_screened_and_fractional_accepted():
    from replay_074 import replay
    r=replay();assert r['old_remaining_candidates']==14 and r['old_all_rejected_dynamics']
    assert len(r['candidates'])==2
    assert all(c['screen_rejection'] is None and c['fractional_semantic']['accepted'] for c in r['candidates'])
    assert any(not c['original_semantic']['accepted'] for c in r['candidates'])


def test_real_chain_exposes_continuation_facts_and_restores_original_validator():
    from replay_074 import source_fixture
    sys.path.insert(0,str(ROOT/'work/skills_scene_facts_prototype_20261028'))
    from test_fact_runtime import answer
    env,state,planner,builder,prior,old,trace=source_fixture();seen=[];original=planner._validate_attack_plan
    engine=TemporalCandidatePlanner(env.cfg.sim,4,extra_recipes=True,rolling_intent=True,explicit_contract=True,deduplicate=True,
        fractional_interaction=True,velocity_continuation=True,motion_evidence=True)
    engine.memory.previous=copy.deepcopy(prior)
    def request(prompt,scene_image=None):
        payload=json.loads(prompt.rsplit('\n',1)[1]);seen.append(payload)
        choices=payload['context']['temporal_candidate_evidence']['visible_continuation_candidate_ids']
        assert choices
        chosen=next(c for c in payload['context']['candidates'] if c['candidate_id']==choices[0])
        payload['context']['candidates']=[chosen];return '',answer(payload)
    planner._request_attack_plan=request;result=engine.plan(planner,env,state,['continue'])
    assert result and result['attack'] and result['candidate_id'].startswith('vc_') and len(seen)==1
    assert planner._validate_attack_plan is original and '_validate_attack_plan' not in planner.__dict__
    assert seen[0]['context']['temporal_candidate_evidence']['current_source_motion']['0']['lateral_relation']=='left'
    assert planner.last_trace['temporal_candidate_protocol']['fractional_semantic_events']
