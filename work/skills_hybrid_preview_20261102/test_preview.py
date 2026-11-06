"""混合路由作用域回归；真实时序类与校验器，所有传输均显式替身。"""
import copy
import json
from pathlib import Path
import sys
import pytest

ROOT=Path('/home2/zhaoyx/scenario-dreamer')
sys.path.insert(0,str(ROOT))
from work.skills_hybrid_preview_20261102.probe_case import build_case,inspect_case,DynamicFirstStaticPlanner
from work.skills_hybrid_preview_20261102.runtime import install


def routed_case():
    cfg,env,low,dyn,candidate=build_case()
    hybrid=DynamicFirstStaticPlanner(dyn,cfg.sim,4)
    return cfg,env,low,dyn,hybrid,candidate


def test_original_sequence_reproduces_rejection_and_wrong_static_routing():
    result=inspect_case()
    assert result['original_preview_candidates']==0
    assert result['under_plan_semantics_candidates']==1
    assert result['old_routing']==['static']
    assert result['baseline_measurement_preflight']['ready']
    assert result['fractional_events'][0]['fractional_result']['min_center_gap_m']==14.


def test_enabled_selects_dynamic_after_same_existing_fractional_preview():
    _,env,low,dyn,hybrid,_=routed_case();branches=[]
    dyn.plan=lambda *a,**k:branches.append('dynamic') or dict(attack=True)
    hybrid.static.plan=lambda *a,**k:branches.append('static') or dict(attack=False)
    original=low._validate_attack_plan
    with install(True) as receipt:plan=hybrid.plan(low,env,env.get_state_for_planning(),[])
    assert branches==['dynamic'] and plan['attack']
    preview=receipt['calls'][0]['previews']
    assert len(preview)==1 and preview[0]['published_candidates']==1
    assert low._validate_attack_plan==original
    assert not hasattr(dyn,'_consistent_preview_active')


def test_real_temporal_plan_has_no_nested_validator_guard_or_extra_requests():
    _,env,low,dyn,hybrid,_=routed_case();requests=[]
    def response(prompt,scene_image=None):
        p=json.loads(prompt.rsplit('\n',1)[1]);requests.append(p)
        candidate=p['context']['candidates'][0];cid=candidate['candidate_id']
        evidence=p['context']['priority_evidence'][cid]
        result={k:p[k] for k in ['request_id','state_hash','catalog_hash']}
        result.update(ranking=[cid],scene_analysis='DETERMINISTIC_FIXTURE_NOT_LLM; exploratory candidate.',
            candidate_reasons={cid:'Explicit source-CV hypothesis, not executed risk.'},no_attack_reason='',
            grounding={cid:dict(target_id=0,target_role='background_vehicle',active_horizon_seconds=1.,
                evidence_ids=[cid+':actor',cid+':validity',cid+':active',cid+':contrast'],
                interaction_basis='exploratory_probe',contrast_direction=evidence['facts'][cid+':contrast']['proxy_direction'])})
        return '',result
    low._request_attack_plan=response
    old_validator=low._validate_attack_plan;old_context=dyn.build_context
    with install(True):plan=hybrid.plan(low,env,env.get_state_for_planning(),['challenge'])
    assert plan and plan['attack'] and plan['attack_kind']=='dynamic'
    assert len(requests)==hybrid.calls==1 and hybrid.static.calls==0
    assert low.last_trace['plan_validation']=='accepted'
    assert low.last_trace['temporal_candidate_protocol']['fractional_interaction']
    assert low.last_trace['hybrid_preview_protocol']['previews'][0]['published_candidates']==1
    assert low._validate_attack_plan==old_validator and dyn.build_context==old_context


@pytest.mark.parametrize('mutation',['outside_radius','cut_in','bad_acceleration'])
def test_scope_does_not_skip_original_rejections(mutation):
    _,env,low,dyn,hybrid,candidate=routed_case();branches=[]
    if mutation=='outside_radius':
        for anchor in candidate['anchors']:anchor[0]+=2.
        env.data_dict['agent'][-1][0,0]+=2.
    elif mutation=='cut_in':candidate['strategy']='cut_in'
    else:candidate['anchors'][2][0]+=30.
    dyn.plan=lambda *a,**k:branches.append('dynamic') or dict(attack=True)
    hybrid.static.plan=lambda *a,**k:branches.append('static') or dict(attack=False)
    with install(True):hybrid.plan(low,env,env.get_state_for_planning(),[])
    assert branches==['static']


def test_off_preserves_class_and_old_routing():
    _,env,low,dyn,hybrid,_=routed_case();before=DynamicFirstStaticPlanner.plan
    branches=[];dyn.plan=lambda *a,**k:branches.append('dynamic') or dict(attack=True)
    hybrid.static.plan=lambda *a,**k:branches.append('static') or dict(attack=False)
    with install() as receipt:
        assert DynamicFirstStaticPlanner.plan is before
        hybrid.plan(low,env,env.get_state_for_planning(),[])
    assert branches==['static'] and receipt['calls']==[]


def test_disabled_fractional_feature_retains_old_semantics():
    _,env,low,dyn,hybrid,_=routed_case();dyn.fractional_interaction=False;branches=[]
    dyn.plan=lambda *a,**k:branches.append('dynamic') or dict(attack=True)
    hybrid.static.plan=lambda *a,**k:branches.append('static') or dict(attack=False)
    with install(True) as receipt:hybrid.plan(low,env,env.get_state_for_planning(),[])
    assert branches==['static']
    assert receipt['calls']==[dict(applied=False,reason='fractional_interaction_disabled')]


def test_existing_budget_is_not_bypassed():
    _,env,low,dyn,hybrid,_=routed_case();dyn.calls=4
    def forbidden(*a,**k):pytest.fail('budget was bypassed')
    low._request_attack_plan=forbidden
    with install(True):plan=hybrid.plan(low,env,env.get_state_for_planning(),[])
    assert not plan['attack'] and plan['reason']=='skills_api_budget_exhausted'
    assert hybrid.calls==4 and hybrid.static.calls==0


def test_runtime_error_restores_instance_and_class_state():
    _,env,low,dyn,hybrid,_=routed_case()
    old_validator=low._validate_attack_plan;old_context=dyn.build_context;old_plan=DynamicFirstStaticPlanner.plan
    def fail(*a,**k):raise RuntimeError('fixture context failure')
    dyn.builder.build=fail
    with pytest.raises(RuntimeError,match='fixture context'):
        with install(True):hybrid.plan(low,env,env.get_state_for_planning(),[])
    assert low._validate_attack_plan==old_validator and dyn.build_context==old_context
    assert DynamicFirstStaticPlanner.plan is old_plan and not hasattr(dyn,'_consistent_preview_active')


def test_nested_install_is_rejected_and_restored():
    before=DynamicFirstStaticPlanner.plan
    with install(True):
        with pytest.raises(RuntimeError,match='install_already_active'):
            with install(True):pass
    assert DynamicFirstStaticPlanner.plan is before


@pytest.mark.parametrize('value',['on',1,None])
def test_flags_are_booleans(value):
    with pytest.raises(ValueError,match='requires_boolean'):
        with install(value):pass
