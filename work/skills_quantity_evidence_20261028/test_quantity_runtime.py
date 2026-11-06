"""旧真实规划链与SDK替身的装配回归；零真实HTTP、不冒充性能结果。"""
from pathlib import Path
import copy
import json
import sys
import pytest
from quantity_planner import QuantityEvidencePlanner,ROOT
sys.path.insert(0,str(ROOT/'work/skills_scene_facts_prototype_20261028'))
from test_fact_runtime import runtime as old_runtime,answer
from incremental_planner import IncrementalSkillsPlanner


def runtime(enabled,budget=4):
    _,planner,env,state=old_runtime(IncrementalSkillsPlanner,False,budget)
    return QuantityEvidencePlanner(env.cfg.sim,budget,extra_recipes=True,quantity_evidence=enabled),planner,env,state


def test_enabled_actual_chain_receives_quantity_facts_and_same_response_contract():
    engine,planner,env,state=runtime(True);seen=[]
    def request(prompt,scene_image=None):
        payload=json.loads(prompt.rsplit('\n',1)[1]);seen.append((prompt,payload));return '',answer(payload)
    planner._request_attack_plan=request
    result=engine.plan(planner,env,state,['pure dynamic'])
    assert result and result['attack'] and len(seen)==1
    assert seen[0][1]['context']['quantity_facts']['available']
    assert 'positive `radial_gap_rate_mps`' in seen[0][0]
    assert planner._request_attack_plan is request
    assert planner.last_trace['quantity_evidence_protocol']['new_acceptance_gates']==0
    assert planner.last_trace['quantity_evidence_protocol']['response_schema_unchanged']
    assert not planner.last_trace['quantity_evidence_protocol']['prose_and_tactics_and_actual_threat_certified']


def test_disabled_actual_prompt_matches_old_except_random_id():
    seen=[]
    for enabled in [None,False]:
        if enabled is None:engine,planner,env,state=old_runtime(IncrementalSkillsPlanner,False)
        else:engine,planner,env,state=runtime(enabled)
        def request(prompt,scene_image=None):
            prefix,raw=prompt.rsplit('\n',1);payload=json.loads(raw);payload['request_id']='normalized';seen.append((prefix,payload))
            original=json.loads(raw);return '',answer(original)
        planner._request_attack_plan=request;assert engine.plan(planner,env,state,['pure dynamic'])['attack']
        assert 'quantity_evidence_protocol' not in planner.last_trace
    assert seen[0]==seen[1] and 'quantity_facts' not in seen[0][1]['context']


def test_old_validator_repair_and_four_logical_budget_are_not_bypassed():
    engine,planner,env,state=runtime(True);calls=[]
    def request(prompt,scene_image=None):
        payload=json.loads(prompt.rsplit('\n',1)[1]);calls.append(payload);r=answer(payload)
        if len(calls)%2:r['grounding'][r['ranking'][0]]['contrast_direction']='invented_direction'
        return '',r
    planner._request_attack_plan=request
    assert engine.plan(planner,env,state,['pure dynamic'])['attack']
    assert engine.plan(planner,env,state,['pure dynamic'])['attack']
    last=engine.plan(planner,env,state,['pure dynamic'])
    assert not last['attack'] and last['reason']=='skills_api_budget_exhausted'
    assert len(calls)==engine.calls==4 and planner._request_attack_plan is request
    assert calls[1]['repair_feedback']['reason']=='incremental_direction_contradicts_fact'
    assert calls[1]['repair_feedback']['previous_response']['grounding'][calls[0]['context']['candidates'][0]['candidate_id']]['contrast_direction']=='invented_direction'


def test_boolean_switch_required():
    _,_,env,_=runtime(False)
    with pytest.raises(ValueError,match='requires_boolean'):QuantityEvidencePlanner(env.cfg.sim,quantity_evidence=1)
