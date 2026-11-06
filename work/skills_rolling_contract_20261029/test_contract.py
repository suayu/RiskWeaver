"""契约、等价几何和真实调用链回归，全部零真实API。"""
import copy
import json
import sys
from pathlib import Path
import numpy as np
import pytest
from contract_planner import deduplicate_intents,render_contract_prompt,ContractRollingPlanner,ROOT
sys.path.insert(0,str(ROOT/'work/skills_scene_facts_prototype_20261028'))


def candidate(cid='old',offset=0.,target=0,strategy='lane_change',duration=10):
    return dict(candidate_id=cid,target_id=target,strategy=strategy,duration=duration,
        anchors=[[offset,0],[offset,1],[offset,2],[offset,3]])


def test_numeric_equivalent_recipe_keeps_first_original_coordinates_and_audit():
    items=[candidate(),candidate('near',1e-9),candidate('different',1e-4)];original=copy.deepcopy(items)
    result,audit=deduplicate_intents(items)
    assert [x['candidate_id'] for x in result]==['old','different'] and items==original
    assert result[0]['anchors']==items[0]['anchors'] and audit[0]['representative_id']=='old'
    assert audit[0]['maximum_anchor_deviation_m']==1e-9


@pytest.mark.parametrize('key,value',[('target_id',1),('strategy','cut_in'),('duration',20)])
def test_discrete_difference_not_merged(key,value):
    other=candidate('other');other[key]=value
    assert len(deduplicate_intents([candidate(),other])[0])==2


def test_euclidean_equivalence_is_rotation_invariant():
    first=candidate();second=candidate('second',.8e-6)
    before=deduplicate_intents([first,second])[1]
    rotation=np.array([[np.cos(.7),-np.sin(.7)],[np.sin(.7),np.cos(.7)]])
    for c in [first,second]:c['anchors']=(np.asarray(c['anchors'])@rotation.T).tolist()
    after=deduplicate_intents([first,second])[1]
    assert len(before)==len(after)==1


def test_bad_geometry_not_repaired_by_deduplicator():
    bad=candidate('bad');bad['anchors']=[[0,0]]
    assert deduplicate_intents([bad,bad])[0]==[bad,bad]


def test_repair_feedback_is_explicit_and_raw_list_response_preserved():
    data=dict(context=dict(numeric_intent_deduplication={'removed':['alias']}),repair_feedback=dict(
        reason='skills_missing_candidate_reasons',previous_response=dict(ranking=['a'],candidate_reasons=['bad type'])))
    text=render_contract_prompt('original\n'+json.dumps(data));rendered=json.loads(text.rsplit('\n',1)[1])
    feedback=rendered['repair_feedback']
    assert feedback['expected_candidate_reasons']['type']=='object' and feedback['expected_candidate_reasons']['keys']==['a']
    assert feedback['previous_response']['candidate_reasons']==['bad type']
    assert 'MUST NOT be an array/list' in text and 'numeric_intent_deduplication' not in rendered['context']
    assert data['context']['numeric_intent_deduplication']


def runtime(contract=True,dedup=True,budget=4):
    sys.path.insert(0,str(ROOT/'work/skills_scene_facts_prototype_20261028'))
    from test_fact_runtime import runtime as old_runtime
    from incremental_planner import IncrementalSkillsPlanner
    _,planner,env,state=old_runtime(IncrementalSkillsPlanner,False,budget)
    return ContractRollingPlanner(env.cfg.sim,budget,extra_recipes=True,rolling_intent=True,
        explicit_contract=contract,deduplicate=dedup),planner,env,state


def test_actual_chain_contract_and_dedup_preserve_old_validator_and_budget():
    from test_fact_runtime import answer
    engine,planner,env,state=runtime();seen=[]
    def request(prompt,scene_image=None):
        assert 'candidate_reasons MUST be a JSON OBJECT' in prompt
        payload=json.loads(prompt.rsplit('\n',1)[1]);seen.append(payload);return '',answer(payload)
    planner._request_attack_plan=request
    result=engine.plan(planner,env,state,['dynamic'])
    assert result and result['attack'] and engine.calls==len(seen)==1
    assert planner.last_trace['compile_attempts'][-1]['accepted']
    assert planner.last_trace['context']['numeric_intent_deduplication']['removed']
    assert planner._request_attack_plan is request


def test_array_is_rejected_then_one_explicit_repair_accepts_object_within_two_calls():
    from test_fact_runtime import answer
    engine,planner,env,state=runtime(budget=2);seen=[]
    def request(prompt,scene_image=None):
        payload=json.loads(prompt.rsplit('\n',1)[1]);seen.append(payload);result=answer(payload)
        if len(seen)==1:result['candidate_reasons']=list(result['candidate_reasons'].values())
        return '',result
    planner._request_attack_plan=request;result=engine.plan(planner,env,state,['dynamic'])
    assert result['attack'] and len(seen)==engine.calls==2
    assert seen[1]['repair_feedback']['expected_candidate_reasons']['type']=='object'
    assert isinstance(seen[1]['repair_feedback']['previous_response']['candidate_reasons'],list)
    assert planner.last_trace['schema_rejections'][0]['reason']=='skills_missing_candidate_reasons'
    result=engine.plan(planner,env,state,['dynamic'])
    assert not result['attack'] and result['reason']=='skills_api_budget_exhausted' and len(seen)==2


def test_persistent_array_failure_is_not_silently_accepted():
    from test_fact_runtime import answer
    engine,planner,env,state=runtime(budget=2)
    def request(prompt,scene_image=None):
        result=answer(json.loads(prompt.rsplit('\n',1)[1]));result['candidate_reasons']=list(result['candidate_reasons'].values());return '',result
    planner._request_attack_plan=request
    assert engine.plan(planner,env,state,['dynamic']) is None and engine.calls==2
    assert engine.memory.previous is None and planner.last_trace['validation_reason']=='skills_missing_candidate_reasons'


def test_switches_off_prompt_matches_original_rolling_path():
    from test_fact_runtime import answer
    from rolling_planner import RollingIntentPlanner
    seen=[]
    for original in [True,False]:
        engine,planner,env,state=runtime(False,False)
        if original:engine=RollingIntentPlanner(env.cfg.sim,4,extra_recipes=True,rolling_intent=True)
        def request(prompt,scene_image=None):
            prefix,raw=prompt.rsplit('\n',1);p=json.loads(raw);r=answer(p);p['request_id']='normalized';seen.append((prefix,p));return '',r
        planner._request_attack_plan=request;assert engine.plan(planner,env,state,['dynamic'])['attack']
    assert seen[0]==seen[1]
