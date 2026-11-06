"""真实规划链装配：SDK替身只用于离线结构/预算回归，不冒充新鲜LLM或执行。"""
from pathlib import Path
from types import SimpleNamespace as NS
import copy
import json

import numpy as np
from omegaconf import OmegaConf
from fact_planner import SceneFactPlanner,ROOT
from incremental_planner import IncrementalSkillsPlanner
from policies.llm_adversarial_planner import LLMAdversarialPlanner


def runtime(engine_type=SceneFactPlanner,enabled=True,budget=4):
    prior=ROOT/'experiments/skills_priority_v5_frozen_development10_pair_20261024_r1'
    item=next(x for x in json.loads((prior/'inputs.json').read_text()) if x['scenario_index']==74)
    state,frame=item['request']['state'],item['frame']
    cfg=OmegaConf.load(ROOT/'experiments/skills_modelpool_recovery_development10_paired_20261023_r3/skills_full_shared_recovery/resolved_config.yaml')
    route=np.asarray(state['route'])
    env=NS(ego_state=np.asarray(frame['ego']),data_dict={'agent':[np.asarray(frame['agents'])]},
        agent_active=np.asarray(frame['active']),current_step=0,current_scene_id='scenario_074',
        scenario_dict={'route':route,'lanes':[route]},get_static_obstacles=lambda:[],cfg=cfg,
        dt=.1,pending_joint_trajectory=None,get_state_for_planning=lambda:state)
    engine=engine_type(cfg.sim,budget,extra_recipes=True,**({'fact_grounding':enabled} if engine_type is SceneFactPlanner else {}))
    planner=object.__new__(LLMAdversarialPlanner)
    planner.allow_trajectory=True;planner.use_obstacles=False;planner.last_request_failed=False;planner.last_trace={}
    return engine,planner,env,state


def answer(payload):
    candidate=payload['context']['candidates'][0];cid=candidate['candidate_id']
    facts=payload['context']['priority_evidence'][cid]
    grounding=dict(target_id=candidate['target_id'],target_role='background_vehicle',active_horizon_seconds=1.,
        evidence_ids=[cid+':actor',cid+':validity',cid+':active',cid+':contrast'],interaction_basis='exploratory_probe',
        contrast_direction=facts['facts'][cid+':contrast']['proxy_direction'])
    if 'scene_fact_cards' in payload['context']:
        c=payload['context']['scene_fact_cards'][cid]
        grounding.update(scene_fact_id=c['fact_id'],source_step=c['source_step'],longitudinal_relation=c['longitudinal_relation'],lateral_relation=c['lateral_relation'],heading_relation=c['heading_relation'])
    return dict(request_id=payload['request_id'],state_hash=payload['state_hash'],catalog_hash=payload['catalog_hash'],
        ranking=[cid],scene_analysis='Fixture; typed facts only, prose not certified.',candidate_reasons={cid:'Exploratory only, not a measured attack effect.'},
        no_attack_reason='',grounding={cid:grounding})


def test_full_chain_fact_prompt_acceptance_and_original_request_restored():
    engine,planner,env,state=runtime();seen=[]
    def request(prompt,scene_image=None):
        payload=json.loads(prompt.rsplit('\n',1)[1]);seen.append((prompt,payload));return '',answer(payload)
    planner._request_attack_plan=request
    plan=engine.plan(planner,env,state,['pure dynamic'])
    assert plan and plan['attack'] and len(seen)==1
    assert planner._request_attack_plan is request
    assert planner.last_trace['scene_fact_protocol']['typed_validation']=='accepted'
    assert 'scene_fact_id, source_step' in seen[0][0]
    assert not planner.last_trace['scene_fact_protocol']['raw_prose_certified']


def test_false_claim_repairs_once_inside_same_four_request_budget():
    engine,planner,env,state=runtime();calls=[]
    def request(prompt,scene_image=None):
        payload=json.loads(prompt.rsplit('\n',1)[1]);calls.append(payload);r=answer(payload)
        if len(calls)%2:
            row=r['grounding'][r['ranking'][0]];row['longitudinal_relation']='behind' if row['longitudinal_relation']=='ahead' else 'ahead'
        return '',r
    planner._request_attack_plan=request
    for _ in range(2):
        result=engine.plan(planner,env,state,['pure dynamic'])
        assert result and result['attack']
    result=engine.plan(planner,env,state,['pure dynamic'])
    assert result and not result['attack'] and result['reason']=='skills_api_budget_exhausted'
    assert len(calls)==engine.calls==4
    assert calls[1]['repair_feedback']['reason']=='scene_fact_longitudinal_relation_mismatch'
    assert calls[1]['repair_feedback']['previous_response']['grounding'][calls[0]['context']['candidates'][0]['candidate_id']]['longitudinal_relation']!=calls[1]['context']['scene_fact_cards'][calls[0]['context']['candidates'][0]['candidate_id']]['longitudinal_relation']


def test_default_off_context_and_prompt_match_original_except_request_id():
    payloads=[]
    for typ,enabled in [(IncrementalSkillsPlanner,False),(SceneFactPlanner,False)]:
        engine,planner,env,state=runtime(typ,enabled)
        def request(prompt,scene_image=None):
            payload=json.loads(prompt.rsplit('\n',1)[1]);payloads.append((prompt.rsplit('\n',1)[0],copy.deepcopy(payload)));return '',answer(payload)
        planner._request_attack_plan=request
        assert engine.plan(planner,env,state,['pure dynamic'])['attack']
        assert 'scene_fact_protocol' not in planner.last_trace
        assert 'scene_fact_cards' not in payloads[-1][1]['context']
    for _,payload in payloads:payload['request_id']='normalized_random_request_id'
    assert payloads[0]==payloads[1]
