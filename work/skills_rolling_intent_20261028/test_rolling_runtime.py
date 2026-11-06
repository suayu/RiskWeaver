"""装配真实高级规划链，SDK使用替身；不产生API消费或新场景性能证据。"""
import copy
import json
import sys
from pathlib import Path
import pytest
from rolling_planner import RollingIntentPlanner,ROOT
sys.path.insert(0,str(ROOT/'work/skills_scene_facts_prototype_20261028'))
from test_fact_runtime import runtime as base_runtime,answer
from incremental_planner import IncrementalSkillsPlanner


def runtime(enabled,budget=4):
    _,planner,env,state=base_runtime(IncrementalSkillsPlanner,False,budget)
    return RollingIntentPlanner(env.cfg.sim,budget,extra_recipes=True,rolling_intent=enabled),planner,env,state


def test_default_off_actual_prompt_equal_to_original():
    seen=[]
    for enabled in [None,False]:
        if enabled is None:engine,planner,env,state=base_runtime(IncrementalSkillsPlanner,False)
        else:engine,planner,env,state=runtime(enabled)
        def request(prompt,scene_image=None):
            prefix,raw=prompt.rsplit('\n',1);p=json.loads(raw);r=answer(p);p['request_id']='normalized';seen.append((prefix,p));return '',r
        planner._request_attack_plan=request
        assert engine.plan(planner,env,state,['dynamic'])['attack']
        assert 'rolling_intent_protocol' not in planner.last_trace
    assert seen[0]==seen[1]


def test_enabled_actual_prompt_memory_and_all_original_validation_still_present():
    engine,planner,env,state=runtime(True);seen=[]
    def request(prompt,scene_image=None):
        p=json.loads(prompt.rsplit('\n',1)[1]);seen.append(p);return '',answer(p)
    planner._request_attack_plan=request
    result=engine.plan(planner,env,state,['dynamic'])
    assert result and result['attack'] and len(seen)==1
    c=seen[0]['context'];assert c['ego_future_reference']['uses_future_ground_truth'] is False
    assert c['rolling_intent']['original_semantic_CV_precheck_retained']
    assert 'proposal_audit' not in c['rolling_intent']
    assert planner.last_trace['rolling_episode_update']['updated'] and engine.memory.previous['accepted_not_execution_confirmed']
    assert planner.last_trace['compile_attempts'][-1]['accepted'] and planner._request_attack_plan is request
    assert planner.last_trace['rolling_intent_protocol']['safety_and_diffusion_changed'] is False


def test_rolling_does_not_add_requests_outside_original_budget():
    engine,planner,env,state=runtime(True,1);count=[]
    def request(prompt,scene_image=None):
        count.append(1);return '',answer(json.loads(prompt.rsplit('\n',1)[1]))
    planner._request_attack_plan=request
    assert engine.plan(planner,env,state,['dynamic'])['attack']
    r=engine.plan(planner,env,state,['dynamic'])
    assert not r['attack'] and r['reason']=='skills_api_budget_exhausted' and len(count)==1
    engine.reset_episode();assert engine.memory.previous is None and engine.calls==0


def test_invalid_switch_rejected_explicitly():
    _,_,env,_=runtime(False)
    with pytest.raises(ValueError,match='requires_boolean'):RollingIntentPlanner(env.cfg.sim,extra_recipes=True,rolling_intent='on')


def test_next_replan_keeps_memory_even_when_source_motion_removes_safe_candidates():
    engine,planner,env,state=runtime(True);seen=[]
    def request(prompt,scene_image=None):
        p=json.loads(prompt.rsplit('\n',1)[1]);seen.append(p);return '',answer(p)
    planner._request_attack_plan=request
    first=engine.plan(planner,env,state,['dynamic']);assert first['attack']
    env.current_step=11
    # 只移动确定性SDK替身的当前输入；这不是新的闭环运行或未来真值证据。
    env.ego_state=env.ego_state.copy();env.ego_state[:2]+=1.1*env.ego_state[2:4]
    env.data_dict['agent'][-1]=env.data_dict['agent'][-1].copy()
    env.data_dict['agent'][-1][:,:2]+=1.1*env.data_dict['agent'][-1][:,2:4]
    state=copy.deepcopy(state);state['ego_state']=env.ego_state.tolist()
    for a in state['agents']:a['state']=env.data_dict['agent'][-1][a['id']].tolist()
    if 'current_step' in state:state['current_step']=11
    env.get_state_for_planning=lambda:state
    result=engine.plan(planner,env,state,['continue same episode'])
    rolling=planner.last_trace['context']['rolling_intent'];old=rolling['previous']
    assert old['available'] and old['previous_lease_expired']
    assert old['previous']['source_step']==0 and old['previous']['target_id']==first['attack_target_id']
    assert all('continuity' in c for c in rolling['candidate_facts'].values())
    # 此确定性推进确实移除了全部可行候选；不能为了连续性强行调用或放行。
    assert len(seen)==1 and not result['attack'] and result['reason']=='no_feasible_candidates'
    assert engine.memory.previous['source_step']==0
