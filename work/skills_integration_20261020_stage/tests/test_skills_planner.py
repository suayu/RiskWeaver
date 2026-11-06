"""技能链路的身份、物理边界、有限恢复与开关隔离回归。"""
import copy
import json
from types import SimpleNamespace as NS
import numpy as np
import pytest
from test_ego_profile import environment,planner as fixture_planner
from policies.skills_planner import SkillsAttackPlanner
from policies.skill_registry import SkillRegistry
from policies.intent_validation import validate_world_intent


def setup(expansion=False,repairs=1,budget=4):
    sim=NS(ego_profile=NS(),llm=NS(skills=NS(enabled=True,candidate_expansion_enabled=expansion,max_repairs=repairs)))
    engine=SkillsAttackPlanner(sim,budget)
    p=fixture_planner();p.provider='dashscope';p.use_obstacles=False
    return environment(),p,engine


def install(p,transform=lambda x:x):
    p.prompts=[]
    def request(prompt,scene_image=None):
        p.prompts.append(prompt)
        payload=json.loads(prompt.split('\n',1)[1]);ids=[c['candidate_id'] for c in payload['context']['candidates']]
        reply={k:payload[k] for k in ('request_id','state_hash','catalog_hash')}
        reply.update(ranking=ids,scene_analysis='Closing forward vehicle based on supplied distances',
                     candidate_reasons={cid:'small aligned distance, screened background' for cid in ids},no_attack_reason='')
        return '',transform(reply)
    p._request_attack_plan=request


def test_skill_routing_and_hashes():
    registry=SkillRegistry()
    assert not any('static' in d['path'] for d in registry.load()['documents'])
    assert any('static' in d['path'] for d in registry.load(True)['documents'])
    assert not SkillRegistry(level='none').load()['documents']


def test_profile_off_plan_has_reason_identity_and_no_ego_profile():
    env,p,engine=setup();install(p)
    plan=engine.plan(p,env,env.get_state_for_planning(),['probe'])
    assert plan['attack'] and plan['request_id'] and plan['reason']
    assert p.last_trace['context']['profile'] is None
    assert p.last_trace['plan_validation']=='accepted'
    assert p.last_trace['api_calls_total']==1


def test_empty_scene_no_api_not_service_failure():
    env,p,engine=setup();install(p);env.agent_active[:]=False
    env.get_state_for_planning=lambda:dict(ego_state=env.ego_state[:5].tolist(),agents=[],route=env.scenario_dict['route'].tolist(),current_step=0)
    result=engine.plan(p,env,env.get_state_for_planning(),[])
    assert result['reason']=='no_feasible_candidates' and not p.prompts and not p.last_request_failed


@pytest.mark.parametrize('field,value',[('request_id','stale'),('state_hash','old'),('catalog_hash','old'),('ranking',['invented']),('candidate_reasons',{})])
def test_invalid_identity_or_schema_rejected(field,value):
    env,p,engine=setup(repairs=0);install(p,lambda r:dict(r,**{field:value}))
    assert engine.plan(p,env,env.get_state_for_planning(),[]) is None
    assert p.last_trace['plan_validation']=='plan_validation_failed'
    assert not p.last_request_failed


def test_one_repair_stays_within_scene_budget():
    env,p,engine=setup(budget=2)
    count=[0]
    def corrupt_first(reply):
        count[0]+=1
        return dict(reply,request_id='wrong') if count[0]==1 else reply
    install(p,corrupt_first)
    assert engine.plan(p,env,env.get_state_for_planning(),[])['attack']
    assert len(p.prompts)==2
    assert engine.plan(p,env,env.get_state_for_planning(),[])['reason']=='skills_api_budget_exhausted'
    assert len(p.prompts)==2


def test_service_error_remains_service_failure():
    env,p,engine=setup()
    from policies.llm_adversarial_planner import PlannerServiceFailure
    p._request_attack_plan=lambda *a,**kw:(_ for _ in ()).throw(PlannerServiceFailure('unavailable'))
    assert engine.plan(p,env,env.get_state_for_planning(),[]) is None
    assert p.last_request_failed and p.last_trace['plan_validation']=='planner_service_failure'


def test_stale_state_after_transport_rejected():
    env,p,engine=setup()
    def advance(reply):env.current_step+=1;return reply
    install(p,advance)
    assert engine.plan(p,env,env.get_state_for_planning(),[]) is None
    assert p.last_trace['validation_reason']=='duplicate_or_stale_response'


def test_precheck_and_final_validation_agree_and_are_rotation_invariant():
    env,p,engine=setup(expansion=True)
    context=engine.build_context(env,env.get_state_for_planning(),p)
    assert context['candidates']
    candidate=context['candidates'][0]
    assert validate_world_intent(p,env.get_state_for_planning(),candidate)['accepted']
    angle=.8;rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]);shift=np.array([17.,-31.])
    state=copy.deepcopy(env.get_state_for_planning())
    for s in [state['ego_state']]+[a['state'] for a in state['agents']]:
        s[:2]=(np.asarray(s[:2])@rot.T+shift).tolist();s[2:4]=(np.asarray(s[2:4])@rot.T).tolist();s[4]+=angle
    moved=dict(candidate,anchors=(np.asarray(candidate['anchors'])@rot.T+shift).tolist())
    a=validate_world_intent(p,state,moved);b=validate_world_intent(p,env.get_state_for_planning(),candidate)
    assert a['accepted']==b['accepted']
    np.testing.assert_allclose(a['metrics']['aligned_ego_distance_m'],b['metrics']['aligned_ego_distance_m'],atol=1e-9)


def test_background_overlap_never_published():
    env,p,engine=setup(expansion=True)
    env.data_dict['agent'][-1]=np.vstack([env.data_dict['agent'][-1],env.data_dict['agent'][-1]])
    env.agent_active=np.array([True,True])
    assert not engine.build_context(env,env.get_state_for_planning(),p)['candidates']


def test_ranked_fallback_uses_same_response(monkeypatch):
    env,p,engine=setup();install(p)
    import policies.skills_planner as module
    original=module.validate_world_intent;attempts={}
    def reject_first_at_compile(planner,state,candidate):
        cid=candidate['candidate_id'];attempts[cid]=attempts.get(cid,0)+1
        if cid=='c0' and attempts[cid]>1:return dict(accepted=False,reason='test_lifecycle_recheck',metrics={})
        return original(planner,state,candidate)
    monkeypatch.setattr(module,'validate_world_intent',reject_first_at_compile)
    plan=engine.plan(p,env,env.get_state_for_planning(),[])
    assert plan['attack'] and plan['candidate_id']=='c1' and len(p.prompts)==1
    assert p.last_trace['compile_attempts'][0]['reason']=='test_lifecycle_recheck'
