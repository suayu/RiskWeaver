"""实际Temporal/压缩/契约/候选校验链的SDK替身回归；没有真实模型请求。"""
from pathlib import Path
from types import SimpleNamespace as NS
import importlib.util,json,sys
import pytest
from policies.rag_runtime import initialize_rag_experiment,AttackRAGRuntime,current_facts

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('candidate_fixture',ROOT/'tests/test_candidate_local_grounding.py')
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)


@pytest.mark.parametrize('enabled',[False,True])
def test_real_temporal_transport_rules_retrieval_and_validation(enabled,tmp_path):
    engine,planner,env,state,outputs=fixture.make_case('candidate_local')
    settings=NS(policy='idm',llm=NS(skills=NS(enabled=True)),rag=NS(enabled=enabled,record_events=True,inherit=False,inherit_path=None))
    initialize_rag_experiment(settings,tmp_path/'experiment');r=AttackRAGRuntime.attach_initialized(settings)
    r.begin_attempt(env,tmp_path/'attempt');r.begin_event(env,state,['probe'],.5)
    facts=current_facts(state,engine.build_context(env,state,planner,None),r.policy_id,r.policy_key,r.protocol_key,0,.5)
    r.store.append(dict(case_id='prior',record_type='attack_event',status='closed',policy_id=r.policy_id,policy_key=r.policy_key,
        protocol_key=r.protocol_key,query_facts=facts,scene_conditions=facts['scene_conditions'],D=None,measurement_valid=False,
        strategy='lane_change',is_failure=True,generation_failure_reason='no_execution',benefit_eligible=False))
    previous=planner._request_attack_plan;captured=[]
    def outgoing(prompt,scene_image=None):
        payload=json.loads(prompt.rsplit('\n',1)[1]);captured.append(payload)
        return previous(prompt,scene_image=scene_image)
    planner._request_attack_plan=outgoing
    r.bind_planner(engine);r.bind_planner(engine)  # 重复调用前检查不得叠加包装或物理请求。
    plan=engine.plan(planner,env,state,['probe'])
    assert plan and plan['attack'] and engine.calls==1 and len(outputs)==len(captured)==1
    docs=captured[0]['knowledge']['documents']
    assert {d['path'] for d in docs}>={'rw-current-fact-reasoning/SKILL.md','rw-case-evidence/SKILL.md'}
    evidence=captured[0]['context'].get('retrieved_attack_evidence')
    assert (evidence is not None)==enabled
    if enabled:assert evidence['documents'][0]['case_id']=='prior'
    assert planner.last_trace['candidate_grounding_validation']['candidates'][-1]['accepted'] is False
    assert captured[0]['catalog_hash']==outputs[0]['catalog_hash']


def test_current_registry_has_rules_but_old_frozen_registry_still_replays():
    from policies.skill_registry import SkillRegistry
    docs=SkillRegistry(level='full').load()['documents']
    assert {'rw-current-fact-reasoning/SKILL.md','rw-case-evidence/SKILL.md'}<={d['path'] for d in docs}
