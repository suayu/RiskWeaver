"""真实SDK与技能/审计/预算组合验证；MockTransport只在内存响应，禁止外网。"""
import json
from types import SimpleNamespace as NS
import httpx
from openai import OpenAI
import pytest
from policies.llm_adversarial_planner import LLMAdversarialPlanner
from experiments.skills.run_instrumented import install_api_audit
from physical_call_budget import install_physical_budget
from compact_planner import CompactSkillsPlanner
from test_ego_profile import environment
from experiments.comparison import run_e1_group as runner


def sdk_response(request,reply):
    return httpx.Response(200,request=request,json=dict(id='unit-response',object='chat.completion',created=0,
        model='contract-unit-b',choices=[dict(index=0,message=dict(role='assistant',content=json.dumps(reply)),finish_reason='stop')],
        usage=dict(prompt_tokens=10,completion_tokens=5,total_tokens=15)))


def rank_reply(request):
    payload=json.loads(json.loads(request.content)['messages'][0]['content'].split('\n',1)[1])
    ids=[c['candidate_id'] for c in payload['context']['candidates']]
    reply={k:payload[k] for k in ['request_id','state_hash','catalog_hash']}
    reply.update(ranking=ids[:6],scene_analysis='Rank supplied feasible candidates',
                 candidate_reasons={i:'supplied risk and safety evidence' for i in ids[:6]},no_attack_reason='')
    return reply


def engine():
    return CompactSkillsPlanner(NS(ego_profile=NS(),llm=NS(skills=NS(enabled=True,
        candidate_expansion_enabled=True,max_repairs=1))),4)


def install(tmp_path,limit=4):
    # 顺序与排队的独立运行入口完全相同，预算层位于审计层之外。
    audit=install_api_audit(tmp_path/'arm');budget=install_physical_budget(tmp_path/'arm',limit)
    return lambda:(budget(),audit())


def test_real_sdk_fallback_then_schema_repair_counts_three_transports(tmp_path,monkeypatch):
    received=[];marks=[]
    def transport(request):
        received.append(json.loads(request.content))
        if len(received)==1:return httpx.Response(404,request=request,json={'error':{'message':'model not found','type':'model_not_found','code':'model_not_found'}})
        reply=rank_reply(request)
        if len(received)==2:reply['request_id']='wrong-request'
        return sdk_response(request,reply)
    monkeypatch.setattr(LLMAdversarialPlanner,'_comment_out_exhausted_model',lambda self,name,reason='free_quota_exhausted':marks.append(name) or True)
    restore=install(tmp_path)
    try:
        with OpenAI(api_key='unit-secret-not-real',base_url='https://unit.invalid/v1',max_retries=0,
                    http_client=httpx.Client(transport=httpx.MockTransport(transport))) as sdk:
            p=LLMAdversarialPlanner(provider='dashscope',model_name='contract-unit-a',
                model_names=['contract-unit-a','contract-unit-b'],client=sdk)
            p._audit_scene_index=53;p._physical_call_budget.reset(53)
            env=environment();e=engine();plan=e.plan(p,env,env.get_state_for_planning(),[])
            assert plan and plan['attack']
            assert e.calls==2 and p._physical_call_budget.calls==3
            assert len(received)==3 and marks==['contract-unit-a']
            assert p.last_trace['plan_validation']=='accepted'
    finally:restore()
    records=[json.loads(x) for x in (tmp_path/'arm_api_calls.jsonl').read_text().splitlines()]
    assert len(records)==3 and records[0]['status_code']==404
    assert all(x['scenario_index']==53 for x in records)
    assert 'unit-secret-not-real' not in (tmp_path/'arm_api_calls.jsonl').read_text()


def test_episode_hooks_reset_actual_sdk_budget_and_trace_scene_id(tmp_path,monkeypatch):
    received=[];e=engine()
    def transport(request):
        received.append(request);return sdk_response(request,rank_reply(request))
    def episode(env,policy,index,global_index,*args,**kwargs):
        e.reset_episode()
        for step in [0,11]:
            env.current_step=step
            plan=e.plan(env.llm_planner,env,env.get_state_for_planning(),[])
            assert plan and plan['attack']
        return env.llm_planner._physical_call_budget.snapshot()
    monkeypatch.setattr(runner,'run_episode',episode)
    restore=install(tmp_path,2)
    try:
        with OpenAI(api_key='unit-secret-not-real',base_url='https://unit.invalid/v1',max_retries=0,
                    http_client=httpx.Client(transport=httpx.MockTransport(transport))) as sdk:
            p=LLMAdversarialPlanner(provider='dashscope',model_name='contract-unit-b',model_names=['contract-unit-b'],client=sdk)
            env=environment();env.llm_planner=p
            first=runner.run_episode(env,None,0,53)
            second=runner.run_episode(env,None,1,56)
            assert first['transport_calls']==second['transport_calls']==2
            assert first['scenario_index']==53 and second['scenario_index']==56
            assert len(received)==4
    finally:restore()
    rows=[json.loads(x) for x in (tmp_path/'arm_api_calls.jsonl').read_text().splitlines()]
    assert [x['scenario_index'] for x in rows]==[53,53,56,56]
    assert [x['source_step'] for x in rows]==[0,11,0,11]
    assert not (tmp_path/'arm_budget_events.jsonl').exists()
