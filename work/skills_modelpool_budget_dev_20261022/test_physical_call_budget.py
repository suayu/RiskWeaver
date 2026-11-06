"""确保模型回退、修复共享真实HTTP预算，且不误停用未调用模型。"""
import json
from types import SimpleNamespace as NS
import pytest
from physical_call_budget import (PhysicalCallBudget,TransportBudgetExhausted,
                                  bounded_model_request,install_physical_budget)
from policies.llm_adversarial_planner import LLMAdversarialPlanner
from experiments.skills.run_instrumented import install_api_audit


def response(model='unit-response',text='{"attack":false}'):
    return NS(model=model,choices=[NS(message=NS(content=text),finish_reason='stop')],usage=None)


def client(create):return NS(api_key='TEST_CREDENTIAL_PLACEHOLDER_0001',chat=NS(completions=NS(create=create)))


def test_failed_transport_still_consumes_budget_and_reset_is_per_scene():
    calls=[]
    def fail(**request):calls.append(request['model']);raise TimeoutError('private exception body')
    planner=NS(client=client(fail));budget=PhysicalCallBudget(1);budget.attach(planner);budget.reset(53)
    with pytest.raises(TimeoutError):planner.client.chat.completions.create(model='a')
    with pytest.raises(TransportBudgetExhausted):planner.client.chat.completions.create(model='b')
    assert calls==['a'] and budget.calls==1
    budget.reset(56)
    with pytest.raises(TimeoutError):planner.client.chat.completions.create(model='c')
    assert calls==['a','c'] and budget.calls==1 and budget.scene_index==56


def test_repair_calls_share_same_sdk_limit():
    seen=[];planner=NS(client=client(lambda **k:seen.append(k['model']) or response()))
    budget=PhysicalCallBudget(2);budget.attach(planner)
    planner.client.chat.completions.create(model='first')
    planner.client.chat.completions.create(model='schema-repair')
    with pytest.raises(TransportBudgetExhausted):planner.client.chat.completions.create(model='extra')
    assert seen==['first','schema-repair'] and budget.calls==2


def test_refused_fallback_does_not_disable_unsent_model(monkeypatch):
    class Error(Exception):status_code=403;code='AllocationQuota.FreeTierOnly'
    marks=[];actual=[]
    def fail(**request):actual.append(request['model']);raise Error('free quota exhausted')
    planner=LLMAdversarialPlanner(provider='dashscope',model_name='empty-a',
        model_names=['empty-a','empty-b','not-sent'],client=client(fail))
    monkeypatch.setattr(planner,'_comment_out_exhausted_model',lambda name:marks.append(name) or True)
    budget=PhysicalCallBudget(2);budget.attach(planner)
    with pytest.raises(Error):bounded_model_request(planner,LLMAdversarialPlanner._request_attack_plan,'prompt')
    assert actual==marks==['empty-a','empty-b']
    assert planner.last_attempted_model_names==actual
    assert planner.model_names==('not-sent',) and budget.calls==2


def test_invalid_json_is_budget_exhaustion_not_service_failure():
    planner=LLMAdversarialPlanner(provider='dashscope',model_name='bad-a',
        model_names=['bad-a','bad-b'],client=client(lambda **k:response(text='not JSON')))
    budget=PhysicalCallBudget(1);budget.attach(planner)
    with pytest.raises(TransportBudgetExhausted) as error:
        bounded_model_request(planner,LLMAdversarialPlanner._request_attack_plan,'prompt')
    assert not planner._is_service_failure(error.value)
    assert planner.last_attempted_model_names==['bad-a'] and budget.calls==1


def test_sdk_budget_sits_outside_audit_so_blocked_send_is_not_http(tmp_path):
    audit_restore=install_api_audit(tmp_path/'run')
    budget_restore=install_physical_budget(tmp_path/'run',limit=1)
    actual=[]
    try:
        planner=LLMAdversarialPlanner(provider='dashscope',model_name='healthy',model_names=['healthy'],
            client=client(lambda **k:actual.append(k['model']) or response()))
        planner._audit_scene_index=53;planner._physical_call_budget.reset(53)
        assert planner._request_attack_plan('first')[1]=={'attack':False}
        with pytest.raises(TransportBudgetExhausted):planner._request_attack_plan('second')
    finally:
        budget_restore();audit_restore()
    http=(tmp_path/'run_api_calls.jsonl').read_text()
    blocked=(tmp_path/'run_budget_events.jsonl').read_text()
    assert len(http.splitlines())==1 and len(blocked.splitlines())==1
    assert actual==['healthy'] and json.loads(blocked)['transport_calls']==1
    assert 'TEST_CREDENTIAL_PLACEHOLDER_0001' not in http+blocked and 'api_key' not in http+blocked


@pytest.mark.parametrize('invalid',[0,31,True,1.5])
def test_budget_boundary_validation(invalid):
    with pytest.raises(ValueError):PhysicalCallBudget(invalid)
