"""参数、回退及物理预算集成回归；所有SDK均替身，零真实模型请求。"""
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import sys

import pytest
from provider_control import install_control, request_parameters
from policies.llm_adversarial_planner import LLMAdversarialPlanner as Planner

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'work/skills_modelpool_budget_dev_20261022'))
from physical_call_budget import PhysicalCallBudget, TransportBudgetExhausted, bounded_model_request


class APIError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status_code = status


def planner(models, create):
    p = object.__new__(Planner)
    p.provider = 'dashscope'; p.model_names = tuple(models)
    p._preferred_model_index = 0; p.model_name = models[0]
    p.use_multimodal = False; p.last_trace = {}
    p.client = NS(chat=NS(completions=NS(create=create)))
    p._exhausted_models_from_source = lambda:set()
    return p


def response():
    return NS(choices=[NS(message=NS(content='{"attack":false}', reasoning_content=''))])


@pytest.mark.parametrize('model', ['glm-5.3', 'qwen3.8-2.4t-a95b'])
def test_exact_observed_models_one_real_adapter_call_and_restore(model):
    calls=[]
    def create(**kw):calls.append(kw);return response()
    p=planner([model], create); prior=Planner._request_attack_plan_once
    restore=install_control('service_only')
    try:
        assert p._request_attack_plan_once(model, 'fixture')[1]=={'attack':False}
        assert len(calls)==1 and calls[0]['extra_body']['enable_thinking'] is True
        assert calls[0]['response_format']=={'type':'json_object'}
        assert p.client.chat.completions.create is create
    finally:restore()
    assert Planner._request_attack_plan_once is prior


@pytest.mark.parametrize('provider,model', [('dashscope','kimi-k3'),('dashscope','qwen3.8-max'),('deepseek','glm-5.3'),('qwen','qwen3.8-2.4t-a95b')])
def test_unobserved_model_and_other_provider_unchanged(provider,model):
    values={'model':model,'extra_body':{'enable_thinking':False,'other':1}}
    assert request_parameters(provider,values)==values
    assert values['extra_body']['enable_thinking'] is False


def test_exception_restores_sdk_without_hidden_retry():
    calls=[]; error=APIError(400,'invalid_parameter')
    def create(**kw):calls.append(kw);raise error
    p=planner(['glm-5.3'],create);restore=install_control()
    try:
        with pytest.raises(APIError) as caught:p._request_attack_plan_once('glm-5.3','fixture')
        assert caught.value is error and len(calls)==1 and p.client.chat.completions.create is create
    finally:restore()


def test_service_only_keeps_semantically_rejected_model_but_default_still_rotates():
    p=planner(['kimi-k3','glm-5.3'],lambda **kw:response())
    restore=install_control('service_only')
    try:
        p._advance_after_invalid_plan()
        assert p.model_name=='kimi-k3' and p._preferred_model_index==0
        assert p.last_trace['semantic_model_transition']['service_fallback_unchanged']
    finally:restore()
    restore=install_control()
    try:p._advance_after_invalid_plan();assert p.model_name=='glm-5.3'
    finally:restore()


@pytest.mark.parametrize('status,message,disabled', [(404,'model not found',True),(403,'Free quota expired or exhausted',True),(429,'requests per minute exceeded',False),(400,'enable_thinking parameter',False)])
def test_genuine_service_fallback_and_disable_classification_unchanged(status,message,disabled):
    calls=[];comments=[]
    def create(**kw):
        calls.append(kw['model'])
        if kw['model']=='kimi-k3':raise APIError(status,message)
        return response()
    p=planner(['kimi-k3','glm-5.3'],create)
    p._comment_out_exhausted_model=lambda model,**kwargs:comments.append((model,kwargs)) or True
    restore=install_control('service_only')
    try:
        assert p._request_attack_plan('fixture')[1]=={'attack':False}
        assert calls==['kimi-k3','glm-5.3']
        assert bool(comments)==disabled
    finally:restore()


def test_budget_four_includes_failed_fallback_and_schema_repair_and_resets_scene():
    calls=[]
    def create(**kw):
        calls.append(kw['model'])
        if len(calls) in [1,3]:raise APIError(429,'rate limit')
        return response()
    p=planner(['kimi-k3','glm-5.3'],create);budget=PhysicalCallBudget(4);budget.attach(p)
    restore=install_control('service_only')
    try:
        for _ in range(2):bounded_model_request(p,Planner._request_attack_plan,'fixture or repair')
        assert len(calls)==4 and budget.calls==4
        with pytest.raises(TransportBudgetExhausted):bounded_model_request(p,Planner._request_attack_plan,'over budget')
        assert len(calls)==4 and p.last_attempted_model_names==[]
        budget.reset(56)
        assert bounded_model_request(p,Planner._request_attack_plan,'next scene')[1]=={'attack':False}
        assert budget.calls==1 and len(calls)==5
    finally:restore()


def test_all_failures_remain_failure_not_normal_no_attack():
    def create(**kw):raise APIError(429,'rate limit')
    p=planner(['kimi-k3','glm-5.3'],create);restore=install_control('service_only')
    try:
        with pytest.raises(APIError):p._request_attack_plan('fixture')
        assert len(p.last_attempted_model_names)==2
    finally:restore()
