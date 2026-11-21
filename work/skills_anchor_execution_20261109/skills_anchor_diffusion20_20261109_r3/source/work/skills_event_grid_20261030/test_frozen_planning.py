"""SDK替身走原请求/记录/预算/编译链；验证初态分母及错误不伪装成功。"""
from pathlib import Path
from types import SimpleNamespace as NS
import json,sys
import pytest
from frozen_planning import ROOT,run_trial
from prepare_planning import samples
from policies.llm_adversarial_planner import LLMAdversarialPlanner


def config():
    from omegaconf import OmegaConf
    return OmegaConf.load(ROOT/'experiments/skills_temporal_development10_paired_20261029_r1/skills_full_shared_recovery/resolved_config.yaml')


def factory(bad=False):
    sys.path.insert(0,str(ROOT/'work/skills_scene_facts_prototype_20261028'))
    from test_fact_runtime import answer
    def create(*args,**kwargs):
        payload=json.loads(kwargs['messages'][-1]['content'].rsplit('\n',1)[1]);result=answer(payload)
        if bad:result['request_id']='wrong-snapshot'
        return NS(model=kwargs['model'],choices=[NS(message=NS(content=json.dumps(result),reasoning_content=None),finish_reason='stop')],
            usage=NS(model_dump=lambda:dict(prompt_tokens=0,completion_tokens=0,total_tokens=0)))
    def build(**kwargs):return LLMAdversarialPlanner(client=NS(chat=NS(completions=NS(create=create))),**kwargs)
    return build


def test_twenty_initial_state_arms_have_exact_same_physics_and_no_historical_responses():
    data=samples();assert len(data)==10 and len({s['scene'] for s in data})==10
    assert all(s['state']['current_step']==0 and 'response' not in s for s in data)
    assert all(set(s['contexts'])=={'temporal_original','full_grid'} for s in data)


@pytest.mark.parametrize('mode',['temporal_original','full_grid'])
def test_real_prompt_request_budget_and_compile_with_sdk_substitute(tmp_path,mode):
    sample=next(s for s in samples() if s['scene']==74)
    r=run_trial(sample,mode,['deepseek-v4.1-flash-1017'],config(),tmp_path/mode,planner_factory=factory())
    assert r['accepted_plan'] and not r['planner_service_failure'] and r['budget']['transport_calls']==1
    assert r['new_execution_frames']==0 and r['D'] is None and r['semantic_review']=='not_reviewed'
    raw=[json.loads(x) for x in (tmp_path/mode/'api_calls.jsonl').read_text().splitlines()]
    assert len(raw)==1 and raw[0]['content'] and raw[0]['messages']


def test_invalid_identity_uses_bounded_repair_and_remains_rejected(tmp_path):
    sample=next(s for s in samples() if s['scene']==74)
    r=run_trial(sample,'full_grid',['deepseek-v4.1-flash-1017'],config(),tmp_path/'bad',planner_factory=factory(bad=True))
    assert not r['accepted_plan'] and not r['planner_service_failure']
    assert r['budget']['transport_calls']==2 and r['logical_requests']==2
    assert r['planner_trace']['schema_rejections']


def test_empty_catalog_remains_in_denominator_without_sending_request(tmp_path):
    sample=next(s for s in samples() if s['scene']==80)
    r=run_trial(sample,'full_grid',['deepseek-v4.1-flash-1017'],config(),tmp_path/'empty',planner_factory=factory())
    assert r['attempted'] and not r['accepted_plan'] and r['budget']['transport_calls']==0
    assert r['plan']['reason']=='no_feasible_candidates' and not (tmp_path/'empty/api_calls.jsonl').exists()
