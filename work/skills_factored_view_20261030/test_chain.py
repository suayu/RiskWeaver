"""真实提示装饰/预算/SDK边界和原编译链，离线替身正确解码不等于真实模型会读。"""
from pathlib import Path
from types import SimpleNamespace as NS
import importlib.util,json,sys
import pytest

ROOT=Path('/home2/zhaoyx/scenario-dreamer');HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import frozen_compare as trial
from factored_view import MARKER,expand_payload
spec=importlib.util.spec_from_file_location('factored_chain_support',ROOT/'work/skills_highlevel_comparison_20261030/test_highlevel_comparison.py')
support=importlib.util.module_from_spec(spec);spec.loader.exec_module(support)


@pytest.mark.parametrize('mode',trial.ARMS)
def test_true_SDK_boundary_sees_requested_view_and_compiles_original_candidate(tmp_path,mode):
    sample,cfg=support.fixture();sent=[]
    from policies.llm_adversarial_planner import LLMAdversarialPlanner
    sys.path.insert(0,str(ROOT/'work/skills_scene_facts_prototype_20261028'))
    from test_fact_runtime import answer
    def create(*args,**kwargs):
        payload=json.loads(kwargs['messages'][-1]['content'].rsplit('\n',1)[1]);sent.append(payload)
        source=expand_payload(payload) if MARKER in payload['context'] else payload
        response=answer(source)
        return NS(model=kwargs['model'],choices=[NS(message=NS(content=json.dumps(response),reasoning_content=None),finish_reason='stop')],usage=NS(model_dump=lambda:dict(total_tokens=0)))
    factory=lambda **kw:LLMAdversarialPlanner(client=NS(chat=NS(completions=NS(create=create))),**kw)
    before=trial.base.engine_for
    result=trial.run_trial(sample,mode,['fixture-bailian-model'],cfg,tmp_path/mode,planner_factory=factory)
    assert result['accepted_plan'] and result['budget']['transport_calls']==1 and not result['planner_service_failure']
    assert (MARKER in sent[0]['context'])==(mode=='factored_wire_view') and trial.base.engine_for is before
    assert result['planner_trace']['candidate_grounding_validation']['policy']=='candidate_local_strict'
    assert result['new_execution_frames']==0 and result['D'] is None


def test_unknown_view_cannot_silently_change_the_old_path(tmp_path):
    with pytest.raises(ValueError,match='unknown_wire'):trial.run_trial({},'guess',[],None,tmp_path/'unused')
