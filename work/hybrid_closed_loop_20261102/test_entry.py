"""共同参数、请求次数、异常恢复和公共配置覆盖，不调用真实SDK。"""
from pathlib import Path
from types import SimpleNamespace as NS
import json,sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from transport import install
from run_rag_worker import common_overrides


class Client:
    def __init__(self,fail=False):
        self.calls=[];self.fail=fail;self.chat=NS(completions=NS(create=self.create))
    def with_options(self,**kwargs):self.options=kwargs;return self
    def create(self,*args,**kwargs):
        self.calls.append(kwargs)
        if self.fail:raise RuntimeError('fixture_failure')
        return {'unchanged_response':True}


class Planner:
    def __init__(self,provider='dashscope',fail=False):self.provider=provider;self.client=Client(fail)


@pytest.mark.parametrize('fail',[False,True])
def test_request_parameters_and_single_call_are_preserved(tmp_path,fail):
    original=Planner.__init__
    with install(tmp_path,Planner):
        planner=Planner(fail=fail)
        request=lambda:planner.client.chat.completions.create(model='qwen3.8-2.4t-a95b',messages=[{'role':'user','content':'fixture'}],extra_body={'enable_thinking':False})
        if fail:
            with pytest.raises(RuntimeError):request()
        else:assert request()=={'unchanged_response':True}
    assert Planner.__init__ is original
    assert len(planner.client.calls)==1
    actual=planner.client.calls[0]
    assert actual['max_tokens']==8192 and actual['reasoning_effort']=='low'
    assert actual['extra_body']['enable_thinking'] is True
    assert planner.client.options==dict(timeout=180,max_retries=0)
    assert len((tmp_path/'common_transport.jsonl').read_text().splitlines())==1


def test_other_provider_is_unchanged(tmp_path):
    with install(tmp_path,Planner):
        planner=Planner('other')
        planner.client.chat.completions.create(model='untouched',messages=[])
    assert planner.client.calls==[dict(model='untouched',messages=[])]
    assert not (tmp_path/'common_transport.jsonl').exists()


def test_two_RAG_modes_share_hybrid_configuration():
    for enabled in [False,True]:
        values=common_overrides(lambda flag:['sim.rag.enabled='+str(flag).lower()],enabled)
        assert 'sim.llm.attack_mode=joint' in values
        assert 'sim.llm.skills.static_fallback_enabled=true' in values
        assert 'sim.llm.skills.source_fact_contract_enabled=true' in values
        assert 'sim.llm.obstacles.enabled=false' in values
