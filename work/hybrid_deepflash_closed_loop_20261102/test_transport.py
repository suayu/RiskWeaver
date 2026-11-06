"""只用SDK替身核验单次发送、失败计账和提供方参数隔离；没有真实模型请求。"""
from types import SimpleNamespace as NS
import json
import pytest
from transport import install, MODEL


def fixture(provider='dashscope', failure=False):
    calls, options = [], []
    def create(**kwargs):
        calls.append(kwargs)
        if failure:
            raise RuntimeError('fixture')
        return 'response'
    client = NS(chat=NS(completions=NS(create=create)))
    def with_options(**kwargs):
        options.append(kwargs)
        return client
    client.with_options = with_options
    class Planner:
        def __init__(self):
            self.provider, self.client = provider, client
    return Planner, calls, options


def test_preserves_provider_thinking_and_JSON_parameters(tmp_path):
    planner, calls, options = fixture()
    before = planner.__init__
    with install(tmp_path, planner):
        instance = planner()
        result = instance.client.chat.completions.create(model=MODEL, messages=[{'role':'user','content':'fixture'}],
            extra_body={'enable_thinking': False}, response_format={'type':'json_object'})
    assert result == 'response' and len(calls) == 1
    assert options == [{'timeout':180,'max_retries':0}]
    assert calls[0]['extra_body'] == {'enable_thinking':False}
    assert calls[0]['response_format'] == {'type':'json_object'}
    assert calls[0]['max_tokens'] == 8192 and 'reasoning_effort' not in calls[0]
    assert planner.__init__ is before


def test_failure_is_one_call_and_receipt_is_preserved(tmp_path):
    planner, calls, _ = fixture(failure=True)
    before = planner.__init__
    with pytest.raises(RuntimeError):
        with install(tmp_path, planner):
            planner().client.chat.completions.create(model=MODEL, messages=[])
    assert len(calls) == 1 and planner.__init__ is before
    rows = [json.loads(x) for x in (tmp_path/'common_transport.jsonl').read_text().splitlines()]
    assert len(rows) == 1 and rows[0]['error_type'] == 'RuntimeError'


def test_other_provider_unmodified(tmp_path):
    planner, calls, options = fixture(provider='other')
    with install(tmp_path, planner):
        planner().client.chat.completions.create(model='other', messages=[])
    assert not options and 'max_tokens' not in calls[0]
    assert not (tmp_path/'common_transport.jsonl').exists()


def test_unfrozen_model_cannot_be_sent(tmp_path):
    planner, calls, _ = fixture()
    with install(tmp_path, planner):
        with pytest.raises(ValueError):
            planner().client.chat.completions.create(model='depleted-model', messages=[])
    assert not calls
