"""模型前后采用同一无凭据记录层；仅传输替身，不请求API。"""
import json
from types import SimpleNamespace as NS
from experiments.skills.run_instrumented import install_api_audit
from policies.llm_adversarial_planner import LLMAdversarialPlanner


def test_usage_content_and_scene_are_recorded_without_client_secret(tmp_path):
    response=NS(model='audit-unit-model',choices=[NS(message=NS(content='{"attack":false}'),finish_reason='stop')],
                usage=NS(model_dump=lambda:dict(prompt_tokens=8,completion_tokens=2,total_tokens=10)))
    client=NS(api_key='TEST_CREDENTIAL_PLACEHOLDER_0001',chat=NS(completions=NS(create=lambda **kwargs:response)))
    original=LLMAdversarialPlanner.__init__
    restore=install_api_audit(tmp_path/'run')
    try:
        planner=LLMAdversarialPlanner(model_name='audit-unit-model',model_names=['audit-unit-model'],provider='dashscope',client=client)
        planner._audit_scene_index=182;planner._audit_step=11
        received=planner.client.chat.completions.create(model='audit-unit-model',messages=[dict(role='user',content='rank catalog')])
        assert received is response
    finally:restore()
    assert LLMAdversarialPlanner.__init__ is original
    text=(tmp_path/'run_api_calls.jsonl').read_text()
    assert 'TEST_CREDENTIAL_PLACEHOLDER_0001' not in text and 'api_key' not in text
    record=json.loads(text)
    assert record['usage']['total_tokens']==10 and record['scenario_index']==182 and record['source_step']==11
    assert record['messages']==[dict(role='user',content='rank catalog')]


def test_transport_error_record_does_not_include_exception_body(tmp_path):
    def failure(**kwargs):raise TimeoutError('sensitive-error-body')
    client=NS(chat=NS(completions=NS(create=failure)))
    restore=install_api_audit(tmp_path/'run')
    try:
        planner=LLMAdversarialPlanner(model_name='audit-unit-model',model_names=['audit-unit-model'],provider='dashscope',client=client)
        try:planner.client.chat.completions.create(model='audit-unit-model',messages=[])
        except TimeoutError:pass
        else:raise AssertionError('transport failure was swallowed')
    finally:restore()
    text=(tmp_path/'run_api_calls.jsonl').read_text()
    assert 'sensitive-error-body' not in text
    assert json.loads(text)['error_type']=='TimeoutError'
