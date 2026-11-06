"""用户确认模型池与永久停用的串联回归；全部使用本地假客户端，不发网络请求。"""
from pathlib import Path
import pytest
from policies.codex_planner_bridge import PlannerServiceFailure
from policies.llm_adversarial_planner import LLMAdversarialPlanner


APPROVED_MODELS = (
    'qwen3.8-27b', 'kimi-k3', 'qwen3.8-max-0902',
    'deepseek-v4.1-flash-1017', 'deepseek-v4.1-flash', 'glm-5.3',
    'deepseek-v4-pro-0813', 'qwen3.8-2.4t-a95b', 'qwen3.8-max', 'qwen3.7-flash',
)


class FakeAPIError(Exception):
    def __init__(self, status, code):
        super().__init__(code)
        self.status_code=status
        self.code=code
        self.body={'code':code}


def isolated_pool(monkeypatch, tmp_path, names):
    """停用测试只写临时文件，绝不能污染用户的真实活动池。"""
    source=tmp_path/'planner.py'
    source.write_text('FREE_MODEL_POOL = (\n'+''.join(f'    "{n}",\n' for n in names)+')\n',encoding='utf8')
    reader=LLMAdversarialPlanner._exhausted_models_from_source
    writer=LLMAdversarialPlanner._comment_out_exhausted_model
    monkeypatch.setattr(LLMAdversarialPlanner,'FREE_MODEL_POOL',tuple(names))
    monkeypatch.setattr(LLMAdversarialPlanner,'_exhausted_models_from_source',staticmethod(lambda:reader(source)))
    monkeypatch.setattr(LLMAdversarialPlanner,'_comment_out_exhausted_model',
                        staticmethod(lambda model,reason='free_quota_exhausted':writer(model,source,reason=reason)))
    return source


def test_user_pool_has_only_confirmed_models_and_preserves_order():
    # 之后额度耗尽会自动注释，测试允许正确停用后的子序列，而非要求永远保留十个活动项。
    disabled=LLMAdversarialPlanner._exhausted_models_from_source()
    expected=tuple(name for name in APPROVED_MODELS if name not in disabled)
    assert LLMAdversarialPlanner.FREE_MODEL_POOL==expected
    assert len(expected)==len(set(expected))
    assert 'qwen3.8-flash' in disabled


def test_quota_then_404_persist_and_new_client_starts_at_survivor(monkeypatch,tmp_path):
    names=APPROVED_MODELS[:3]
    source=isolated_pool(monkeypatch,tmp_path,names)
    planner=LLMAdversarialPlanner(provider='bailian',client=object())
    seen=[]
    def request(model,*args):
        seen.append(model)
        if model==names[0]:raise FakeAPIError(403,'AllocationQuota.FreeTierOnly')
        if model==names[1]:raise FakeAPIError(404,'ModelNotFound')
        return '',{'attack':False}
    monkeypatch.setattr(planner,'_request_attack_plan_once',request)
    assert planner._request_attack_plan('local test')==('',{'attack':False})
    assert seen==list(names) and planner.model_names==(names[2],)
    content=source.read_text(encoding='utf8')
    assert f'# "{names[0]}",  # 百炼免费额度已耗尽，自动停用' in content
    assert f'# "{names[1]}",  # 百炼模型404，自动停用' in content
    restarted=LLMAdversarialPlanner(provider='dashscope',client=object())
    assert restarted.model_names==(names[2],)


def test_entire_pool_exhaustion_raises_service_failure_and_stays_disabled(monkeypatch,tmp_path):
    names=('test-free-a','test-free-b')
    isolated_pool(monkeypatch,tmp_path,names)
    planner=LLMAdversarialPlanner(provider='dashscope',client=object())
    seen=[]
    def fail(model,*args):
        seen.append(model);raise FakeAPIError(403,'AllocationQuota.FreeTierOnly')
    monkeypatch.setattr(planner,'_request_attack_plan_once',fail)
    with pytest.raises(FakeAPIError) as error:planner._request_attack_plan('local test')
    assert planner._is_service_failure(error.value)
    assert seen==list(names) and planner.model_names==()
    with pytest.raises(PlannerServiceFailure):
        LLMAdversarialPlanner(provider='dashscope',client=object())


def test_deepseek_named_models_stay_on_bailian_provider(monkeypatch):
    # 百炼上的DeepSeek模型名称不意味着切换至DeepSeek官方付费端点。
    monkeypatch.setattr(LLMAdversarialPlanner,'_exhausted_models_from_source',staticmethod(lambda:set()))
    for key in ('LLM_BASE_URL','DASHSCOPE_BASE_URL'):monkeypatch.delenv(key,raising=False)
    for name in APPROVED_MODELS:
        planner=LLMAdversarialPlanner(model_name=name,model_names=[name],provider='dashscope',client=object())
        assert planner.provider=='dashscope'
        assert planner.base_url=='https://dashscope.aliyuncs.com/compatible-mode/v1'
