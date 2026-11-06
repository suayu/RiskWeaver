"""实际skills.plan调用层的重复响应拒绝：仅传输替身，无SDK或模型请求。"""
from types import SimpleNamespace as NS
from test_skills_planner import setup,install


def test_duplicate_response_identity_rejected_after_mocked_transport(monkeypatch):
    import policies.skills_planner as module
    env,planner,skills=setup(budget=4,repairs=0)
    install(planner)
    # 强制碰撞的请求ID用于覆盖seen保护；不影响其他进程或原始源码。
    monkeypatch.setattr(module.uuid,'uuid4',lambda:NS(hex='repeat-for-boundary-test'))
    first=skills.plan(planner,env,env.get_state_for_planning(),[])
    assert first['attack'] and len(planner.prompts)==1
    second=skills.plan(planner,env,env.get_state_for_planning(),[])
    assert second is None and len(planner.prompts)==2
    assert planner.last_trace['validation_reason']=='duplicate_or_stale_response'
    assert not planner.last_request_failed
