"""验证开发入口的作用域、旧路径和设备隔离；不进行真实模型调用。"""
from types import SimpleNamespace as NS
import pytest
from run_closed_loop import install_variant, validate_model_environment


def options(**kwargs):
    return NS(**dict(dict(compact_view='off', event_budget='off',
                          reachability_preflight='off', max_screening_queries=40, device='cuda'), **kwargs))


def test_default_configuration_preserves_model_solver_and_device():
    source = ['sim.llm.max_planning_calls=4', 'sim.traffic_model.guidance.params.inner_lr=0.2']
    original = object()
    runner = NS(AdversarialScenarioGenerator=original, group_overrides=lambda *a, **k: list(source))
    old_overrides = runner.group_overrides
    restore = install_variant(options(), runner)
    try:
        result = runner.group_overrides()
        assert result[:len(source)] == source
        assert result[len(source):] == ['+sim.llm.skills.event_budget_enabled=false',
                                      '+sim.llm.skills.max_screening_queries=40',
                                      '+sim.llm.skills.reachability_preflight_enabled=false']
    finally:
        restore()
    assert runner.AdversarialScenarioGenerator is original and runner.group_overrides is old_overrides


def test_cpu_compact_event_overrides_are_explicit_and_scoped():
    runner = NS(AdversarialScenarioGenerator=object(), group_overrides=lambda: ['original=true'])
    restore = install_variant(options(device='cpu', compact_view='on', event_budget='on'), runner)
    try:
        values = runner.group_overrides()
        assert 'sim.traffic_model.device=cpu' in values
        assert '+sim.llm.skills.event_budget_enabled=true' in values
        assert any(v.startswith('sim.llm.skills.root=') and v.endswith('/skills') for v in values)
        assert not any('inner_lr' in v or 'duration' in v or 'weight' in v for v in values)
    finally:
        restore()


def test_launcher_default_model_must_not_silently_override_cli(monkeypatch):
    monkeypatch.setenv('LLM_MODEL_NAME', 'qwen-plus')
    monkeypatch.delenv('LLM_MODEL_NAMES', raising=False)
    with pytest.raises(ValueError, match='environment model differs'):
        validate_model_environment(['--model', 'qwen3.8-flash'])


def test_environment_pool_must_match_fixed_protocol(monkeypatch):
    monkeypatch.setenv('LLM_MODEL_NAME', 'qwen3.8-flash')
    monkeypatch.setenv('LLM_MODEL_NAMES', 'qwen3.8-flash,other')
    with pytest.raises(ValueError, match='model pool differs'):
        validate_model_environment(['--model', 'qwen3.8-flash'])
    monkeypatch.setenv('LLM_MODEL_NAMES', 'qwen3.8-flash')
    validate_model_environment(['--model', 'qwen3.8-flash'])
