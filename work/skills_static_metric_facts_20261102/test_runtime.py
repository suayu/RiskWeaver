"""复用公共静态工厂与请求链；仅SDK替身，无实际API或env.step。"""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path('/home2/zhaoyx/scenario-dreamer')
sys.path[:0] = [str(ROOT), str(ROOT/'tests'), str(Path(__file__).resolve().parent)]
spec = importlib.util.spec_from_file_location('static_metric_public_fixture',
    ROOT/'work/hybrid_public_integration_20261101/test_public_hybrid.py')
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
from prototype import make_static_planner
from policies.hybrid_skills_planner import StaticCombinationPlanner
from policies.intent_validation import canonical_hash


def cfg():
    return NS(ego_profile=NS(), llm=NS(skills=NS(enabled=True, max_repairs=1,
        source_fact_contract_enabled=True)))


def test_off_returns_original_class_without_extra_context():
    env, planner, _, _ = fixture.setup()
    subject = make_static_planner(cfg(), enabled=False)
    assert type(subject) is StaticCombinationPlanner
    assert 'static_metric_facts' not in subject.build_context(env, env.get_state_for_planning(), planner)


def test_actual_public_request_contains_facts_and_fixed_skill_before_hash():
    env, planner, engine, calls = fixture.setup()
    engine.static = make_static_planner(cfg(), enabled=True)
    plan = engine.plan(planner, env, env.get_state_for_planning(), ['challenge'])
    assert plan['static_attack'] and engine.calls == 1 and len(calls) == 1
    payload = calls[0]
    assert payload['catalog_hash'] == canonical_hash(payload['context'])
    assert 'static_metric_facts' in payload['context']
    assert any(d['path'] == 'prototype/rw-static-metric-facts/SKILL.md'
        for d in payload['knowledge']['documents'])
    assert 'source_facts_v1' in payload['context']
    json.dumps(payload, allow_nan=False)
    # 校验使用真正公共类；不能因历史SDK替身污染sys.modules而误测镜像。
    import policies.hybrid_skills_planner as public
    assert Path(public.__file__).resolve() == ROOT/'policies/hybrid_skills_planner.py'


def test_same_selected_id_keeps_original_compiled_static_plan_and_budget():
    outcomes = []
    for enabled in [False, True]:
        env, planner, engine, calls = fixture.setup()
        engine.static = make_static_planner(cfg(), enabled=enabled)
        plan = engine.plan(planner, env, env.get_state_for_planning(), ['challenge'])
        outcomes.append((plan['obstacle_plan'], plan['attack_kind'], engine.calls))
        assert len(calls) == 1
    assert outcomes[0] == outcomes[1]
