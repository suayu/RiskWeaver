"""公共混合引擎接线与最新状态复核；所有模型回复均为显式替身。"""
from pathlib import Path
from types import SimpleNamespace as NS
import importlib.util
import json
import sys
import numpy as np
import pytest

ROOT = Path('/home2/zhaoyx/scenario-dreamer')
sys.path.insert(0, str(ROOT))
from work.skills_static_physical15_20261102.planner import make_static, RULE_PATH
from work.skills_static_physical15_20261102.runtime import install
from policies.hybrid_skills_planner import StaticCombinationPlanner, DynamicFirstStaticPlanner
from policies.intent_validation import canonical_hash


def setup(active=False, budget=4):
    spec = importlib.util.spec_from_file_location('physical15_public_fixture',
        ROOT/'work/hybrid_public_integration_20261101/test_public_hybrid.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.setup(active, budget)


def curved(env):
    env.scenario_dict['route'] = np.array([[0., 0.], [10., 0.], [10., 100.]])
    env.scenario_dict['lanes'] = np.array([env.scenario_dict['route']])


def test_disabled_factory_and_scope_preserve_exact_old_objects():
    cfg = NS(ego_profile=NS(), llm=NS(skills=NS(enabled=True, max_repairs=1)))
    assert type(make_static(cfg)) is StaticCombinationPlanner
    original = DynamicFirstStaticPlanner.__init__
    with install() as receipt:
        assert DynamicFirstStaticPlanner.__init__ is original
        assert not receipt['enabled']
        assert receipt['constructed'] == []


def test_static_request_rule_and_final_validation_are_both_physical15():
    with install(True) as receipt:
        env, low, engine, calls = setup()
        curved(env)
        original = low._validate_obstacle_plan
        plan = engine.plan(low, env, env.get_state_for_planning(), ['动态优先，静态兜底'])
        assert plan['static_attack'] and len(calls) == engine.calls == 1
        context = calls[0]['context']
        policy = context['static_clearance_policy']
        assert policy['fallback_initial_ego_OBB_margin_m'] == 15.
        assert policy['background_margin_m'] == .5
        assert calls[0]['catalog_hash'] == canonical_hash(context)
        assert any(d['path'] == RULE_PATH for d in calls[0]['knowledge']['documents'])
        trace = low.last_trace['static_physical15']
        assert trace['final_checks'][-1]['accepted']
        assert trace['final_checks'][-1]['new_route_check_events'] > 0
        assert all(e['initial_margin_m'] == 15 for e in trace['geometry']['events'])
        assert all(min(e['initial_ego_OBB_clearances_m']) >= 15
                   for e in trace['geometry']['events'] if e['accepted'])
        assert low._validate_obstacle_plan == original
        assert receipt['constructed'][0]['call_budget'] == 4
        json.dumps(low.last_trace, allow_nan=False)


def test_latest_state_rejects_previously_published_selection():
    with install(True):
        env, low, engine, calls = setup()
        curved(env)
        original = low._validate_obstacle_plan
        select = engine.static.select_obstacle
        def changed(response, context, ranking):
            selected = next(c for c in context['obstacle_candidates']
                            if c['obstacle_candidate_id'] == response['obstacle_candidate_id'])
            env.ego_state[:2] = selected['placements'][0]['center']
            return select(response, context, ranking)
        engine.static.select_obstacle = changed
        plan = engine.plan(low, env, env.get_state_for_planning(), ['静态组合'])
        assert plan is None or not plan.get('obstacle_plan')
        checks = low.last_trace['static_physical15']['final_checks']
        assert checks and not any(c['accepted'] for c in checks)
        assert low._validate_obstacle_plan == original
        assert len(calls) <= 2  # 原一次JSON修复预算不变。


def test_dynamic_priority_and_no_static_rule_in_dynamic_prompt():
    with install(True):
        env, low, engine, calls = setup(True)
        engine.static.plan = lambda *a, **k: pytest.fail('dynamic priority bypassed')
        plan = engine.plan(low, env, env.get_state_for_planning(), ['动态优先'])
        assert plan['attack_kind'] == 'dynamic'
        assert 'static_clearance_policy' not in calls[0]['context']
        assert not any(d['path'] == RULE_PATH for d in calls[0]['knowledge']['documents'])
        assert engine.calls == len(calls) == 1


def test_background_safety_is_revalidated_after_selection():
    with install(True):
        env, low, engine, _ = setup()
        curved(env)
        select = engine.static.select_obstacle
        def changed(response, context, ranking):
            selected = next(c for c in context['obstacle_candidates']
                            if c['obstacle_candidate_id'] == response['obstacle_candidate_id'])
            env.agent_active[0] = True
            env.data_dict['agent'][-1][0, :2] = selected['placements'][0]['center']
            env.data_dict['agent'][-1][0, 2:4] = 0.
            return select(response, context, ranking)
        engine.static.select_obstacle = changed
        plan = engine.plan(low, env, env.get_state_for_planning(), ['静态组合'])
        assert plan is None or not plan.get('obstacle_plan')
        checks = low.last_trace['static_physical15']['final_checks']
        assert any('background_CV_clearance' in c.get('reason', '') for c in checks)
        assert not any(c['accepted'] for c in checks)


def test_combined_budget_unchanged_across_dynamic_and_static():
    with install(True):
        env, low, engine, calls = setup(True, budget=1)
        assert engine.plan(low, env, env.get_state_for_planning(), [])['attack']
        env.agent_active[:] = False
        plan = engine.plan(low, env, env.get_state_for_planning(), [])
        assert not plan['attack'] and plan['reason'] == 'skills_api_budget_exhausted'
        assert engine.calls == len(calls) == 1


def test_scope_restores_constructor_after_error_and_rejects_nesting():
    before = DynamicFirstStaticPlanner.__init__
    with pytest.raises(RuntimeError, match='fixture'):
        with install(True):
            with pytest.raises(RuntimeError, match='already_installed'):
                with install(True):
                    pass
            raise RuntimeError('fixture')
    assert DynamicFirstStaticPlanner.__init__ is before


@pytest.mark.parametrize('value', ['on', 1, None])
def test_flags_are_explicit_booleans(value):
    with pytest.raises(ValueError):
        with install(value):
            pass
