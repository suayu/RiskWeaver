"""调用真实生成器step，传输/意图层使用替身，验证不增加模型预算。"""
from types import SimpleNamespace as NS
from test_generator_wiring_unit import fixture
from event_budget_generator import EventBudgetGenerator


def generator(enabled=True,screen_limit=40,consumption=None,service_failure=False):
    original=fixture();g=EventBudgetGenerator.__new__(EventBudgetGenerator);g.__dict__.update(original.__dict__)
    g.event_budget_enabled=enabled;g.max_screening_queries=screen_limit;g.screening_query_count=0
    g.skills_planner=NS(calls=0);g._max_call_times=4;g.attack_frequency=10;g.query_trace=[]
    g.llm_planner.last_request_failed=False;g.probed=[]
    def plan(env):
        g.probed.append(env.current_step);g.last_query_frame=env.current_step
        g.skills_planner.calls+=(consumption or {}).get(env.current_step,0)
        if service_failure:g.llm_planner.last_request_failed=True;g._attacks_enabled=False
    g.plan_and_inject=plan
    return g


def run(g,steps=110):
    env=NS(attack_intent=None,current_step=0)
    for step in range(steps):env.current_step=step;g.step(env,step)


def test_empty_early_catalog_does_not_consume_model_budget():
    g=generator(consumption={44:1,55:1,66:1,77:1});run(g)
    assert g.probed==[0,11,22,33,44,55,66,77]
    assert g.skills_planner.calls==4 and g._llm_call_times==4 and not g._attacks_enabled


def test_disabled_mode_keeps_original_four_opportunities():
    g=generator(enabled=False,consumption={44:1});run(g)
    assert g.probed==[0,11,22,33] and g.skills_planner.calls==0


def test_screening_has_its_own_finite_cap():
    g=generator(screen_limit=3);run(g,400)
    assert g.probed==[0,11,22] and g.skills_planner.calls==0 and not g._attacks_enabled


def test_schema_repairs_consume_the_same_four_call_budget():
    g=generator(consumption={0:2,11:2});run(g)
    assert g.probed==[0,11] and g.skills_planner.calls==4 and not g._attacks_enabled


def test_service_circuit_breaker_is_not_reopened():
    g=generator(consumption={0:1},service_failure=True);run(g)
    assert g.probed==[0] and not g._attacks_enabled
