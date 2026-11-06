import importlib.util
from pathlib import Path
from types import MethodType
from test_generator_events import make

spec=importlib.util.spec_from_file_location('staged_budget_v2',Path(__file__).with_name('scenario_generator_budget_v2.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def test_empty_checks_do_not_disable_future_opportunity():
    g,e,s=make('event');g.step=MethodType(module.AdversarialScenarioGenerator.step,g)
    s['route']=[[0,0],[25,0]]
    def no_api(env):g.last_query_frame=env.current_step
    g.plan_and_inject=no_api
    for t in range(120):e.current_step=t;g.step(e,t)
    assert g._attacks_enabled and g._llm_call_times==0
    def actual(env):g.calls.append(env.current_step);g.last_query_frame=env.current_step;g.llm_planner._physical_call_budget.calls+=1
    g.plan_and_inject=actual;s['agents']=[dict(id=1,state=[20,0,-5,0,0])]
    for t in range(120,150):e.current_step=t;g.step(e,t)
    assert g.calls and g._llm_call_times>0

def test_fixed_counting_unchanged():
    g,e,s=make('fixed');g.step=MethodType(module.AdversarialScenarioGenerator.step,g)
    for t in range(30):e.current_step=t;g.step(e,t)
    assert g.calls==[0,11,22] and g._llm_call_times==3
