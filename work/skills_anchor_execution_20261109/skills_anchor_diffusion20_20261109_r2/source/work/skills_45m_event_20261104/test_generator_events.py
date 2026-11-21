"""通过真实step入口验证事件触发与旧fixed时序；不调用API或仿真。"""
from types import SimpleNamespace as NS
from scenario_generator import AdversarialScenarioGenerator
from policies.planning_events import OpportunityScheduler


def make(mode):
    g=AdversarialScenarioGenerator.__new__(AdversarialScenarioGenerator)
    g.last_attack_frame=0;g.attack_duration=10;g.last_query_frame=0
    g._attacks_enabled=True;g.min_planning_step=0;g.attack_frequency=10
    g._llm_call_times=0;g._max_call_times=4;g._ignore_call_times=0
    g.planning_trigger_mode=mode;g.planning_scheduler=OpportunityScheduler();g.query_trace=[]
    budget=NS(calls=0,limit=4);g.llm_planner=NS(_physical_call_budget=budget)
    g._finish_iterative_window=lambda:None
    g.calls=[]
    def plan(env):
        g.calls.append(env.current_step);g.last_query_frame=env.current_step;budget.calls+=1
    g.plan_and_inject=plan
    state=dict(ego_state=[0,0,0,0,0],agents=[],route=[],static_obstacles=[])
    env=NS(current_step=0,attack_intent=None,get_state_for_planning=lambda:state)
    return g,env,state


def test_empty_early_frames_preserve_real_budget_for_later_vehicle():
    g,e,s=make('event')
    for t in range(50):e.current_step=t;g.step(e,t)
    assert not g.calls and g.llm_planner._physical_call_budget.calls==0
    s['agents']=[dict(id=1,state=[20,0,-5,0,0])]
    e.current_step=50;g.step(e,50)
    assert g.calls==[50]
    assert g.query_trace[-1]['kind']=='planning_trigger'


def test_fixed_mode_preserves_strict_old_interval():
    g,e,s=make('fixed')
    for t in range(30):e.current_step=t;g.step(e,t)
    assert g.calls==[0,11,22]


def test_no_calls_after_physical_budget_exhausted():
    g,e,s=make('event');g.llm_planner._physical_call_budget.calls=4
    s['agents']=[dict(id=1,state=[20,0,-5,0,0])]
    g.step(e,0);assert not g.calls
