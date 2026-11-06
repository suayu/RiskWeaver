"""机会触发不执行攻击，验证静默、预算、冷却及后续新车事件。"""
from types import SimpleNamespace as NS
from policies.planning_events import OpportunityScheduler


def state(actors=(),route=(),static=()):
    return dict(ego_state=[0,0,0,0,0],agents=[dict(id=i,state=s) for i,s in actors],route=list(route),static_obstacles=list(static))


def test_empty_scene_does_not_consume_request():
    s=OpportunityScheduler();assert not s.decide(state(),0,4)['query']


def test_static_opportunity_and_existing_obstacle():
    s=OpportunityScheduler();v=state(route=[[0,0],[25,0]])
    assert s.decide(v,0,4)['query']
    assert not s.decide(state(route=[[0,0],[25,0]],static=[dict(center=[20,0])]),1,4)['query']


def test_later_vehicle_triggers_without_periodic_empty_queries():
    s=OpportunityScheduler()
    for t in range(50):assert not s.decide(state(),t,4)['query']
    assert s.decide(state([(1,[20,0,-5,0,0])]),50,4)['query']


def test_healthy_intent_is_not_replaced():
    s=OpportunityScheduler();v=state([(1,[8,0,-1,0,0])])
    assert not s.decide(v,5,4,active_intent=dict(target_id=1))['query']


def test_shared_budget_and_reservation():
    s=OpportunityScheduler();v=state(route=[[0,0],[25,0]])
    assert not s.decide(v,0,0)['query']
    assert not s.decide(v,10,1)['query']
    assert s.decide(v,60,1)['query']


def test_no_repeated_query_inside_cooldown():
    s=OpportunityScheduler();v=state([(1,[20,0,-1,0,0])])
    first=s.decide(v,0,4);assert first['query'];s.record_query(0,first)
    assert not s.decide(v,1,3)['query']
    assert s.decide(v,20,3)['query']
