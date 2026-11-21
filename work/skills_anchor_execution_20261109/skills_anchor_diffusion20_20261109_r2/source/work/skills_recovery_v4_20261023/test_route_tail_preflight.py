"""只筛选新攻击机会；原始路线、状态和3秒测量窗口保持。"""
from types import SimpleNamespace as NS
import copy
import numpy as np
import pytest
from route_tail_preflight import route_tail_readiness


def env(remaining=80,speed=10):
    route=np.column_stack([np.arange(0,201,dtype=float),np.zeros(201)])
    return NS(ego_state=np.array([200-remaining,0,speed,0,0,4.5,1.8,1.]),dt=.1,
        scenario_dict={'route':route},data_dict={'agent':[np.empty((0,8))]},agent_active=np.array([],bool))


def test_tail_guard_can_decline_while_current_horizon_is_still_valid():
    from policies.risk_metrics import RouteReachability
    e=env(29,14);cfg=NS(horizon_seconds=3.)
    original=copy.deepcopy(e.ego_state);route=e.scenario_dict['route'].copy();e.pending_joint_trajectory=None
    assert RouteReachability(cfg).compute(e,False,[])['area_m2']>0
    result=route_tail_readiness(e,cfg)
    assert not result['ready'] and result['unchanged_evaluation_horizon_seconds']==3
    np.testing.assert_array_equal(e.ego_state,original);np.testing.assert_array_equal(e.scenario_dict['route'],route)


def test_long_remaining_route_preserves_opportunity():
    result=route_tail_readiness(env(120,10),NS())
    assert result['ready'] and result['future_unobstructed_area_m2']>0


def test_no_synthetic_road_is_added_when_duration_reaches_end():
    result=route_tail_readiness(env(5,10),NS())
    assert not result['ready'] and result['reason']=='nominal_route_end_within_attack_duration'


@pytest.mark.parametrize('duration',[0,101,True,1.2])
def test_invalid_duration_rejected(duration):
    with pytest.raises(ValueError):route_tail_readiness(env(),NS(),duration)


def test_other_backend_is_not_silently_replaced_by_route_grid():
    with pytest.raises(ValueError,match='backend_not_supported'):route_tail_readiness(env(),NS(backend='commonroad'))


def test_route_tail_decline_keeps_catalog_and_makes_no_LLM_request(monkeypatch):
    import route_tail_preflight as module
    candidate=dict(candidate_id='kept',target_id=0)
    monkeypatch.setattr(module.ReachabilityReadyPlanner,'build_context',
        lambda *a,**k:dict(candidates=[candidate],measurement_preflight=dict(ready=True)))
    monkeypatch.setattr(module,'route_tail_readiness',
        lambda *a,**k:dict(ready=False,reason='route_tail_measurement_margin_insufficient'))
    model=NS(last_trace={})
    def forbidden(*a,**k):raise AssertionError('declined opportunity must not call LLM')
    model._request_attack_plan=forbidden
    skills=object.__new__(module.RouteTailReadyPlanner);skills.calls=0
    e=NS(cfg=NS(sim=NS(evaluation=NS(reachability=NS()))))
    result=skills.plan(model,e,{},[])
    assert not result['attack'] and skills.calls==0
    assert model.last_trace['stage']=='route_tail_preflight_decline'
    assert model.last_trace['context']['route_tail_withheld_candidates']==[candidate]
