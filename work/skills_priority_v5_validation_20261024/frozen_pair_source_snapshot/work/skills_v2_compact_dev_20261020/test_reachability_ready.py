"""在真实可达算法上核验路线尾部失效，不把测量问题冒充攻击过强。"""
from types import SimpleNamespace as NS
import copy
import numpy as np
import pytest
from reachability_ready_planner import baseline_readiness, ReachabilityReadyPlanner
from compact_planner import CompactSkillsPlanner


def env(x=10):
    return NS(scenario_dict={'route':np.column_stack((np.arange(101),np.zeros(101)))},
              ego_state=np.array([x,0,15,0,0,4.5,2]),
              data_dict={'agent':[np.empty((0,8))]},agent_active=np.empty(0,dtype=bool),
              pending_joint_trajectory=None,dt=.1,get_static_obstacles=lambda:[],
              cfg=NS(sim=NS(evaluation=NS(reachability=NS(backend='route_grid')))))


def test_route_end_can_empty_baseline_without_any_obstacle():
    early=baseline_readiness(env(10),NS(backend='route_grid'))
    late=baseline_readiness(env(80),NS(backend='route_grid'))
    assert early['ready'] and early['baseline_area_m2']>0
    assert not late['ready'] and late['baseline_area_m2']==0
    assert late['area_curve'][0]['area_m2']>0 and late['area_curve'][-1]['area_m2']==0


def test_withheld_candidates_are_preserved_with_distinct_reason(monkeypatch):
    supplied={'candidates':[{'candidate_id':'x0','anchors':[[0.,0.],[1.,0.]]}],
              'obstacle_candidates':[{'candidate_id':'o0'}]}
    monkeypatch.setattr(CompactSkillsPlanner,'build_context',lambda *a,**k:copy.deepcopy(supplied))
    p=ReachabilityReadyPlanner.__new__(ReachabilityReadyPlanner)
    p.settings=NS(reachability_preflight_enabled=True)
    p.sim_cfg=env().cfg.sim
    result=p.build_context(env(80),None,None)
    assert result['candidates']==[] and result['obstacle_candidates']==[]
    assert result['preflight_withheld_candidates']==supplied['candidates']
    assert result['preflight_withheld_obstacles']==supplied['obstacle_candidates']
    assert result['measurement_preflight']['reason']=='baseline_reachability_not_ready'
    assert supplied['candidates']


def test_default_off_does_not_add_a_new_gate(monkeypatch):
    supplied={'candidates':[{'candidate_id':'x0'}]}
    monkeypatch.setattr(CompactSkillsPlanner,'build_context',lambda *a,**k:copy.deepcopy(supplied))
    p=ReachabilityReadyPlanner.__new__(ReachabilityReadyPlanner);p.settings=NS()
    assert p.build_context(env(80),None,None)==supplied


def test_commonroad_cannot_be_silently_replaced_by_route_grid():
    with pytest.raises(ValueError,match='route_grid'):
        baseline_readiness(env(),NS(backend='commonroad'))
