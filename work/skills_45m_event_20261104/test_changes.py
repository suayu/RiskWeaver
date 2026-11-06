"""验证真实观察方法及混合路由状态恢复，传输全部使用替身。"""
from types import SimpleNamespace as NS
import numpy as np
import pytest
from policies.hybrid_skills_planner import DynamicFirstStaticPlanner


class Branch:
    def __init__(self,result,empty=False,service=False,error=False):
        self.calls=0;self.budget=4;self.result=result;self.empty=empty;self.service=service;self.error=error
    def build_context(self,*args,**kwargs):
        return {'candidates':[] if self.empty else [{'candidate_id':'c'}]}
    def plan(self,planner,*args,**kwargs):
        self.build_context(planner)
        self.calls+=1;planner.last_trace={'branch_calls':self.calls};planner.last_request_failed=self.service
        if self.error:raise RuntimeError('branch_failure')
        return self.result


def setup_router(dynamic,budget=4):
    router=object.__new__(DynamicFirstStaticPlanner)
    router.dynamic=dynamic;router.static=Branch({'attack':False,'obstacle_plan':[{'id':'o'}]})
    router.budget=budget;router.rag_runtime=None;router.dynamic_first_in_plan=True
    p=NS(use_obstacles=True,allow_trajectory=True,last_trace={},last_request_failed=False)
    return router,p


def test_empty_directory_routes_before_request():
    r,p=setup_router(Branch(None,empty=True));plan=r.plan(p,None,{},'')
    assert r.dynamic.calls==0 and r.static.calls==1 and plan['attack_kind']=='static'
    assert p.use_obstacles and p.allow_trajectory


def test_decline_can_use_remaining_budget():
    r,p=setup_router(Branch({'attack':False}));plan=r.plan(p,None,{},'')
    assert r.calls==2 and plan['attack_kind']=='static'
    assert p.last_trace['hybrid_routing']['dynamic_attempt_trace']


def test_service_failure_never_becomes_static_success():
    r,p=setup_router(Branch(None,service=True));assert r.plan(p,None,{},'') is None
    assert r.static.calls==0 and p.last_request_failed


def test_budget_is_shared():
    r,p=setup_router(Branch({'attack':False}),budget=1);plan=r.plan(p,None,{},'')
    assert not plan['attack'] and r.calls==1


def test_dynamic_priority():
    r,p=setup_router(Branch({'attack':True}));plan=r.plan(p,None,{},'')
    assert plan['attack_kind']=='dynamic' and r.static.calls==0


def test_exception_restores_state_and_budgets():
    r,p=setup_router(Branch(None,error=True));before=r.dynamic.build_context
    with pytest.raises(RuntimeError):r.plan(p,None,{},'')
    assert r.dynamic.build_context==before and r.dynamic.budget==4
    assert p.use_obstacles and p.allow_trajectory


def test_real_planning_state_uses45_for_agents_and_route():
    from simulator import Simulator
    states=np.array([[40,0,0,0,0,4,2,1],[46,0,0,0,0,4,2,1]],float)
    e=NS(ego_state=np.array([0,0,0,0,0,4,2,1]),nearby_distance=45,
        cfg=NS(sim=NS(traffic_model=NS(history_frames=11))),data_dict={'agent':[states]},
        agent_active=np.array([True,True]),current_step=0,
        scenario_dict={'route':np.array([[0,0],[40,0],[46,0]]),'agent_types':np.array([[0,1,0,0,0],[0,1,0,0,0]])},
        get_static_obstacles=lambda:[],get_scenario_frame=lambda **kw:NS(history_global=np.zeros((2,11,5)),history_mask=np.ones((2,11),bool)))
    state=Simulator.get_state_for_planning(e)
    assert [a['id'] for a in state['agents']]==[0]
    assert state['route']==[[0.,0.],[40.,0.]]
    assert set(state)=={'ego_state','route','agents','static_obstacles','history','history_order','current_step'}
