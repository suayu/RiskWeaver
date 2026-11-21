"""直接执行IDM真实积分方法，覆盖强制动、正常运动和旧行为。"""
from types import SimpleNamespace as NS
import numpy as np
import pytest
from policies.idm_policy import IDMPolicy


def policy(acceleration, enabled):
    p = IDMPolicy.__new__(IDMPolicy)
    p.cfg = NS(sim=NS(idm_nonnegative_speed=enabled))
    p.dt=.1
    p.t=2
    p.controlled_agent_ids=[0]
    p.actors_current_lane={0:0}
    p.lane_geometries={0:np.array([[0.,0.],[10.,0.]])}
    p._get_accelerations=lambda states:{0:acceleration}
    p._get_ego_path=lambda ident:[0]
    p._get_closest_lane_point_from_position=lambda *args:(0,0)
    p._get_next_path_position=lambda *args:(np.array([10.,0.]),0,1,False)
    return p


def test_strong_brake_stops_at_analytic_stopping_distance():
    p=policy(-100.,True)
    x,y,heading,v=p._get_next_states(np.array([[0.,0.,5.,0.,0.,4.,2.,1.]]))
    assert v[0]==0.
    assert x[0]==pytest.approx(.125)
    assert y[0]==0.


def test_legacy_flag_retains_historical_negative_speed():
    p=policy(-100.,False)
    x,y,heading,v=p._get_next_states(np.array([[0.,0.,5.,0.,0.,4.,2.,1.]]))
    assert v[0]==-5.
    assert x[0]==pytest.approx(.5)


def test_normal_acceleration_retains_original_integration():
    p=policy(2.,True)
    x,y,heading,v=p._get_next_states(np.array([[0.,0.,5.,0.,0.,4.,2.,1.]]))
    assert v[0]==pytest.approx(5.2)
    assert x[0]==pytest.approx(.5)


def test_stopped_vehicle_stays_stopped_under_braking():
    p=policy(-100.,True)
    x,y,heading,v=p._get_next_states(np.array([[0.,0.,0.,0.,0.,4.,2.,1.]]))
    assert v[0]==0. and x[0]==0.
