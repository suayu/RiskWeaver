"""必要加速度包络的边界、不可达反例和刚体不变性。"""
import numpy as np
import pytest
from intent_envelope import acceleration_envelope


def test_maximum_constant_acceleration_hits_necessary_boundary():
    state=np.array([1.,2.,3.,4.]);t=np.arange(4.)[:,None]
    points=state[:2]+t*state[2:4]+.5*t*t*np.array([6.,0.])
    r=acceleration_envelope(state,points)
    assert r['necessary_exact_anchor_condition_pass'] and np.allclose(r['signed_excess_m'],0)
    assert not r['sufficient_dynamic_or_background_or_ego_area_certificate']


def test_far_first_second_point_proves_exact_tracking_impossible():
    r=acceleration_envelope([0,0,0,0],[[0,0],[4,0],[5,0],[6,0]])
    assert not r['necessary_exact_anchor_condition_pass'] and r['signed_excess_m'][0]==1.
    assert not r['new_runtime_gate']


def test_rotation_and_translation_do_not_change_condition():
    state=np.array([1.,2.,3.,4.]);points=np.array([[1.,2.],[5,9],[9,15],[15,20]])
    theta=.7;rot=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    transformed=np.r_[state[:2]@rot.T+[100,-50],state[2:4]@rot.T]
    a=acceleration_envelope(state,points);b=acceleration_envelope(transformed,points@rot.T+[100,-50])
    assert np.allclose(a['signed_excess_m'],b['signed_excess_m'])


def test_nonfinite_anchor_cannot_be_certified():
    with pytest.raises(ValueError):acceleration_envelope([0,0,0,0],[[0,0],[1,0],[2,0],[float('nan'),0]])
