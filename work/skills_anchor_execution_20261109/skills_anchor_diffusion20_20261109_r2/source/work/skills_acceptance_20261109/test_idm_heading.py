"""直接调用 IDM 类方法，验证停车与移动车辆行为。"""
from types import SimpleNamespace

import numpy as np

from policies.idm_policy import IDMPolicy
from run_idm_heading import install

original = IDMPolicy._get_next_states
fixture = SimpleNamespace(
    _get_accelerations=lambda states:{0:0}, controlled_agent_ids=[0],
    _get_ego_path=lambda i:[0], actors_current_lane={0:0},
    lane_geometries={0:np.array([[0,0],[0,1]])},
    _get_closest_lane_point_from_position=lambda line,pos:(0,0),
    t=2, dt=.1, _get_next_path_position=lambda *args:(np.array([0.,1.]),0,1,False),
)

def states(speed):
    return np.array([[0.,0.,0.,speed,np.pi/2]])

old_still = original(fixture,states(0))
old_moving = original(fixture,states(1))
restore = install(IDMPolicy)
try:
    still = IDMPolicy._get_next_states(fixture,states(0))
    assert old_still[2][0] == 0 and still[2][0] == np.pi/2
    assert all(still[i] == old_still[i] for i in (0,1,3))
    assert IDMPolicy._get_next_states(fixture,states(1)) == old_moving
finally:
    restore()
assert IDMPolicy._get_next_states is original
print("IDM stationary, moving and restore checks passed")
