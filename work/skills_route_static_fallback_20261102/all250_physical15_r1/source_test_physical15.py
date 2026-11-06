"""强化后备只改变新增分支净距；复核折返近处反例与原成功路径。"""
import math
from adapter import install
from test_adapter import setup


def test_near_foldback_rejected_by_physical15():
    env, planner, _, _ = setup()
    env.scenario_dict['route'] = [[0, 0], [0, 20], [20, 20], [20, 0], [4, 0]]
    # fixture自车朝向为全局x，规划器局部y为前向；世界(4,0)对应局部(0,4)。
    payload = [dict(type='cone_line', center=[0, 4], yaw=math.pi/2, count=1, spacing=1)]
    assert planner._validate_obstacle_plan(payload, {}) == []
    with install(planner, env, enabled=True, initial_margin=.5) as old:
        assert planner._validate_obstacle_plan(payload, {}) == payload
        assert old['events'][0]['route_clearance_m'] > 70
    with install(planner, env, enabled=True, initial_margin=15.) as strict:
        assert planner._validate_obstacle_plan(payload, {}) == []
        assert strict['events'][0]['reason'] == 'initial_ego_OBB_clearance'


def test_far_curved_route_still_has_legal_placement():
    env, planner, _, _ = setup()
    env.scenario_dict['route'] = [[0, 0], [10, 0], [10, 100]]
    payload = [dict(type='cone_line', center=[-20, 10], yaw=math.pi, count=1, spacing=1)]
    with install(planner, env, enabled=True, initial_margin=15.) as receipt:
        assert planner._validate_obstacle_plan(payload, {}) == payload
        assert min(receipt['events'][0]['initial_ego_OBB_clearances_m']) >= 15


def test_original_success_unchanged_with_stricter_fallback():
    env, planner, _, _ = setup()
    payload = [dict(type='cone_line', center=[0, 25], yaw=math.pi/2, count=1, spacing=1)]
    expected = planner._validate_obstacle_plan(payload, {})
    with install(planner, env, enabled=True, initial_margin=15.) as receipt:
        assert planner._validate_obstacle_plan(payload, {}) == expected
        assert receipt['events'] == []
