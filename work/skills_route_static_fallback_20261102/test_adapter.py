"""用公共原验证器检查关闭路径、曲线后备、非法模板和背景安全；零API。"""
import importlib.util
import math
import sys
from pathlib import Path

ROOT=Path('/home2/zhaoyx/scenario-dreamer')
sys.path[:0]=[str(ROOT),str(ROOT/'tests'),str(Path(__file__).resolve().parent)]
import pytest
from adapter import install
from policies.static_attack_catalog import validate_placements


def setup():
    spec=importlib.util.spec_from_file_location('route_clearance_public_fixture',
        ROOT/'work/hybrid_public_integration_20261101/test_public_hybrid.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.setup(False)


def test_disabled_is_exact_original_and_does_not_bind_new_method():
    env,p,_,_=setup();before=p._validate_obstacle_plan
    payload=[dict(type='cone_line',center=[0,25],yaw=math.pi/2,count=1,spacing=1)]
    expected=before(payload,{})
    with install(p,env,enabled=False) as receipt:
        assert p._validate_obstacle_plan==before
        assert p._validate_obstacle_plan(payload,{})==expected
        assert not receipt['events']
    assert p._validate_obstacle_plan==before


def test_original_straight_accepts_unchanged_and_guard_records_no_fallback():
    env,p,_,_=setup();payload=[dict(type='cone_line',center=[0,25],yaw=math.pi/2,count=1,spacing=1)]
    original=p._validate_obstacle_plan;expected=original(payload,{})
    with install(p,env,enabled=True) as receipt:
        assert p._validate_obstacle_plan(payload,{})==expected and not receipt['events']
    assert p._validate_obstacle_plan==original


def test_curved_placement_with_far_future_route_is_accepted_by_fallback():
    env,p,_,_=setup()
    env.scenario_dict['route']=[[0,0],[10,0],[10,100]]
    payload=[dict(type='cone_line',center=[-20,10],yaw=math.pi,count=1,spacing=1)]
    assert p._validate_obstacle_plan(payload,{})==[]
    with install(p,env,enabled=True) as receipt:
        assert p._validate_obstacle_plan(payload,{})==payload
        assert receipt['events'][0]['accepted']
        assert receipt['events'][0]['route_clearance_m']>15


def test_invalid_template_and_group_limit_remain_rejected():
    env,p,_,_=setup()
    with install(p,env,enabled=True) as receipt:
        assert p._validate_obstacle_plan([dict(type='imaginary',center=[0,30],yaw=0,count=1,spacing=1)],{})==[]
        with pytest.raises(ValueError):p._validate_obstacle_plan([{}]*3,{})
        assert not receipt['events']


def test_initial_ego_collision_still_rejected():
    env,p,_,_=setup();payload=[dict(type='cone_line',center=[0,1],yaw=0,count=1,spacing=1)]
    with install(p,env,enabled=True) as receipt:
        assert p._validate_obstacle_plan(payload,{})==[]
        assert receipt['events'][0]['reason']=='initial_ego_OBB_clearance'


def test_public_background_safety_still_rejects_even_when_route_gate_passes():
    env,p,engine,_=setup();env.scenario_dict['route']=[[0,0],[10,0],[10,100]]
    env.agent_active[0]=True;env.data_dict['agent'][-1][0,:5]=[10,20,0,0,0]
    placement=[dict(type='cone_line',center=[10,20],yaw=math.pi/2,count=1,spacing=1)]
    with install(p,env,enabled=True) as receipt:
        result=validate_placements(env,p,engine.static.builder,placement)
        assert any(x['accepted'] for x in receipt['events'])
        assert not result['accepted'] and result['reason']=='background_CV_clearance'


def test_method_restores_on_exception():
    env,p,_,_=setup();before=p._validate_obstacle_plan
    with pytest.raises(RuntimeError):
        with install(p,env,enabled=True):raise RuntimeError('fixture')
    assert p._validate_obstacle_plan==before


def test_margin_cannot_be_lowered():
    env,p,_,_=setup()
    with pytest.raises(ValueError):
        with install(p,env,enabled=True,initial_margin=0):pass
