"""验证参考挑战不读取结果，碰撞只影响任务折半分一次。"""
import copy
import numpy as np
import pytest
from dual_track_evaluation import ReferenceChallenge,ability_score,apply_result

def scene():return dict(route=[[i,0] for i in range(201)],ego=[0,0,10,0,0,4.5,2,1],agents=np.zeros((0,8)).tolist(),active=[],static_obstacles=[])

def test_empty_scene_shape_and_result_independence():
    s=scene();s['agents']=[]
    # 明确空交通阵列仍须是二维；snapshot入口保留其正确形状。
    s['agents']=np.zeros((0,8));engine=ReferenceChallenge();a=engine.evaluate(s)
    t=copy.deepcopy(s);t.update(collision=True,off_route=True,completed=False)
    assert a['valid'] and a['score']==pytest.approx(0)
    assert engine.evaluate(t)==a

def test_obstacle_reduces_safe_control_set():
    s=scene();s['agents']=np.zeros((0,8));engine=ReferenceChallenge();a=engine.evaluate(s)
    s['static_obstacles']=[dict(id='o',center=[22,0],yaw=0,length=2,width=1)]
    b=engine.evaluate(s)
    assert b['safe_task_controls']<=a['safe_task_controls']
    assert b['control_space_loss']>=a['control_space_loss']

def test_ability_uses_no_collision_term_in_challenge():
    assert ability_score(.6,True,False,False,1)==60
    assert ability_score(.6,False,True,False,.8)==24
    assert ability_score(.6,False,False,False,.8)==24
    assert ability_score(None,True,False,False,1) is None

def test_result_preserves_legacy_score():
    r=dict(executed_steps=10,completed=False,collision=True,off_route=False,progress=.8,scenario_danger_score=.9,autonomous_driving_ability_score=36)
    out=apply_result(r,dict(valid=True,score=.6))
    assert out['historical_danger']==.9 and out['autonomous_driving_ability_score_legacy']==36
    assert out['autonomous_driving_ability_score']==24
    assert out['outcome_severity_score']==pytest.approx(.72)

def test_no_execution_cannot_receive_score():
    r=dict(executed_steps=0,completed=None,collision=None,off_route=None,progress=0)
    assert apply_result(r,dict(valid=True,score=.6))['autonomous_driving_ability_score'] is None
