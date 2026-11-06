"""验证重放状态绑定、只一次注入和时域处理，不调用API或仿真。"""
import copy
import pytest
from fixed_intent_lifecycle import RecordedIntentPlanner


def test_only_initial_state_can_receive_the_recorded_intent():
    state=dict(current_step=0,ego_state=[0,0,1,0],agents=[],route=[[0,0],[10,0]],static_obstacles=[])
    plan=dict(attack=True,attack_target_id=0,strategy='lane_change',anchors=[[0,0],[1,1],[2,2],[3,3]],duration=10)
    p=RecordedIntentPlanner(state,plan,30);first=p.generate_attack_plan(state)
    assert first['duration']==30 and first['anchors']==plan['anchors'] and plan['duration']==10
    assert not p.generate_attack_plan(state)['attack'] and p.calls==1
    assert p.last_trace['inference_calls']==0


def test_geometry_mismatch_is_refused_instead_of_reusing_stale_plan():
    state=dict(current_step=0,ego_state=[0,0,1,0],agents=[],route=[[0,0],[10,0]])
    p=RecordedIntentPlanner(state,{},10);changed=copy.deepcopy(state);changed['ego_state'][0]=.01
    with pytest.raises(RuntimeError,match='physics_state_mismatch'):p.generate_attack_plan(changed)
