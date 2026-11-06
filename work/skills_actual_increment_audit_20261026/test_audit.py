"""验证实际执行审计不会混淆自然运动、过期计划和无效状态。"""
import copy
import math
import numpy as np
import pytest
from audit import clearance,measure_plan,execution_ledger


def example():
    states={i:dict(ego=[0,0,0,0,0,2,2,1],agents=[[10-.2*i,0,-2,0,math.pi,2,2,1]],active=[True]) for i in range(11)}
    plan=dict(source_step=0,target_id=0,request_id=None,duration_frames=10,actual_execution_steps=list(range(1,11)))
    return plan,states


def test_OBB_clearance_touching_overlap_and_rigid_transform():
    a=[0,0,0,0,0,2,2];b=[4,0,0,0,0,2,2]
    assert clearance(a,b)==2
    assert clearance(a,[2,0,0,0,0,2,2])==0
    assert clearance(a,[1.5,0,0,0,0,2,2])==0
    for x in [a,b]:x[:2]=[-x[1]+3,x[0]-7];x[4]+=math.pi/2
    assert abs(clearance(a,b)-2)<1e-10
    assert abs(clearance([0,0,0,0,0,2,2],[4,4,0,0,0,2,2])-math.sqrt(8))<1e-10


def test_natural_approach_is_not_increment():
    plan,states=example();before=copy.deepcopy(states);result=measure_plan(plan,states)
    assert result['valid'] and result['increment_fixed_ego_CV_OBB_m']==0
    assert states==before


def test_faster_actual_closure_positive_braking_negative():
    plan,states=example()
    for i,s in states.items():s['agents'][0][0]=10-.3*i
    assert abs(measure_plan(plan,states)['increment_fixed_ego_CV_OBB_m']-1)<1e-10
    for i,s in states.items():s['agents'][0][0]=10-.1*i
    assert abs(measure_plan(plan,states)['increment_fixed_ego_CV_OBB_m']+1)<1e-10


def test_missing_states_and_expired_markers_are_invalid_not_zero():
    plan,states=example();del states[5]
    result=measure_plan(plan,states)
    assert not result['valid'] and result['increment_fixed_ego_CV_OBB_m'] is None
    plan,states=example();plan['actual_execution_steps'].append(11)
    assert measure_plan(plan,states)['reason']=='not_full_declared_guided_window'
    plan,states=example();states[7]['active'][0]=False
    assert measure_plan(plan,states)['reason']=='inactive_or_missing_target'


def test_null_request_ids_are_attributed_by_target_and_latest_time():
    rows=[dict(kind='plan_accepted',step=s,target_id=0,request_id=None) for s in [0,11]]
    rows += [dict(kind='attack_execution',step=12,target_id=0,request_id=None,diffusion_source_step=11,executed=True),
             dict(kind='execution_prediction',step=11,joint=dict(metadata=dict(anchor_guidance_active=True)))]
    result=execution_ledger(rows)
    assert result['plans'][0]['actual_execution_steps']==[]
    assert result['plans'][1]['actual_execution_steps']==[12]
    assert not result['unmatched'] and not result['unguided']
