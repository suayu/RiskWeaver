"""避免把相对时间锚点错配、换车误合并或超期实际轨迹当作跟随误差。"""
import numpy as np
import pytest
from temporal_audit import sample,continuity,tracking_metrics


def test_resampled_same_world_future_has_zero_replan_jump():
    old=dict(source_step=0,target_id=0,anchors=[[0,0],[0,10],[0,20],[0,30]])
    new=dict(source_step=10,target_id=0,anchors=[[0,10],[0,20],[0,30],[0,40]])
    r=continuity(old,new,[0,10])
    assert r['comparable'] and r['max_raw_overlap_deviation_m']==0 and r['overlap_seconds']==2.


def test_observed_tracking_offset_is_separate_from_plan_shape_change():
    old=dict(source_step=0,target_id=0,anchors=[[0,0],[0,10],[0,20],[0,30]])
    new=dict(source_step=10,target_id=0,anchors=[[2,10],[2,20],[2,30],[2,40]])
    r=continuity(old,new,[2,10])
    assert r['source_tracking_error_m']==2 and r['max_raw_overlap_deviation_m']==2
    assert r['max_tracking_aligned_shape_deviation_m']==0


def test_different_vehicle_is_new_episode_not_large_jump():
    old=dict(source_step=0,target_id=0,anchors=[[0,0]]*4)
    new=dict(source_step=10,target_id=1,anchors=[[100,100]]*4)
    assert continuity(old,new,[100,100])==dict(comparable=False,reason='different_target_new_episode')


def test_no_extrapolation_outside_sparse_intention_horizon():
    with pytest.raises(ValueError,match='outside_sparse'):sample([[0,0]]*4,[3.1])


def test_prediction_zero_is_next_frame_and_actual_stops_at_declared_lease():
    plan=dict(source_step=0,target_id=0,duration_frames=10,anchors=[[0,0],[0,10],[0,20],[0,30]])
    states={i:dict(active=[True],agents=[[0,i]]) for i in range(1,11)}
    joint=dict(agent_ids=[0],positions_global=[[[0,i] for i in range(1,33)]],valid_mask=[[True]*32],metadata={})
    r=tracking_metrics(plan,states,joint)
    assert r['actual_lease_end_error_m']==0 and r['prediction_first_sample_matches_next_actual']
    assert [x['prediction_index'] for x in r['predicted_anchor_errors']]==[9,19,29]
    assert [x['error_m'] for x in r['predicted_anchor_errors']]==[0,0,0]
    assert len(r['actual_declared_lease_samples'])==10 and r['later_actual_states_not_scored_against_expired_plan']


def test_missing_actual_observation_keeps_tracking_invalid():
    p=dict(source_step=0,target_id=0,duration_frames=10,anchors=[[0,0]]*4)
    assert tracking_metrics(p,{},None)['valid'] is False
