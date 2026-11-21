"""未来时间、跨场景生命周期与同次攻击延续的窄范围回归。"""
import copy
import numpy as np
import pytest
from rolling_facts import ego_hypotheses,forecast_point,future_card,AttackMemory,bounded_future_proposals,remaining_intent_candidate


def state():return dict(ego_state=[0,0,0,10,np.pi/2],route=[[0,0],[0,50],[0,100]])


def plan(target=0):return dict(attack=True,attack_target_id=target,strategy='lane_change',duration=10,
    anchors=[[0,30],[0,20],[0,10],[0,0]])


def test_future_is_time_aligned_and_derived_only_from_source():
    f=ego_hypotheses(state(),11)
    assert np.allclose(forecast_point(f,f['hypotheses'][0],1),[0,10])
    assert f['absolute_times_seconds'][0]==1.1 and not f['uses_future_ground_truth']
    assert not f['actual_policy_rollout_available'] and f['probabilities'] is None


def test_route_tail_does_not_become_constant_endpoint_or_extrapolated_future():
    s=state();s['route']=[[0,0],[0,5]]
    f=ego_hypotheses(s,0)
    assert len(f['hypotheses'])==1 and 'no_extrapolation' in f['route_status']


def test_opposite_route_not_mistaken_for_ego_future():
    s=state();s['route']=[[0,0],[0,-100]]
    assert len(ego_hypotheses(s,0)['hypotheses'])==1


def test_forecast_min_outside_lease_is_not_active_attack():
    p=plan();c=dict(anchors=p['anchors'],duration=p['duration'])
    card=future_card(c,ego_hypotheses(state(),0))
    assert card['rows'][0]['forecast_min_time_seconds']==1.5
    assert card['rows'][0]['forecast_min_outside_active_lease']
    assert card['rows'][0]['active_min_center_gap_m']==10.


def test_memory_preserves_absolute_event_and_distinguishes_expired_control():
    m=AttackMemory();f=ego_hypotheses(state(),0);r=m.commit('s',0,plan(),f,dict(reason='first'))
    prior=m.inspect('s',11,[True]);assert prior['previous_lease_expired']
    p=plan();p['anchors']=[[0,19],[0,9],[0,-1],[0,-11]]
    r2=m.commit('s',11,p,ego_hypotheses(state(),11),prior)
    assert r2['episode_id']==r['episode_id'] and r2['event_absolute_seconds']==r['event_absolute_seconds']
    assert not r2['execution_certified']


@pytest.mark.parametrize('scene,step,active,reason',[('different',11,[True],'scene_changed'),
    ('s',11,[False],'previous_target_inactive'),('s',31,[True],'previous_forecast_expired'),
    ('s',-1,[True],'source_time_reversed')])
def test_memory_reset_has_explicit_reason(scene,step,active,reason):
    m=AttackMemory();m.commit('s',0,plan(),ego_hypotheses(state(),0),{})
    assert m.inspect(scene,step,active)['reason']==reason and m.previous is None


def test_no_attack_does_not_overwrite_last_accepted_intent():
    m=AttackMemory();m.commit('s',0,plan(),ego_hypotheses(state(),0),{});old=copy.deepcopy(m.previous)
    assert not m.commit('s',11,dict(attack=False),ego_hypotheses(state(),11),{})['updated']
    assert m.previous==old


def test_continuation_only_reuses_covered_old_world_times():
    old=dict(target_id=0,source_step=0,anchors=[[0,0],[0,10],[0,20],[0,30]],episode_id='e')
    fresh=dict(target_id=0,anchors=[[2,11],[2,9],[2,7],[2,5]],duration=10)
    new=remaining_intent_candidate(old,fresh,[2,11],11)
    assert np.allclose(new['anchors'][0],[2,11]) and np.allclose(new['anchors'][1],[2,21])
    assert new['anchors'][2:]==fresh['anchors'][2:] and new['reused_relative_anchor_seconds']==[1]
    assert new['fresh_tail_relative_anchor_seconds']==[2,3] and not new['old_forecast_extrapolated']
    assert remaining_intent_candidate(old,dict(fresh,target_id=1),[2,11],11) is None


def test_rules_target_future_ego_location_and_keep_sparse_acceleration_cap():
    f=ego_hypotheses(state(),0)
    proposals=bounded_future_proposals([0,0,0,10,1.57],[10,20,-2,0,3.14],0,f)
    assert proposals and any(c['event_relative_seconds']==1. for c in proposals)
    for c in proposals:
        a=np.asarray(c['anchors']);assert a.shape==(4,2) and c['duration']==10
        assert np.linalg.norm(a[2]-2*a[1]+a[0])<=3.8+1e-8
        assert c['recipe_basis'].startswith('future_')

