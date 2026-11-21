"""当前趋势、比较集合、时间分离与异常回退，不把描述阈值变为运行门槛。"""
import copy
import json
import math
import pytest
from quantity_facts import motion_fact,comparison_groups,build_quantity_facts


def source():
    state=dict(current_step=3,ego_state=[0,0,0,5,math.pi/2],agents=[dict(id=0,state=[0,-20,0,3,math.pi/2])])
    candidates=[dict(candidate_id='a',target_id=0,background_safety_margin_m=16.164,recipe_basis='active_time_0.500s'),
        dict(candidate_id='b',target_id=0,background_safety_margin_m=14.851)]
    cards={cid:dict(active_seconds=1.,anchor_horizon_seconds=3.,full_active_window_covered=True,min_gap_difference_m=value) for cid,value in [('a',1.),('b',.766)]}
    return state,dict(source_step=3,candidates=candidates,incremental_cards=cards)


def test_source_separating_vs_possible_future_candidate_closing_are_distinct():
    s,c=source();out=build_quantity_facts(s,c,3)
    assert out['actor_current_motion']['0']['radial_gap_rate_mps']==2.
    assert out['actor_current_motion']['0']['current_distance_direction']=='increasing_separating'
    assert out['comparisons']['active_center_gap_CV_increment_m']['values']['a']==1.
    assert out['comparisons']['active_center_gap_CV_increment_m']['larger_first_tie_groups']==[['a'],['b']]
    assert out['comparisons']['supplied_background_margin_m']['smaller_first_tie_groups']==[['b'],['a']]


def test_recipe_time_does_not_shorten_validity_or_forecast():
    s,c=source();out=build_quantity_facts(s,c,3);a=out['per_candidate_time']['a']
    assert [a['recipe_design_time_seconds'],a['active_horizon_seconds'],a['anchor_forecast_horizon_seconds']]==[.5,1.,3.]
    assert out['per_candidate_time']['b']['recipe_design_time_seconds'] is None


def test_negative_null_and_partial_proxy_never_dropped_from_candidate_catalog_or_filled_zero():
    s,c=source();c['incremental_cards']['a']['min_gap_difference_m']=-2.
    c['incremental_cards']['b']['min_gap_difference_m']=None;c['candidates'][0]['background_safety_margin_m']=None
    before=copy.deepcopy((s,c));out=build_quantity_facts(s,c,3)
    assert (s,c)==before
    assert out['comparisons']['active_center_gap_CV_increment_m']['values']=={'a':-2.}
    assert out['comparisons']['active_center_gap_CV_increment_m']['missing_or_incomplete']==['b']
    assert out['comparisons']['supplied_background_margin_m']['missing_or_nonnumeric']==['a']
    assert out['candidate_order_changed'] is False and out['additional_acceptance_gate'] is False


def test_ties_use_numeric_tolerance_without_chained_group_drift():
    groups=comparison_groups({'a':0.,'b':.9e-6,'c':1.8e-6},False)
    assert groups==[['a','b'],['c']]


@pytest.mark.parametrize('bad', ['time','actor','candidate','missing_motion'])
def test_bad_enrichment_input_does_not_invent_facts_or_safety_pass(bad):
    s,c=source()
    if bad=='time':c['source_step']=4
    if bad=='actor':s['agents'].append(copy.deepcopy(s['agents'][0]))
    if bad=='candidate':c['candidates'].append(copy.deepcopy(c['candidates'][0]))
    if bad=='missing_motion':s['agents'][0]['state']=None
    out=build_quantity_facts(s,c,3)
    if bad=='missing_motion':assert not out['actor_current_motion'] and out['issues']
    else:assert out['available'] is False
    assert out.get('background_safety_certified') is not True
    json.dumps(out,allow_nan=False)


def test_same_center_radial_rate_is_undefined_not_zero_and_numpy_states_supported():
    import numpy as np
    out=motion_fact(np.array([0,0,0,2,0],dtype=np.float32),np.array([0,0,1,3,1],dtype=np.float32),0,0)
    assert out['radial_gap_rate_mps'] is None and out['current_distance_direction']=='undefined_same_center'
    json.dumps(out,allow_nan=False)

