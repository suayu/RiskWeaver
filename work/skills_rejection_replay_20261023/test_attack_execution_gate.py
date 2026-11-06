"""硬约束保持正面积；无效相对测量不得冒充有效或0危险度。"""
import math
import pytest
from policies.attack_execution_gate import assess_reachability_execution


def event(baseline=187.,actual=224.,difficulty=-.2,solvable=True):
    return dict(original_area_m2=baseline,dangerous_area_m2=actual,difficulty=difficulty,dangerous_solvable=solvable)


def test_scene181_current_future_positive_but_CV_denominator_zero():
    e=event(baseline=0.,actual=224.,difficulty=float('nan'))
    assert not assess_reachability_execution(e)['allowed']
    result=assess_reachability_execution(e,'positive_area')
    assert result['allowed'] and not result['relative_measurement_valid']
    assert result['relative_difficulty'] is None and math.isnan(e['difficulty'])
    assert result['original_area_m2']==0 and result['dangerous_area_m2']==224


@pytest.mark.parametrize('actual',[0.,-1.,float('nan'),float('inf'),None])
def test_actual_nonpositive_or_invalid_area_is_always_rejected(actual):
    result=assess_reachability_execution(event(actual=actual),'positive_area')
    assert not result['allowed']


def test_solvability_cannot_be_bypassed_by_positive_area():
    assert not assess_reachability_execution(event(solvable=False),'positive_area')['allowed']


def test_scene74_route_tail_both_areas_zero_remains_rejected():
    e=event(baseline=0.,actual=0.,difficulty=float('nan'),solvable=False)
    assert not assess_reachability_execution(e,'positive_area')['allowed']


def test_both_modes_accept_existing_valid_event_without_changing_it():
    e=event();before=e.copy()
    assert assess_reachability_execution(e)['allowed'] and assess_reachability_execution(e,'positive_area')['allowed']
    assert e==before


def test_missing_event_and_bad_configuration_fail_closed():
    assert not assess_reachability_execution(None,'positive_area')['allowed']
    with pytest.raises(ValueError):assess_reachability_execution(event(),'unknown')
