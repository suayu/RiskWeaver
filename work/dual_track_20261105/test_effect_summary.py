"""分母与配对交集不得因故障或缺失而悄然变成成功。"""
from effect_summary import summarize,paired


def test_fault_excluded_but_normal_no_attack_retained():
    rows=[dict(scenario_index=1,runtime_failure=True),dict(scenario_index=2,collision=False),dict(scenario_index=3,collision=True)]
    report={3:dict(effective_danger_constructed=True)}
    v=summarize([1,2,3,4],rows,report)
    assert (v['effective_denominator'],v['fault_count'],v['unattempted'])==(2,1,[4])
    assert v['effective_danger_rate']==.5 and v['collision_rate']==.5


def test_all_faults_are_undefined_not_perfect():
    v=summarize([1],[dict(scenario_index=1,planner_service_failure_count=1)],{})
    assert v['effective_danger_rate'] is None and v['target_rate_90pct_observed'] is None
    assert v['scores']['scene_challenge_score']['mean'] is None


def test_pair_uses_jointly_nonfault_scenes():
    old=[dict(scenario_index=1),dict(scenario_index=2),dict(scenario_index=3,runtime_failure=True)]
    new=[dict(scenario_index=1),dict(scenario_index=2,runtime_failure=True),dict(scenario_index=3)]
    v=paired(old,new,{}, {1:dict(effective_danger_constructed=True)})
    assert v['common_nonfault_scenes']==[1]
    assert v['mean_coverage_difference']==1
